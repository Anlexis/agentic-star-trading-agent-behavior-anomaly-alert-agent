"""AgentCore Platform v1.0"""

# FIN-C2-105 — DomainWorkflowGraph (inner BaseGraph)
#
# The INNER graph of the Cat 2 two-layer nested architecture. It encapsulates the
# whole trading-anomaly detection workflow:
#
#   START → ingest_validate → anomaly_analyze → risk_alert_compose → END
#
# Called by TradingAnomalyAlertGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   ✅ Inherits BaseGraph (fully custom topology — no forced backbone)
#   ✅ Implements all 7 BaseGraph ABC methods
#   ✅ register_nodes() does NOT call super() (abstract in BaseGraph)
#   ✅ Does NOT register initialize / finalize (outer backbone concerns)
#   ✅ get_output() designed together with TradingAnomalyAlertGraphNode.merge_output()
#   ❌ No platform SDK imports

from typing import Any, Dict, Optional

from langgraph.graph import END, START

from framework.errors import ConfigError
from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus

from src.graph.context_bridge import current_caller_context
from src.nodes.anomaly_analyze_node import ANOMALY_CONFIG_BOUNDS, AnomalyAnalyzeNode
from src.nodes.ingest_validate_node import IngestValidateNode
from src.nodes.risk_alert_compose_node import RiskAlertComposeNode
from src.schemas.state import State


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for FIN-C2-105.

    Pipeline (linear):
        START → ingest_validate → anomaly_analyze → risk_alert_compose → END

    All three nodes are FunctionNode subclasses returning partial-dict state
    updates. initialize / finalize are outer backbone concerns and are not
    registered here.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config or {})

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "trading_anomaly_detection_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Validate the declared detection thresholds before compilation.

        These values come from config/config.yaml and change what the agent
        detects, so a malformed one must stop the graph rather than degrade
        silently to a default — a threshold that quietly reverts is a detection
        rule that quietly stops firing.
        """
        anomaly = self.config.get("anomaly")
        if anomaly is None:
            return
        if not isinstance(anomaly, dict):
            raise ConfigError(f"[{self.__class__.__name__}] 'anomaly' must be a mapping, got: {type(anomaly).__name__}")
        for key, (low, high) in ANOMALY_CONFIG_BOUNDS.items():
            if key not in anomaly:
                continue
            value = anomaly[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"[{self.__class__.__name__}] 'anomaly.{key}' must be a number")
            if not (low <= float(value) <= high):
                raise ConfigError(f"[{self.__class__.__name__}] 'anomaly.{key}' must be within [{low}, {high}]")
        tags = anomaly.get("forbidden_policy_tags")
        if tags is not None and not (isinstance(tags, list) and all(isinstance(t, str) for t in tags)):
            raise ConfigError(f"[{self.__class__.__name__}] 'anomaly.forbidden_policy_tags' must be a list of strings")

    # ── Initial state ─────────────────────────────────────────────────────────

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the inner initial state with the caller context from the outer layer.

        GraphNode.execute() calls ``subgraph.invoke(user_input, ...)`` with no
        ``input_context`` argument, so without this hook the inner graph starts
        with an empty context and every inner node reading it sees nothing.
        The payload itself arrives as the inner ``user_input``.
        """
        return {"input_context": current_caller_context()}

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register the 3 domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract. Do NOT register
        initialize or finalize; those are outer backbone concerns handled by
        AgentBaseGraph in graph.py. Every key registered here is referenced in
        add_edges().
        """
        self._nodes["ingest_validate"] = IngestValidateNode(config=self.config)
        self._nodes["anomaly_analyze"] = AnomalyAnalyzeNode(config=self.config)
        self._nodes["risk_alert_compose"] = RiskAlertComposeNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear trading-anomaly detection topology.

        Intentionally linear — no conditional branching between domain nodes.
        route() is implemented to satisfy the ABC, and add_conditional_edges() is
        deliberately not used, so no path callable's annotation can project state
        fields away.
        """
        self._sg.add_edge(START, "ingest_validate")
        self._sg.add_edge("ingest_validate", "anomaly_analyze")
        self._sg.add_edge("anomaly_analyze", "risk_alert_compose")
        self._sg.add_edge("risk_alert_compose", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: AgentState) -> str:
        """Required by the BaseGraph ABC; never called for this linear topology.

        Returns END so an unexpected call cannot re-enter a processing node.
        """
        return END

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the sub_result dict returned to the outer graph.

            Inner get_output()   emits: "output", "status", "trace_id",
                                        "correlation_id", "node_history"
            Outer merge_output() reads: sub_result["output"] → outer formatted_output
                                        sub_result["status"]

        "output" carries the formatted alert report produced by
        RiskAlertComposeNode. On any non-success status it is withheld: the outer
        layer's on_subgraph_error() then substitutes the closed-set refusal
        notice, so a partially built alert can never ride out on an error path.
        """
        status = state.get("status")
        succeeded = status in (AgentStatus.SUCCESS, AgentStatus.SUCCESS.value)
        return {
            "output": state.get("formatted_output") if succeeded else None,
            "status": status,
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
