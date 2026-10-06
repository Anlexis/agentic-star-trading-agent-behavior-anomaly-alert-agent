"""AgentCore Platform v1.0"""

# FIN-C2-105 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed — identical to Cat 1, do NOT override add_edges()):
#     START → initialize → pre_process → main → {route} → post_process → finalize → END
#                                             ↓ (RETRY, max_retry from config/config.yaml)
#                                          pre_process
#
#   The `main` slot is a GraphNode subclass (TradingAnomalyAlertGraphNode) that
#   delegates the whole domain workflow to DomainWorkflowGraph (inner BaseGraph).
#
# Directory layout:
#   src/graph/graph.py                 ← outer graph (this file)
#   src/graph/domain_workflow_graph.py ← inner graph (multi-step topology)
#   src/graph/context_bridge.py        ← caller-context bridge across the two layers
#
# Rules enforced:
#   ✅ TradingAnomalyAlertAgent inherits AgentBaseGraph (direct framework inheritance)
#   ✅ super().register_nodes() called first (fills initialize + finalize)
#   ✅ TradingAnomalyAlertGraphNode assigned to self._nodes["main"]
#   ✅ merge_output() returns only changed keys
#   ❌ add_edges() NOT overridden on the outer graph
#   ❌ No platform SDK imports

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.utils.audit_logger import emit_trace_event
from framework.utils.config_loader import load_agent_config

from src.graph.context_bridge import stash_caller_context
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State
from src.services.service import REFUSAL_NOTICE

# Repository root — src/graph/graph.py → src/graph → src → <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]


class TradingAnomalyAlertGraphNode(GraphNode):
    """GraphNode assigned to the `main` slot of TradingAnomalyAlertAgent.

    Wraps DomainWorkflowGraph (the inner Cat 2 BaseGraph).

    Contracts:
      get_subgraph()       — instantiate the inner graph with the SAME runtime config
      extract_input()      — hand the validated payload to the inner graph and stash
                             the caller context so the inner graph can see it
      merge_output()       — map sub_result fields into the outer state delta
      on_subgraph_error()  — contain: closed-set notice, output-bearing fields cleared
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    # "handle" rather than "propagate": a rejected trade log is an expected
    # outcome, not a crash. Propagating turns it into a SubgraphError whose
    # string form is written into error_log together with a traceback, and the
    # caller learns nothing actionable. on_subgraph_error() below converts every
    # inner failure into one closed-set notice instead.
    error_strategy: ClassVar[str] = "handle"

    # False: HITL interrupts are handled inside the inner graph only.
    propagate_hitl: ClassVar[bool] = False

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._config: Dict[str, Any] = dict(config or {})

    def get_subgraph(self) -> Any:
        """Instantiate the inner domain workflow graph with the outer runtime config.

        Imported lazily to avoid a circular import at module load time.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._config)

    def extract_input(self, state: AgentState) -> str:
        """Return the payload passed into the inner graph, and bridge the context.

        GraphNode.execute() calls ``subgraph.invoke(user_input, ...)`` without an
        ``input_context`` argument, so the inner graph would otherwise start with
        an empty context. The stash happens on EVERY call (empty context included)
        so one request can never observe another's value.
        """
        stash_caller_context(state.get("input_context") or {})
        return str(state.get("validated_input") or state.get("user_input") or "")

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner sub_result back into the outer state delta (changed keys only).

        Inner get_output() emits: output / status / trace_id / correlation_id / node_history.
        """
        return {
            "formatted_output": sub_result.get("output"),
            "status": sub_result.get("status"),
        }

    def on_subgraph_error(self, state: AgentState, error: Exception) -> Dict[str, Any]:
        """Contain an inner-graph failure at the subgraph boundary.

        Returns ERROR *and* replaces every output-bearing field. Raising here (or
        returning ERROR while leaving the fields alone) is not containment:
        AgentBaseGraph.get_output() falls back to ``state.get("result")`` even on
        the error path, so an un-cleared field ships the un-gated inner answer
        inside the error envelope.

        The notice is a fixed closed-set string — never ``str(error)``, which on
        this framework carries the inner error_log and a traceback.
        """
        emit_trace_event(
            "domain_workflow_contained",
            {"reason": "subgraph_error", "error_type": type(error).__name__},
            state,
        )
        return {
            "status": AgentStatus.ERROR.value,
            "formatted_output": REFUSAL_NOTICE,
            "result": None,
            "risk_alert": None,
            "detected_anomalies": None,
            "alert_severity": None,
        }


class TradingAnomalyAlertAgent(AgentBaseGraph):
    """Outer graph for FIN-C2-105 (Cat 2).

    Inherits AgentBaseGraph directly (L1 Base). Domain logic is encapsulated in
    TradingAnomalyAlertGraphNode (main slot), which delegates to
    DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed): START → initialize → pre_process → main → post_process → finalize → END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills initialize + finalize (framework defaults)
      - pre_process : PreProcessNode  (caller-contract validation)
      - main        : TradingAnomalyAlertGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output containment gate)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load config/config.yaml when the caller supplies no config.

        AgentRegistry passes the parsed runtime config; a standalone entry point
        may construct the graph bare. Loading the shipped file in that case is
        what keeps the DECLARED values live on both paths — without it every
        value in config/config.yaml is inert on the standalone path and the file
        documents behaviour the agent does not have.
        """
        super().__init__(config if config is not None else load_agent_config(_REPO_ROOT))

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "TradingAnomalyAlertAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the framework's
        default InitializeNode (schema_version, session_id, trust_level) and
        FinalizeNode (response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode(config=self.config)
        self._nodes["main"] = TradingAnomalyAlertGraphNode(config=self.config)
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Back-compat alias — callers may reference either name.
Graph = TradingAnomalyAlertAgent

__all__ = [
    "Graph",
    "REFUSAL_NOTICE",
    "TradingAnomalyAlertAgent",
    "TradingAnomalyAlertGraphNode",
]
