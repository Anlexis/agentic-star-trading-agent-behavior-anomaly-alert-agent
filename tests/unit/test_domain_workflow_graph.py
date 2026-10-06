# FIN-C2-105 — graph wiring, runtime config, and the caller-context bridge.

import pytest

from framework.errors import ConfigError
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import caller_context, current_caller_context, stash_caller_context
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import Graph, TradingAnomalyAlertAgent, TradingAnomalyAlertGraphNode
from src.schemas.state import State
from src.services.service import REFUSAL_NOTICE


class TestInnerGraph:
    def test_topology_is_the_declared_pipeline(self):
        graph = DomainWorkflowGraph()
        graph.register_nodes()
        assert list(graph._nodes) == ["ingest_validate", "anomaly_analyze", "risk_alert_compose"]

    def test_backbone_nodes_are_not_registered_here(self):
        graph = DomainWorkflowGraph()
        graph.register_nodes()
        assert "initialize" not in graph._nodes
        assert "finalize" not in graph._nodes

    def test_state_schema_is_shared_with_the_outer_graph(self):
        assert DomainWorkflowGraph().state_schema is State
        assert TradingAnomalyAlertAgent().state_schema is State

    def test_it_compiles(self):
        graph = DomainWorkflowGraph()
        graph.compile()
        assert graph._compiled is not None

    def test_output_is_withheld_on_a_non_success_status(self):
        graph = DomainWorkflowGraph()
        held = graph.get_output({"status": AgentStatus.ERROR.value, "formatted_output": "partial alert"})
        assert held["output"] is None

    def test_output_is_returned_on_success(self):
        graph = DomainWorkflowGraph()
        released = graph.get_output({"status": AgentStatus.SUCCESS.value, "formatted_output": "alert"})
        assert released["output"] == "alert"


class TestConfigValidation:
    def test_a_malformed_declared_threshold_stops_the_graph(self):
        # A threshold that silently reverts to a default is a detection rule that
        # silently stops firing, so this must fail rather than degrade.
        with pytest.raises(ConfigError):
            DomainWorkflowGraph(config={"anomaly": {"concentration_threshold": 5.0}}).compile()

    def test_a_non_numeric_declared_threshold_stops_the_graph(self):
        with pytest.raises(ConfigError):
            DomainWorkflowGraph(config={"anomaly": {"concentration_threshold": "high"}}).compile()

    def test_a_boolean_is_not_accepted_as_a_number(self):
        with pytest.raises(ConfigError):
            DomainWorkflowGraph(config={"anomaly": {"latency_spike_multiplier": True}}).compile()

    def test_bad_forbidden_tag_list_stops_the_graph(self):
        with pytest.raises(ConfigError):
            DomainWorkflowGraph(config={"anomaly": {"forbidden_policy_tags": "HALT"}}).compile()

    def test_a_valid_declared_block_compiles(self):
        DomainWorkflowGraph(config={"anomaly": {"concentration_threshold": 0.2}}).compile()

    def test_the_shipped_config_file_is_loaded_when_none_is_passed(self):
        # A graph constructed bare must still run on the DECLARED values,
        # otherwise config/config.yaml documents behaviour the agent lacks.
        agent = TradingAnomalyAlertAgent()
        assert agent.config.get("anomaly", {}).get("concentration_threshold") == 0.40
        assert agent.config.get("limits", {}).get("max_trade_log_entries") == 2000

    def test_an_explicit_config_wins_over_the_file(self):
        agent = TradingAnomalyAlertAgent(config={"max_retry": 1})
        assert agent.config == {"max_retry": 1}

    def test_config_reaches_the_inner_graph(self):
        agent = TradingAnomalyAlertAgent(config={"anomaly": {"concentration_threshold": 0.11}})
        agent.register_nodes()
        inner = agent._nodes["main"].get_subgraph()
        assert inner.config["anomaly"]["concentration_threshold"] == 0.11


class TestOuterGraph:
    def test_backbone_slots_are_filled(self):
        agent = TradingAnomalyAlertAgent()
        agent.register_nodes()
        assert set(agent._nodes) == {"initialize", "pre_process", "main", "post_process", "finalize"}
        assert isinstance(agent._nodes["main"], TradingAnomalyAlertGraphNode)

    def test_alias_and_name(self):
        assert Graph is TradingAnomalyAlertAgent
        assert TradingAnomalyAlertAgent().name == "TradingAnomalyAlertAgent"

    def test_manifest_entry_point_resolves_to_the_implemented_class(self):
        # config/agent.yaml used to name a class that does not exist, so
        # AgentRegistry auto-discovery would fail with AttributeError.
        import importlib
        import pathlib
        import re

        manifest = pathlib.Path(__file__).resolve().parents[2] / "config" / "agent.yaml"
        dotted = re.search(r'^class:\s*"([^"]+)"', manifest.read_text(), re.M).group(1)
        module_path, _, class_name = dotted.rpartition(".")
        assert getattr(importlib.import_module(module_path), class_name) is TradingAnomalyAlertAgent

    def test_merge_output_returns_only_changed_keys(self):
        node = TradingAnomalyAlertGraphNode()
        merged = node.merge_output({}, {"output": "alert", "status": AgentStatus.SUCCESS.value})
        assert merged == {"formatted_output": "alert", "status": AgentStatus.SUCCESS.value}


class TestContextBridge:
    def test_extract_input_stashes_the_caller_context(self):
        # GraphNode.execute() calls subgraph.invoke() with no input_context, so
        # without the stash the inner graph starts with an empty context.
        node = TradingAnomalyAlertGraphNode()
        node.extract_input({"validated_input": "payload", "input_context": {"desk_code": "tokyo"}})
        assert current_caller_context() == {"desk_code": "tokyo"}

    def test_extract_input_clears_a_previous_request(self):
        stash_caller_context({"desk_code": "stale"})
        node = TradingAnomalyAlertGraphNode()
        node.extract_input({"validated_input": "payload"})
        assert current_caller_context() == {}

    def test_extract_input_prefers_validated_input(self):
        node = TradingAnomalyAlertGraphNode()
        assert node.extract_input({"validated_input": "v", "user_input": "u"}) == "v"
        assert node.extract_input({"user_input": "u"}) == "u"

    def test_inner_graph_seeds_the_context_from_the_bridge(self):
        with caller_context({"desk_code": "osaka"}):
            assert DomainWorkflowGraph()._extra_initial_state() == {"input_context": {"desk_code": "osaka"}}


class TestSubgraphContainment:
    def test_an_inner_failure_is_replaced_by_the_closed_set_notice(self):
        node = TradingAnomalyAlertGraphNode()
        contained = node.on_subgraph_error({}, RuntimeError("inner detail /src/nodes/x.py line 42"))
        assert contained["status"] == AgentStatus.ERROR.value
        assert contained["formatted_output"] == REFUSAL_NOTICE
        assert contained["result"] is None
        assert "inner detail" not in str(contained)
        assert "/src/" not in str(contained)

    def test_the_notice_is_truthy(self):
        assert REFUSAL_NOTICE

    def test_error_strategy_is_handle_so_a_rejection_is_not_a_crash(self):
        assert TradingAnomalyAlertGraphNode.error_strategy == "handle"
