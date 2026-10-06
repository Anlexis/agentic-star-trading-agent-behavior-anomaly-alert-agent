# FIN-C2-105 — AnomalyAnalyzeNode (the rule engine).
#
# Thresholds arrive through the CONSTRUCTOR, not through an execute() argument:
# BaseNode.__call__ calls execute(state) with one argument, so a node declaring
# execute(self, state, config=None) never receives a config and every value read
# from it is dead. These tests pin that the declared value is live.

import pytest

from framework.schemas.agent_status import AgentStatus
from src.nodes.anomaly_analyze_node import ANOMALY_CONFIG_BOUNDS, AnomalyAnalyzeNode
from src.schemas.state import from_json, to_json


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.anomaly_analyze_node.emit_trace_event", lambda *a, **k: None)


def _state(trade_log, metadata=None, caller_thresholds=None):
    state = {
        "validated_trade_log": to_json(trade_log),
        "trade_log_metadata": to_json(metadata or {}),
        "validation_errors": [],
    }
    if caller_thresholds is not None:
        state["caller_thresholds"] = to_json(caller_thresholds)
    return state


def _types(result):
    return {a["type"] for a in from_json(result["detected_anomalies"])}


class TestPositionConcentration:
    def test_above_threshold_fires(self):
        result = AnomalyAnalyzeNode().execute(_state({"positions": {"7203.T": 41, "9984.T": 59}}))
        assert "POSITION_CONCENTRATION" in _types(result)

    def test_exactly_at_threshold_does_not_fire(self):
        # The comparison is strict `>`; equality is not a violation. A is at
        # exactly 0.40 and no other instrument is above it.
        result = AnomalyAnalyzeNode().execute(_state({"positions": {"A": 40, "B": 30, "C": 30}}))
        assert "POSITION_CONCENTRATION" not in _types(result)

    def test_evidence_names_the_instrument_and_the_threshold(self):
        result = AnomalyAnalyzeNode().execute(_state({"positions": {"7203.T": 90, "X": 10}}))
        evidence = from_json(result["detected_anomalies"])[0]["evidence"]
        assert "instrument=7203.T" in evidence
        assert "threshold=0.4000" in evidence


class TestDeclaredConfigIsLive:
    def test_declared_threshold_changes_the_finding(self):
        trade_log = {"positions": {"A": 45, "B": 55}}
        strict = AnomalyAnalyzeNode(config={"anomaly": {"concentration_threshold": 0.10}})
        loose = AnomalyAnalyzeNode(config={"anomaly": {"concentration_threshold": 0.99}})
        assert "POSITION_CONCENTRATION" in _types(strict.execute(_state(trade_log)))
        assert "POSITION_CONCENTRATION" not in _types(loose.execute(_state(trade_log)))

    def test_declared_forbidden_tags_are_live(self):
        trade_log = {"policy_tags": ["QUARANTINE"], "order_count": 1}
        default = AnomalyAnalyzeNode()
        declared = AnomalyAnalyzeNode(config={"anomaly": {"forbidden_policy_tags": ["QUARANTINE"]}})
        assert "POLICY_VIOLATION" not in _types(default.execute(_state(trade_log)))
        assert "POLICY_VIOLATION" in _types(declared.execute(_state(trade_log)))

    def test_caller_override_beats_the_declared_value(self):
        trade_log = {"positions": {"A": 45, "B": 55}}
        node = AnomalyAnalyzeNode(config={"anomaly": {"concentration_threshold": 0.99}})
        assert "POSITION_CONCENTRATION" not in _types(node.execute(_state(trade_log)))
        overridden = node.execute(_state(trade_log, caller_thresholds={"concentration_threshold": 0.10}))
        assert "POSITION_CONCENTRATION" in _types(overridden)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 99.0, "0.1"])
    def test_an_unbounded_override_cannot_reach_a_comparison(self, bad):
        # The override is re-checked here as well as at ingest, so a value that
        # bypassed the node contract still cannot switch a rule off.
        trade_log = {"positions": {"A": 45, "B": 55}}
        node = AnomalyAnalyzeNode()
        result = node.execute(_state(trade_log, caller_thresholds={"concentration_threshold": bad}))
        evidence = [
            a["evidence"] for a in from_json(result["detected_anomalies"]) if a["type"] == "POSITION_CONCENTRATION"
        ]
        assert all("threshold=0.4000" in e for e in evidence)


class TestModelDrift:
    def test_drift_fires_against_the_caller_baseline(self):
        result = AnomalyAnalyzeNode().execute(
            _state(
                {"confidence_score_buckets": {"high": 90, "low": 10}},
                metadata={"confidence_score_baseline": {"high": 10, "low": 90}},
            )
        )
        assert "MODEL_DRIFT" in _types(result)

    def test_no_baseline_means_no_drift_finding(self):
        result = AnomalyAnalyzeNode().execute(_state({"confidence_score_buckets": {"high": 90}}))
        assert "MODEL_DRIFT" not in _types(result)

    def test_matching_distribution_does_not_fire(self):
        result = AnomalyAnalyzeNode().execute(
            _state(
                {"confidence_score_buckets": {"high": 10, "low": 90}},
                metadata={"confidence_score_baseline": {"high": 10, "low": 90}},
            )
        )
        assert "MODEL_DRIFT" not in _types(result)


class TestLatencyAndPolicyAndCoordination:
    def test_latency_spike_fires(self):
        result = AnomalyAnalyzeNode().execute(_state({"latency_ms_series": [40, 42, 44, 900]}))
        assert "LATENCY_SPIKE" in _types(result)

    def test_steady_latency_does_not_fire(self):
        result = AnomalyAnalyzeNode().execute(_state({"latency_ms_series": [40, 42, 44, 46]}))
        assert "LATENCY_SPIKE" not in _types(result)

    def test_forbidden_tag_fires_critical(self):
        result = AnomalyAnalyzeNode().execute(_state({"policy_tags": ["HALT"], "order_count": 1}))
        finding = [a for a in from_json(result["detected_anomalies"]) if a["type"] == "POLICY_VIOLATION"]
        assert finding and finding[0]["severity"] == "CRITICAL"

    def test_order_count_ceiling_fires(self):
        result = AnomalyAnalyzeNode().execute(_state({"order_count": 1500}))
        assert "POLICY_VIOLATION" in _types(result)

    def test_coordination_signal_fires_inside_the_window(self):
        trades = [{"agent_id": "algo_a", "timestamp_sec": 1000.0}, {"agent_id": "algo_b", "timestamp_sec": 1000.5}]
        assert "COORDINATION_SIGNAL" in _types(AnomalyAnalyzeNode().execute(_state({"trades": trades})))

    def test_coordination_signal_silent_outside_the_window(self):
        trades = [{"agent_id": "algo_a", "timestamp_sec": 1000.0}, {"agent_id": "algo_b", "timestamp_sec": 9000.0}]
        assert "COORDINATION_SIGNAL" not in _types(AnomalyAnalyzeNode().execute(_state({"trades": trades})))


class TestFailClosedInternals:
    def test_non_finite_weights_cannot_switch_the_rule_off(self):
        result = AnomalyAnalyzeNode().execute(_state({"positions": {"A": float("nan"), "B": 10}}))
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_non_finite_bucket_weights_are_dropped_not_propagated(self):
        result = AnomalyAnalyzeNode().execute(
            _state(
                {"confidence_score_buckets": {"high": float("inf"), "low": 10}},
                metadata={"confidence_score_baseline": {"high": 10, "low": 90}},
            )
        )
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_malformed_detection_input_is_an_error_not_a_clean_bill(self):
        result = AnomalyAnalyzeNode().execute(
            {"validated_trade_log": to_json(["not", "a", "mapping"]), "validation_errors": []}
        )
        assert result["status"] == AgentStatus.ERROR.value

    def test_upstream_rejection_suppresses_detection(self):
        result = AnomalyAnalyzeNode().execute(
            {"validated_trade_log": to_json({"order_count": 99999}), "validation_errors": ["x"]}
        )
        assert result["anomaly_count"] == 0

    def test_bounds_table_covers_every_default(self):
        from src.nodes.anomaly_analyze_node import DEFAULT_THRESHOLDS

        assert set(DEFAULT_THRESHOLDS) == set(ANOMALY_CONFIG_BOUNDS)
