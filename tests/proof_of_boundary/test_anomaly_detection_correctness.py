# FIN-C2-105 — Proof-of-Boundary: anomaly detection correctness.
#
# The deterministic detection boundaries of AnomalyAnalyzeNode. Thresholds are
# supplied through the CONSTRUCTOR from the graph's runtime config: BaseNode
# calls execute(state) with one argument, so a config passed to execute() would
# never arrive and every threshold read from it would be dead.

import pytest

from framework.schemas.agent_status import AgentStatus
from src.nodes.anomaly_analyze_node import AnomalyAnalyzeNode
from src.schemas.state import from_json, to_json


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.anomaly_analyze_node.emit_trace_event", lambda *a, **k: None)


def _node(threshold=None):
    config = {"anomaly": {"concentration_threshold": threshold}} if threshold is not None else None
    return AnomalyAnalyzeNode(config=config)


def _run(node, trade_log, metadata=None):
    return node.execute(
        {
            "validated_trade_log": to_json(trade_log),
            "trade_log_metadata": to_json(metadata or {}),
            "validation_errors": [],
        }
    )


def _types(result):
    return {a["type"] for a in from_json(result["detected_anomalies"])}


class TestPositionConcentrationBoundary:
    """The comparison is strict `>`, so equality is not a violation."""

    def test_above_threshold_fires(self):
        assert "POSITION_CONCENTRATION" in _types(_run(_node(0.40), {"positions": {"7203.T": 41, "9984.T": 59}}))

    def test_exactly_at_threshold_does_not_fire(self):
        # 7203.T sits at exactly 0.4000 and no other instrument is above it.
        assert "POSITION_CONCENTRATION" not in _types(
            _run(_node(0.40), {"positions": {"7203.T": 40, "9984.T": 30, "6758.T": 30}})
        )

    def test_below_threshold_does_not_fire(self):
        assert "POSITION_CONCENTRATION" not in _types(
            _run(_node(0.40), {"positions": {"7203.T": 39, "9984.T": 31, "6758.T": 30}})
        )

    def test_severity_escalates_past_sixty_percent(self):
        findings = from_json(_run(_node(0.40), {"positions": {"A": 70, "B": 30}})["detected_anomalies"])
        assert findings[0]["severity"] == "CRITICAL"
        findings = from_json(_run(_node(0.40), {"positions": {"A": 50, "B": 50}})["detected_anomalies"])
        assert findings[0]["severity"] == "HIGH"


class TestCleanLog:
    def test_a_log_within_every_threshold_reports_nothing(self):
        clean = {
            "positions": {"A": 25, "B": 25, "C": 25, "D": 25},
            "confidence_score_buckets": {"high": 10},
            "latency_ms_series": [40, 41, 42, 43],
            "policy_tags": ["OK"],
            "order_count": 4,
            "trades": [{"agent_id": "algo_alpha", "timestamp_sec": 1000.0}],
        }
        result = _run(_node(), clean)
        assert result["anomaly_count"] == 0
        assert result["status"] == AgentStatus.SUCCESS.value


class TestDetectionDependsOnInput:
    """A detector whose output does not move with its input is not a detector."""

    def test_the_same_log_under_two_thresholds_gives_two_answers(self):
        trade_log = {"positions": {"A": 45, "B": 55}}
        assert "POSITION_CONCENTRATION" in _types(_run(_node(0.10), trade_log))
        assert "POSITION_CONCENTRATION" not in _types(_run(_node(0.99), trade_log))

    def test_the_count_scales_with_the_number_of_breaching_instruments(self):
        one = _run(_node(0.40), {"positions": {"A": 60, "B": 40}})
        two = _run(_node(0.10), {"positions": {"A": 60, "B": 40}})
        assert one["anomaly_count"] == 1
        assert two["anomaly_count"] == 2
