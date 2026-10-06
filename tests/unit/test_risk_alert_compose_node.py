# FIN-C2-105 — RiskAlertComposeNode (composition + the domain output gate).
#
# The gate is exercised through __call__ as well as directly, because the
# previous implementation was only ever tested through execute(): it declared
# `_extra_security_gate_output(...) -> None`, the framework contract requires it
# to RETURN the result dict, and the resulting None made every single
# invocation fail with an AttributeError. execute()-only tests stayed green.

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.risk_alert_compose_node import CONTAINMENT_NOTICE, RiskAlertComposeNode
from src.schemas.state import from_json, to_json
from src.services.service import MASK_SENTINEL

ANOMALIES = [
    {
        "type": "POSITION_CONCENTRATION",
        "severity": "CRITICAL",
        "evidence": "instrument=JGB10Y weight_fraction=0.6667 threshold=0.4000",
        "threshold": 0.4,
    },
    {
        "type": "LATENCY_SPIKE",
        "severity": "HIGH",
        "evidence": "spike_count=1 rolling_median_ms=42.00",
        "threshold": 126.0,
    },
]
METADATA = {
    "period": "2026-06-01T09:00:00Z / 2026-06-01T09:00:02Z",
    "agent_id": "algo_alpha",
    "instrument_count": 2,
    "entry_count": 3,
    "desk_code": "tokyo_rates",
}


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.risk_alert_compose_node.emit_trace_event", lambda *a, **k: None)


def _state(anomalies=ANOMALIES, metadata=METADATA):
    return {
        "detected_anomalies": to_json(anomalies),
        "anomaly_count": len(anomalies),
        "trade_log_metadata": to_json(metadata),
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
    }


class TestComposition:
    def test_alert_is_produced_with_the_highest_severity(self):
        alert = from_json(RiskAlertComposeNode().execute(_state())["risk_alert"])
        assert alert["overall_severity"] == "CRITICAL"
        assert alert["affected_agent_id"] == "algo_alpha"

    def test_each_finding_carries_its_own_evidence(self):
        # description used to be read from a key the detector never emits, so
        # every published finding shipped with an empty description.
        alert = from_json(RiskAlertComposeNode().execute(_state())["risk_alert"])
        descriptions = [a["description"] for a in alert["anomalies"]]
        assert all(descriptions)
        assert "instrument=JGB10Y" in descriptions[0]

    def test_no_findings_produces_a_low_severity_clean_report(self):
        alert = from_json(RiskAlertComposeNode().execute(_state(anomalies=[]))["risk_alert"])
        assert alert["overall_severity"] == "LOW"
        assert alert["anomalies"] == []

    def test_provenance_is_quoted_only_when_supplied(self):
        with_desk = from_json(RiskAlertComposeNode().execute(_state())["risk_alert"])
        without = from_json(
            RiskAlertComposeNode().execute(_state(metadata={k: v for k, v in METADATA.items() if k != "desk_code"}))[
                "risk_alert"
            ]
        )
        assert with_desk["desk_code"] == "tokyo_rates"
        assert "desk_code" not in without

    def test_audit_trail_id_is_deterministic(self):
        first = RiskAlertComposeNode().execute(_state())["risk_alert"]
        second = RiskAlertComposeNode().execute(_state())["risk_alert"]
        assert from_json(first)["audit_trail_id"] == from_json(second)["audit_trail_id"]

    def test_advisory_is_always_present(self):
        alert = from_json(RiskAlertComposeNode().execute(_state())["risk_alert"])
        assert "コンプライアンス" in alert["advisory"]


class TestGateContract:
    def test_the_hook_returns_a_dict(self):
        # Returning None makes BaseNode.__call__ raise
        # "AttributeError: 'NoneType' object has no attribute 'setdefault'"
        # on EVERY invocation — the defect this template shipped with.
        node = RiskAlertComposeNode()
        result = node.execute(_state())
        assert isinstance(node._extra_security_gate_output(result), dict)

    def test_full_node_call_succeeds_end_to_end(self):
        result = RiskAlertComposeNode()(_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"]
        assert "Traceback" not in json.dumps(result, default=str)


class TestOutputInvariant:
    @pytest.mark.parametrize(
        "alert",
        [
            {"overall_severity": "HIGH", "order_amount": 5_000_000},
            {"overall_severity": "HIGH", "anomalies": [{"anomaly_type": "X", "price": 2950.0}]},
            {"overall_severity": "HIGH", "anomalies": [{"nested": {"position_delta": -1}}]},
        ],
    )
    def test_a_raw_financial_key_anywhere_is_contained(self, alert):
        contained = RiskAlertComposeNode()._extra_security_gate_output(
            {"risk_alert": to_json(alert), "formatted_output": json.dumps(alert)}
        )
        assert contained["status"] == AgentStatus.ERROR.value
        assert contained["risk_alert"] is None
        assert contained["result"] is None
        assert contained["formatted_output"] == CONTAINMENT_NOTICE

    def test_the_replacement_notice_is_truthy(self):
        # A falsy formatted_output re-opens AgentBaseGraph.get_output()'s
        # `formatted_output or result` fallback onto un-gated state.
        assert CONTAINMENT_NOTICE

    def test_the_containment_notice_quotes_nothing_from_the_alert(self):
        alert = {"overall_severity": "HIGH", "order_amount": 5_000_000, "affected_agent_id": "algo_x"}
        contained = RiskAlertComposeNode()._extra_security_gate_output(
            {"risk_alert": to_json(alert), "formatted_output": json.dumps(alert)}
        )
        blob = json.dumps(contained, default=str)
        assert "5000000" not in blob
        assert "algo_x" not in blob

    def test_redaction_sentinel_reported_as_a_finding_is_contained(self):
        alert = {"overall_severity": "HIGH", "affected_agent_id": MASK_SENTINEL}
        contained = RiskAlertComposeNode()._extra_security_gate_output(
            {"risk_alert": to_json(alert), "formatted_output": json.dumps(alert)}
        )
        assert contained["status"] == AgentStatus.ERROR.value

    def test_clean_alert_passes_through_unchanged(self):
        node = RiskAlertComposeNode()
        result = node.execute(_state())
        assert node._extra_security_gate_output(result) is result

    def test_empty_result_is_not_treated_as_a_violation(self):
        assert RiskAlertComposeNode()._extra_security_gate_output({}) == {}
