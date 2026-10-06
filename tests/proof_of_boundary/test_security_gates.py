# FIN-C2-105 — Proof-of-Boundary: security gates.
#
# S-1  Trust     : the adapter resolves the caller's trust level; ANONYMOUS is refused.
# S-2  Input     : raw financial values do not survive ingestion into any later state.
# S-3  Output    : the domain gate CONTAINS a violating alert — it returns ERROR and
#                  replaces every output-bearing field. Raising is not containment:
#                  AgentBaseGraph.get_output() returns `formatted_output or result`
#                  on the error path too, so an un-cleared field ships the un-gated
#                  answer inside the error envelope.
# S-5  No creds  : no credential-like key appears in any state a node produces.
#
# Audit free functions are muted per node module (never stub shared.* in
# sys.modules — the CI wheel ships a real shared package).

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.ingest_validate_node import IngestValidateNode
from src.nodes.post_process_node import EGRESS_NOTICE, PostProcessNode
from src.nodes.risk_alert_compose_node import CONTAINMENT_NOTICE, RiskAlertComposeNode
from src.schemas.state import from_json, to_json


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    for module in ("ingest_validate_node", "risk_alert_compose_node", "post_process_node"):
        monkeypatch.setattr(f"src.nodes.{module}.emit_trace_event", lambda *a, **k: None)


_CREDENTIAL_KEYS = {"api_key", "password", "token", "secret", "jwt", "credential"}


def _assert_no_credential_keys(value, where):
    if isinstance(value, dict):
        for key, item in value.items():
            assert str(key).lower() not in _CREDENTIAL_KEYS, f"S-5: {key!r} leaked into {where}"
            _assert_no_credential_keys(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_credential_keys(item, f"{where}[{index}]")


PAYLOAD = {
    "trade_log_entries": [
        {
            "order_id": "ORD-0001",
            "timestamp": "2026-06-01T09:00:00Z",
            "instrument_id": "JGB10Y",
            "agent_id": "algo_alpha",
            "order_amount": 12_000_000,
            "price": 99.87,
            "position_delta": -4_000,
            "confidence_score_bucket": "high",
            "latency_ms": 42.0,
            "policy_tags": ["OK"],
        }
    ],
    "baseline": {},
    "thresholds": {},
}


class TestPBS1TrustGate:
    """S-1: the trust level comes from the adapter, and ANONYMOUS is refused."""

    def test_nodes_declare_the_manifest_trust_level(self):
        for node in (IngestValidateNode(), RiskAlertComposeNode(), PostProcessNode()):
            assert node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_an_anonymous_caller_is_denied_before_execute(self):
        result = IngestValidateNode()(
            {"validated_input": json.dumps(PAYLOAD), "caller_trust_level": TrustLevel.ANONYMOUS.value}
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert "validated_trade_log" not in result

    def test_the_declared_level_is_served(self):
        result = IngestValidateNode()(
            {"validated_input": json.dumps(PAYLOAD), "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.SUCCESS.value


class TestPBS2Sanitisation:
    """S-2: raw financial values are dropped at ingestion and never travel on."""

    def test_raw_values_do_not_survive_ingestion(self):
        result = IngestValidateNode().execute({"validated_input": json.dumps(PAYLOAD)})
        serialised = json.dumps(result, default=str)
        for forbidden in ("order_amount", "position_delta", "12000000", "99.87"):
            assert forbidden not in serialised

    def test_only_declared_metadata_survives(self):
        detection = from_json(
            IngestValidateNode().execute({"validated_input": json.dumps(PAYLOAD)})["validated_trade_log"]
        )
        assert set(detection) == {
            "positions",
            "confidence_score_buckets",
            "latency_ms_series",
            "policy_tags",
            "order_count",
            "trades",
        }


class TestPBS3DomainOutputGate:
    """S-3 (domain layer): a violating alert is contained, not raised."""

    def test_a_forbidden_key_contains_the_alert(self):
        alert = {"overall_severity": "HIGH", "order_amount": 5_000_000}
        contained = RiskAlertComposeNode()._extra_security_gate_output(
            {"risk_alert": to_json(alert), "formatted_output": json.dumps(alert)}
        )
        assert contained["status"] == AgentStatus.ERROR.value
        assert contained["formatted_output"] == CONTAINMENT_NOTICE
        assert contained["result"] is None
        assert contained["risk_alert"] is None

    def test_a_forbidden_key_nested_in_a_finding_is_caught(self):
        alert = {"overall_severity": "HIGH", "anomalies": [{"anomaly_type": "X", "price": 2950.0}]}
        contained = RiskAlertComposeNode()._extra_security_gate_output(
            {"risk_alert": to_json(alert), "formatted_output": json.dumps(alert)}
        )
        assert contained["status"] == AgentStatus.ERROR.value

    def test_the_replacement_is_truthy(self):
        # A falsy formatted_output re-opens the `formatted_output or result`
        # fallback onto the very state the gate just refused.
        assert CONTAINMENT_NOTICE and EGRESS_NOTICE

    def test_a_clean_alert_is_released(self):
        node = RiskAlertComposeNode()
        result = node.execute(
            {
                "detected_anomalies": to_json([]),
                "anomaly_count": 0,
                "trade_log_metadata": to_json({"agent_id": "algo_alpha"}),
            }
        )
        assert node._extra_security_gate_output(result) is result

    def test_the_hook_honours_the_framework_contract(self):
        # It must RETURN a dict. Returning None makes BaseNode.__call__ raise on
        # every invocation, which is how this node shipped without ever having
        # produced a single alert.
        node = RiskAlertComposeNode()
        state = {
            "detected_anomalies": to_json([]),
            "anomaly_count": 0,
            "trade_log_metadata": to_json({}),
            "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        }
        assert node(state)["status"] == AgentStatus.SUCCESS.value


class TestPBS3EgressGate:
    """S-3 (egress layer): the caller boundary owns the credential invariant."""

    @pytest.mark.parametrize(
        "secret",
        [
            "Bearer abcdefghij0123456789",
            "AKIA0123456789ABCDEF",
            "password=hunter2hunter2",
        ],
    )
    def test_a_credential_is_withheld_at_the_boundary(self, secret):
        leaky = json.dumps({"overall_severity": "HIGH", "note": secret})
        result = PostProcessNode().execute({"formatted_output": leaky})
        assert result["status"] == AgentStatus.ERROR.value
        assert result["result"] == EGRESS_NOTICE
        assert secret not in json.dumps(result, default=str)


class TestPBS5NoCredentialsInState:
    """S-5: nothing a node writes may look like a credential store."""

    def test_no_credential_keys_in_any_node_result(self):
        ingest = IngestValidateNode().execute({"validated_input": json.dumps(PAYLOAD)})
        _assert_no_credential_keys(ingest, "IngestValidateNode")
        compose = RiskAlertComposeNode().execute(
            {
                "detected_anomalies": to_json([]),
                "anomaly_count": 0,
                "trade_log_metadata": ingest["trade_log_metadata"],
            }
        )
        _assert_no_credential_keys(compose, "RiskAlertComposeNode")
        _assert_no_credential_keys(
            PostProcessNode().execute({"formatted_output": compose["formatted_output"]}),
            "PostProcessNode",
        )
