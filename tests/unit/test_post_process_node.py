# FIN-C2-105 — PostProcessNode (the egress gate).
#
# This is the last code that runs before the caller sees anything, so it owns the
# credential invariant. The domain invariant lives one layer in, in
# RiskAlertComposeNode: the two layers check DIFFERENT things on purpose, so
# neither can contain the other's mutant and make the other's test decorative.

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from src.nodes.post_process_node import EGRESS_NOTICE, MAX_RELEASED_CHARS, PostProcessNode

CLEAN_ALERT = json.dumps(
    {
        "overall_severity": "CRITICAL",
        "affected_agent_id": "algo_alpha",
        "anomalies": [{"anomaly_type": "LATENCY_SPIKE", "description": "spike_count=1"}],
    }
)


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)


class TestRelease:
    def test_a_clean_alert_is_published(self):
        result = PostProcessNode().execute({"formatted_output": CLEAN_ALERT})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["result"] == CLEAN_ALERT

    def test_absent_output_is_not_an_error(self):
        assert PostProcessNode().execute({})["status"] == AgentStatus.SUCCESS.value


class TestEgressGate:
    @pytest.mark.parametrize(
        "secret",
        [
            "Bearer abcdefghij0123456789",
            "AKIA0123456789ABCDEF",
            "sk-0123456789abcdefghij0123",
            "eyJhbGciOiJIUzI1NiJ9.payload",
            "postgresql://user:example@host/db",
        ],
    )
    def test_framework_detected_credentials_are_withheld(self, secret):
        leaky = json.dumps({"overall_severity": "HIGH", "note": secret})
        result = PostProcessNode().execute({"formatted_output": leaky})
        assert result["status"] == AgentStatus.ERROR.value
        assert result["result"] == EGRESS_NOTICE
        assert result["formatted_output"] == EGRESS_NOTICE

    @pytest.mark.parametrize(
        "secret",
        [
            "password=hunter2hunter2",
            "client_secret = s3cr3tvalue",
            "-----BEGIN RSA PRIVATE KEY-----",
        ],
    )
    def test_locally_detected_credentials_are_withheld_too(self, secret):
        # These match NO framework pattern. Delegating this gate wholly to the
        # framework detector would make it narrower while looking like an upgrade.
        from framework.security.credential_detector import detect_credentials_in_value

        assert detect_credentials_in_value(secret) == []
        leaky = json.dumps({"overall_severity": "HIGH", "note": secret})
        assert PostProcessNode().execute({"formatted_output": leaky})["status"] == AgentStatus.ERROR.value

    def test_the_withheld_envelope_carries_no_released_text(self):
        leaky = json.dumps({"affected_agent_id": "algo_secret", "note": "Bearer abcdefghij0123456789"})
        result = PostProcessNode().execute({"formatted_output": leaky})
        blob = json.dumps(result, default=str)
        assert "Bearer abcdefghij0123456789" not in blob
        assert "algo_secret" not in blob
        assert "Traceback" not in blob
        assert "/src/" not in blob

    def test_the_replacement_is_truthy_and_result_is_replaced(self):
        # Clearing to a FALSY value would re-open get_output()'s
        # `formatted_output or result` fallback onto the un-gated answer.
        result = PostProcessNode().execute({"formatted_output": json.dumps({"n": "Bearer abcdefghij0123456789"})})
        assert result["formatted_output"]
        assert result["result"] == EGRESS_NOTICE

    def test_oversized_release_is_withheld(self):
        result = PostProcessNode().execute({"formatted_output": "x" * (MAX_RELEASED_CHARS + 1)})
        assert result["status"] == AgentStatus.ERROR.value

    def test_at_the_size_limit_the_alert_is_still_released(self):
        result = PostProcessNode().execute({"formatted_output": "x" * MAX_RELEASED_CHARS})
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_ordinary_alert_numbers_are_not_mistaken_for_secrets(self):
        alert = json.dumps(
            {
                "description": "order_count=1500 ceiling=1000 weight_fraction=0.6667",
                "audit_trail_id": "FIN-C2-105-algo_alpha-4-CRITICAL",
            }
        )
        assert PostProcessNode().execute({"formatted_output": alert})["status"] == AgentStatus.SUCCESS.value
