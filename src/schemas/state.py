"""AgentCore Platform v1.0"""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# ⚠️ ADR-005 (msgpack safety): structured fields (dict / list[dict]) are stored
# as JSON STRINGS, not bare Python containers — a bare dict/list in a
# checkpointed State field is a CoE gate-state-safety violation. Producers
# serialize with to_json() on write; consumers deserialize with from_json()
# on read.  (SEC-FIN-C2-105-002.)
#
# FIN-C2-105 — Financial Trade Log Anomaly Detection & FSA Risk Alert Agent
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover the full node pipeline:
#   PreProcessNode → IngestValidateNode → AnomalyAnalyzeNode
#   → RiskAlertComposeNode → PostProcessNode
#
# Security note: this state MUST NOT hold JWT tokens, API keys, or trading
# system credentials (S-5).  trade_log_metadata carries only aggregated
# metadata (period, agent_id, instrument_count) — no individual position
# detail beyond what the caller already supplied.

import json
from typing import Any, List, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (ADR-005 msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input → the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for FIN-C2-105.

    All shared fields (user_input, status, result, error_log, session_id,
    correlation_id, trace_id, node_history, hitl_*, etc.) are inherited
    from AgentState.  Only FIN-C2-105-specific fields are declared here.

    Producer / consumer alignment (structured fields are JSON strings — ADR-005)
    ────────────────────────────────────────────────────────────────────────────
    validated_input       PreProcessNode       → IngestValidateNode
    validated_trade_log   IngestValidateNode   → AnomalyAnalyzeNode      (to_json/from_json)
    trade_log_metadata    IngestValidateNode   → AnomalyAnalyzeNode,     (to_json/from_json)
                                                 RiskAlertComposeNode
    caller_thresholds     IngestValidateNode   → AnomalyAnalyzeNode      (to_json/from_json)
    detected_anomalies    AnomalyAnalyzeNode   → RiskAlertComposeNode    (to_json/from_json)
    anomaly_count         AnomalyAnalyzeNode   → RiskAlertComposeNode    (int)
    risk_alert            RiskAlertComposeNode → S-3 gate                (to_json/from_json)
    formatted_output      RiskAlertComposeNode → PostProcessNode / merge_output()
    """

    # ------------------------------------------------------------------
    # Input / validated input — PreProcessNode (outer backbone)
    # ------------------------------------------------------------------

    # Raw JSON string of the trade log as submitted by the caller.
    # Set by the framework / outer backbone before domain nodes run.
    user_input: Optional[str]

    # Sanitised trade log string written by PreProcessNode after S-1
    # input-validation (removes injection vectors, normalises encoding).
    validated_input: Optional[str]

    # ------------------------------------------------------------------
    # Domain: IngestValidateNode
    # ------------------------------------------------------------------

    # JSON STRING (to_json) of the detection-ready dict (JSON-deserialisable).
    # Shape mirrors the upstream FSA submission format; no PII beyond what the
    # caller supplied.  AnomalyAnalyzeNode reads it via from_json().
    validated_trade_log: Optional[str]

    # List of human-readable validation error messages (msgpack-native list[str]).
    # Empty list means the trade log is structurally clean.
    validation_errors: Optional[List[str]]

    # JSON STRING (to_json) of aggregated metadata:
    # {"period": str, "agent_id": str, "instrument_count": int, "entry_count": int,
    #  "confidence_score_baseline": dict[str, float],
    #  "desk_code": str, "report_reference": str}   ← the last two only when the
    # caller supplied them on input_context; both are inert identifiers.
    trade_log_metadata: Optional[str]

    # JSON STRING (to_json) of the caller's per-request threshold overrides,
    # already validated against ANOMALY_CONFIG_BOUNDS. Empty mapping when the
    # caller sent none. AnomalyAnalyzeNode layers these over the declared config.
    caller_thresholds: Optional[str]

    # ------------------------------------------------------------------
    # Domain: AnomalyAnalyzeNode
    # ------------------------------------------------------------------

    # JSON STRING (to_json) of the anomaly list; each item:
    # {"type": str, "severity": str, "evidence": str, "threshold": str}
    detected_anomalies: Optional[str]

    # Convenience count — len(detected_anomalies); avoids re-counting
    # in downstream nodes.
    anomaly_count: Optional[int]

    # ------------------------------------------------------------------
    # Domain: RiskAlertComposeNode (main slot)
    # ------------------------------------------------------------------

    # JSON STRING (to_json) of the structured FSA MRM alert payload.
    # The S-3 gate reads it back via from_json().
    risk_alert: Optional[str]

    # Severity classification: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    alert_severity: Optional[str]

    # Relevant FSA article reference, e.g. "金商法 §156"
    fsa_article_ref: Optional[str]

    # Final JSON string consumed by PostProcessNode and merge_output().
    # The main-slot node (RiskAlertComposeNode) also writes this value
    # into state["result"] for backbone compatibility.
    formatted_output: Optional[str]

    # ------------------------------------------------------------------
    # Out-of-scope / error flag
    # ------------------------------------------------------------------

    # True when the submitted trade log falls outside the supported
    # instrument types or date range.  Paired with status=SUCCESS —
    # never use a non-existent AgentStatus member to represent this.
    out_of_scope: Optional[bool]
