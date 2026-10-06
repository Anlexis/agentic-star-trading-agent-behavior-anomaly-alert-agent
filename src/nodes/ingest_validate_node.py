"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(self, state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# FIN-C2-105 — IngestValidateNode
# First inner domain node: parse, bound, sanitise and aggregate the trade log.
#
# Input:  state["validated_input"] (or the inner graph's user_input)
#         state["input_context"]   — caller provenance identifiers
# Output: state["validated_trade_log"] — detection-ready aggregate (JSON string)
#         state["trade_log_metadata"]  — period / algorithm id / counts / baseline
#         state["caller_thresholds"]   — validated per-request threshold overrides
#         state["validation_errors"]   — [] on the accepted path
#
# THIS NODE OWNS THE DATA CONTRACT, AND IT FAILS CLOSED.
#   * every caller-supplied NUMBER goes through finite_in_range(): NaN and
#     Infinity parse fine through float() and compare False against everything,
#     so an unchecked one silently switches a detection rule off — on precisely
#     the decision this agent exists to make;
#   * every caller-supplied STRING that is rendered into the alert is locked to
#     an inert identifier. Free text there is caller-controlled output injection,
#     and a newline in a rendered value is how a forged step gets manufactured
#     inside an otherwise trustworthy report;
#   * raw financial values (order_amount, position_delta, price) never leave this
#     node — they are dropped before anything downstream can see them;
#   * a rejection names the FIELD and never echoes the value.
#
# S-4: emit_trace_event — counts only.

import json
import logging
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Set

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.nodes.anomaly_analyze_node import ANOMALY_CONFIG_BOUNDS
from src.schemas.state import to_json
from src.services.service import (
    InputRejected,
    bounded_sequence,
    finite_in_range,
    inert_identifier,
)

logger = logging.getLogger(__name__)

# Fields carrying raw financial values — dropped, never forwarded.
_MNPI_STRIP_FIELDS = frozenset({"order_amount", "position_delta", "price", "raw_order", "notional"})

# Fields required in each trade-log entry.
_REQUIRED_ENTRY_FIELDS = ("order_id", "timestamp", "instrument_id", "agent_id")

# Structural ceilings — a submission beyond any of these is refused, not truncated.
MAX_ENTRIES = 2_000
MAX_POLICY_TAGS_PER_ENTRY = 32
MAX_DISTINCT_INSTRUMENTS = 500
MAX_BASELINE_BUCKETS = 64
MAX_TIMESTAMP_CHARS = 40
MAX_LATENCY_MS = 3_600_000.0

# Caller provenance identifiers accepted on the input_context channel. Anything
# else the caller sends there is DROPPED at the adapter; this list is what the
# alert may quote back.
CONTEXT_FIELDS = ("desk_code", "report_reference")


class IngestValidateNode(FunctionNode):
    """Parse, bound, sanitise and aggregate the incoming trade log.

    Output state keys (partial dict):
        validated_trade_log  JSON string — detection-ready aggregate
        trade_log_metadata   JSON string — period / algorithm id / counts / baseline / provenance
        caller_thresholds    JSON string — validated per-request threshold overrides
        validation_errors    list[str]   — [] on the accepted path
        status               AgentStatus.SUCCESS, or ERROR when the contract is not met
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        limits = (config or {}).get("limits") or {}
        self._max_entries = int(limits.get("max_trade_log_entries", MAX_ENTRIES))

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input") or ""
        context = state.get("input_context") or {}

        try:
            payload = self._parse_payload(raw)
            entries = self._validate_entries(payload)
            thresholds = self._validate_thresholds(payload.get("thresholds"))
            baseline = self._validate_baseline(payload.get("baseline"))
            provenance = self._validate_context(context)
        except InputRejected as rejection:
            logger.warning("IngestValidateNode: submission refused (field=%s)", rejection.field)
            emit_trace_event(
                "ingest_validate_rejected",
                {"field": rejection.field, "reason": rejection.reason},
                state,
            )
            return {
                "validation_errors": [f"{rejection.field}: {rejection.reason}"],
                "status": AgentStatus.ERROR.value,
                "error_log": [f"IngestValidateNode: {rejection.field} — {rejection.reason}"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (f"IngestValidateNode: {rejection.field} — {rejection.reason}"),
            }

        metadata = _extract_metadata(entries)
        metadata["confidence_score_baseline"] = baseline
        metadata.update(provenance)
        detection = _aggregate_detection_dict(entries)

        logger.info(
            "IngestValidateNode: accepted %d entries across %d instruments",
            metadata["entry_count"],
            metadata["instrument_count"],
        )

        emit_trace_event(
            "ingest_validate_complete",
            {
                "entry_count": metadata["entry_count"],
                "instrument_count": metadata["instrument_count"],
                "threshold_overrides": sorted(thresholds),
                "baseline_buckets": len(baseline),
            },
            state,
        )

        return {
            # ADR-005: structured fields travel as JSON strings for msgpack-safe checkpointing.
            "validated_trade_log": to_json(detection),
            "trade_log_metadata": to_json(metadata),
            "caller_thresholds": to_json(thresholds),
            "validation_errors": [],
            "status": AgentStatus.SUCCESS.value,
        }

    # ------------------------------------------------------------------
    # Contract enforcement
    # ------------------------------------------------------------------

    def _parse_payload(self, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, str) or not raw.strip():
            raise InputRejected("validated_input", "is empty")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise InputRejected("validated_input", "is not valid JSON") from None
        if not isinstance(payload, dict):
            raise InputRejected("validated_input", "must be a JSON object at the top level")
        return payload

    def _validate_entries(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_entries = payload.get("trade_log_entries")
        if raw_entries is None:
            raw_entries = payload.get("entries")
        entries = bounded_sequence(raw_entries, field="trade_log_entries", max_items=self._max_entries)
        if not entries:
            # An empty log is refused rather than reported clean. "No anomaly
            # exceeded its threshold" over zero orders reads as a clean bill of
            # health for a period nobody actually submitted.
            raise InputRejected("trade_log_entries", "must contain at least one entry")

        sanitised: List[Dict[str, Any]] = []
        instruments: Set[str] = set()
        for index, entry in enumerate(entries):
            field = f"trade_log_entries[{index}]"
            if not isinstance(entry, dict):
                raise InputRejected(field, f"must be an object, got {type(entry).__name__}")
            for required in _REQUIRED_ENTRY_FIELDS:
                if required not in entry:
                    raise InputRejected(f"{field}.{required}", "is required")

            clean: Dict[str, Any] = {
                "order_id": inert_identifier(entry["order_id"], field=f"{field}.order_id"),
                "instrument_id": inert_identifier(entry["instrument_id"], field=f"{field}.instrument_id"),
                "agent_id": inert_identifier(entry["agent_id"], field=f"{field}.agent_id"),
                "timestamp": _validate_timestamp(entry["timestamp"], field=f"{field}.timestamp"),
            }
            instruments.add(clean["instrument_id"])
            if len(instruments) > MAX_DISTINCT_INSTRUMENTS:
                raise InputRejected(
                    "trade_log_entries", f"exceeds the maximum of {MAX_DISTINCT_INSTRUMENTS} distinct instruments"
                )

            if "confidence_score_bucket" in entry:
                clean["confidence_score_bucket"] = inert_identifier(
                    entry["confidence_score_bucket"], field=f"{field}.confidence_score_bucket"
                )
            if "latency_ms" in entry:
                clean["latency_ms"] = finite_in_range(
                    entry["latency_ms"], field=f"{field}.latency_ms", low=0.0, high=MAX_LATENCY_MS
                )
            if "policy_tags" in entry:
                tags = bounded_sequence(
                    entry["policy_tags"], field=f"{field}.policy_tags", max_items=MAX_POLICY_TAGS_PER_ENTRY
                )
                clean["policy_tags"] = [
                    inert_identifier(tag, field=f"{field}.policy_tags[{n}]") for n, tag in enumerate(tags)
                ]

            # Raw financial values are dropped here and never reach any later node.
            for stripped_field in _MNPI_STRIP_FIELDS:
                clean.pop(stripped_field, None)
            sanitised.append(clean)

        return sanitised

    def _validate_thresholds(self, thresholds: Any) -> Dict[str, float]:
        """Validate per-request threshold overrides against the shared bounds table."""
        if thresholds is None:
            return {}
        if not isinstance(thresholds, dict):
            raise InputRejected("thresholds", f"must be an object, got {type(thresholds).__name__}")
        validated: Dict[str, float] = {}
        for key, value in thresholds.items():
            if key not in ANOMALY_CONFIG_BOUNDS:
                raise InputRejected(f"thresholds.{key}", "is not a supported threshold")
            low, high = ANOMALY_CONFIG_BOUNDS[key]
            validated[key] = finite_in_range(value, field=f"thresholds.{key}", low=low, high=high)
        return validated

    def _validate_baseline(self, baseline: Any) -> Dict[str, float]:
        """Validate the caller's confidence-bucket baseline distribution."""
        if baseline is None:
            return {}
        if not isinstance(baseline, dict):
            raise InputRejected("baseline", f"must be an object, got {type(baseline).__name__}")
        buckets = baseline.get("confidence_score_baseline")
        if buckets is None:
            return {}
        if not isinstance(buckets, dict):
            raise InputRejected("baseline.confidence_score_baseline", "must be an object")
        if len(buckets) > MAX_BASELINE_BUCKETS:
            raise InputRejected(
                "baseline.confidence_score_baseline", f"exceeds the maximum of {MAX_BASELINE_BUCKETS} buckets"
            )
        validated: Dict[str, float] = {}
        for key, value in buckets.items():
            name = inert_identifier(key, field="baseline.confidence_score_baseline.<key>")
            validated[name] = finite_in_range(
                value, field=f"baseline.confidence_score_baseline.{name}", low=0.0, high=1e12
            )
        return validated

    def _validate_context(self, context: Any) -> Dict[str, str]:
        """Validate the caller provenance identifiers carried on input_context."""
        if not isinstance(context, dict):
            return {}
        provenance: Dict[str, str] = {}
        for name in CONTEXT_FIELDS:
            if name in context and context[name] is not None:
                provenance[name] = inert_identifier(context[name], field=f"input_context.{name}")
        return provenance


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _validate_timestamp(value: Any, *, field: str) -> str:
    """Accept an ISO-8601 timestamp string; refuse anything else."""
    if not isinstance(value, str):
        raise InputRejected(field, f"must be an ISO-8601 string, got {type(value).__name__}")
    if len(value) > MAX_TIMESTAMP_CHARS:
        raise InputRejected(field, "is too long to be an ISO-8601 timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise InputRejected(field, "is not a valid ISO-8601 timestamp") from None
    return value


def _extract_metadata(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive trade_log_metadata from the sanitised entry list."""
    if not entries:
        return {"period": None, "agent_id": None, "instrument_count": 0, "entry_count": 0}

    timestamps = [e["timestamp"] for e in entries if e.get("timestamp")]
    period = f"{min(timestamps)} / {max(timestamps)}" if timestamps else None
    agent_id = next((e.get("agent_id") for e in entries if e.get("agent_id") is not None), None)
    instruments = {e.get("instrument_id") for e in entries if e.get("instrument_id")}

    return {
        "period": period,
        "agent_id": agent_id,
        "instrument_count": len(instruments),
        "entry_count": len(entries),
    }


def _aggregate_detection_dict(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate sanitised entries into the shape AnomalyAnalyzeNode consumes.

    Building this here keeps the aggregation responsibility on the producer
    rather than the detector, and guarantees the detector only ever sees values
    that have already passed the contract above.
    """
    detection: Dict[str, Any] = {
        "positions": {},
        "confidence_score_buckets": {},
        "latency_ms_series": [],
        "policy_tags": [],
        "order_count": len(entries),
        "trades": [],
    }
    for entry in entries:
        instrument = entry.get("instrument_id")
        if instrument:
            detection["positions"][instrument] = detection["positions"].get(instrument, 0) + 1

        bucket = entry.get("confidence_score_bucket")
        if bucket:
            detection["confidence_score_buckets"][bucket] = detection["confidence_score_buckets"].get(bucket, 0) + 1

        latency = entry.get("latency_ms")
        if latency is not None:
            detection["latency_ms_series"].append(latency)

        detection["policy_tags"].extend(entry.get("policy_tags") or [])

        agent_id = entry.get("agent_id")
        timestamp = entry.get("timestamp")
        if agent_id and timestamp:
            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            detection["trades"].append({"agent_id": agent_id, "timestamp_sec": parsed.timestamp()})

    return detection
