"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(self, state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Read input_context via state.get("input_context", {}) — read-only
#  - Never import from mediator/, api/, or other agents
#
# FIN-C2-105 — PreProcessNode
# Outer backbone pre_process slot. Owns the CALLER CONTRACT: it decides what a
# well-formed submission is and refuses everything else, before any domain code
# sees the payload.
#
# S-1/S-2 rules:
#   - user_input must be a non-empty JSON object within the declared size cap
#   - the expected top-level keys must be present
#     (trade_log_entries, baseline, thresholds)
#   - the parsed payload is screened for injection content — control tokens AND
#     directive phrasing, raw AND post-sanitize, values AND keys
#   - never log order values, position deltas, prices or confidence scores
#   - never echo a rejected value; name the FIELD only
#
# The template owns this refusal. The framework's own S-2 gate runs first where
# it is active, but a template that relies on it alone fails OPEN wherever that
# gate is absent or configured off, so the checks below are enforced here and
# proved by calling execute() directly.
#
# S-4: emit_trace_event — once after the main side-effect; non-sensitive metadata only.

import json
import logging
from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.service import REFUSAL_NOTICE, InputRejected, screen_injection

logger = logging.getLogger(__name__)

# Expected top-level keys in the trade-log JSON payload.
_REQUIRED_KEYS = frozenset({"trade_log_entries", "baseline", "thresholds"})

# Structural ceilings. Defaults are used when config/config.yaml declares none;
# an over-large submission is refused rather than truncated, because a truncated
# trade log produces an alert about a log the caller never sent.
DEFAULT_MAX_INPUT_BYTES = 262_144
DEFAULT_MAX_TRADE_LOG_ENTRIES = 2_000


class PreProcessNode(FunctionNode):
    """Structural validation of the raw trade-log JSON submitted by the caller.

    Input state keys:
        user_input: raw trade-log JSON string supplied by the caller

    Output state keys (partial dict):
        validated_input: canonical trade-log JSON string
        status:          AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:       (on error) one message naming the offending field
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        limits = (config or {}).get("limits") or {}
        self._max_input_bytes = int(limits.get("max_input_bytes", DEFAULT_MAX_INPUT_BYTES))
        self._max_entries = int(limits.get("max_trade_log_entries", DEFAULT_MAX_TRADE_LOG_ENTRIES))

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw = state.get("user_input", "")

        try:
            payload = self._parse(raw)
        except InputRejected as rejection:
            logger.warning("PreProcessNode: submission refused (field=%s)", rejection.field)
            emit_trace_event(
                "pre_process_rejected",
                {"field": rejection.field, "reason": rejection.reason},
                state,
            )
            # formatted_output carries the closed-set notice because the framework
            # short-circuits every downstream node once status is ERROR: the main
            # slot never runs, so nothing later can publish a refusal, and without
            # this line the caller receives a bare `output: null`.
            return {
                "status": AgentStatus.ERROR.value,
                "formatted_output": REFUSAL_NOTICE,
                "result": None,
                "error_log": [f"PreProcessNode: {rejection.field} — {rejection.reason}"],
            }

        validated_input = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        logger.info(
            "PreProcessNode: structural validation passed — keys=%s payload_size=%d",
            sorted(payload.keys()),
            len(validated_input),
        )

        entries = payload.get("trade_log_entries")
        emit_trace_event(
            "pre_process_complete",
            {
                "top_level_keys": sorted(payload.keys()),
                "payload_size": len(validated_input),
                "entry_count": len(entries) if isinstance(entries, list) else None,
            },
            state,
        )

        return {
            "validated_input": validated_input,
            "status": AgentStatus.SUCCESS.value,
        }

    # ------------------------------------------------------------------
    # Caller contract
    # ------------------------------------------------------------------

    def _parse(self, raw: Any) -> Dict[str, Any]:
        """Return the parsed payload, or raise InputRejected naming the field."""
        if not isinstance(raw, str):
            raise InputRejected("user_input", f"must be a string, got {type(raw).__name__}")

        stripped = raw.strip()
        if not stripped:
            raise InputRejected("user_input", "is empty or blank")

        encoded_size = len(stripped.encode("utf-8"))
        if encoded_size > self._max_input_bytes:
            raise InputRejected("user_input", f"exceeds the {self._max_input_bytes}-byte submission limit")

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            # The decoder's message quotes the offending text, so it is not
            # forwarded — the caller gets the field, not their own payload back.
            raise InputRejected("user_input", "is not valid JSON") from None

        if not isinstance(payload, dict):
            raise InputRejected("user_input", "must be a JSON object at the top level")

        missing = sorted(_REQUIRED_KEYS - payload.keys())
        if missing:
            raise InputRejected("user_input", f"missing required top-level keys: {missing}")

        entries = payload.get("trade_log_entries")
        if not isinstance(entries, list):
            raise InputRejected("trade_log_entries", "must be a list")
        if len(entries) > self._max_entries:
            raise InputRejected("trade_log_entries", f"exceeds the maximum of {self._max_entries} entries")

        # Depth-first over the PARSED structure, keys included: a \u-escaped
        # control token has already been decoded by json.loads() at this point,
        # so the escape buys the attacker nothing.
        screen_injection(payload, field="user_input")
        return payload
