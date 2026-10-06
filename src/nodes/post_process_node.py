"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(self, state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import the platform SDK
#
# FIN-C2-105 — PostProcessNode
# Outer backbone post_process slot: the LAST code that runs before the caller
# sees anything. It reads state["formatted_output"] (written by
# TradingAnomalyAlertGraphNode.merge_output() from the inner graph's result) and
# publishes it as state["result"] for the framework FinalizeNode.
#
# THIS IS THE EGRESS GATE.
#   * Credential scan = the framework's own detect_credentials UNION a local
#     pattern set. Neither is a superset of the other: the framework's patterns
#     describe credential FORMATS (AKIA…, sk-…, Bearer …, JWT, DB URIs) and match
#     nothing of the shape `password=…`, which the local set catches. Taking the
#     union is the only direction that is safe — swapping local for framework
#     would make this gate NARROWER while looking like a tightening.
#   * On violation the node returns ERROR *and* replaces every output-bearing
#     field. Raising is not containment: AgentBaseGraph.get_output() returns
#     `formatted_output or result` on the error path too, so an un-cleared field
#     ships the un-gated answer inside the error envelope. The replacement notice
#     is TRUTHY, because a falsy formatted_output re-opens that same fallback.
#   * The size cap is here rather than in the composer because it is a property
#     of what is released, not of what was computed.
#
# The domain invariant (no raw financial values, no redaction sentinel reported
# as a finding) is enforced one layer in, by RiskAlertComposeNode. The two layers
# check different things ON PURPOSE, so neither can contain the other's mutant.
#
# S-4: emit_trace_event on completion; never the released text itself.

import json
import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.service import credential_findings
from src.services.llm_factory import resolve_llm
from src.services.llm_review import render_review, review_result

logger = logging.getLogger(__name__)

# Maximum size of a released alert. A structurally valid but enormous report is a
# denial-of-service against whoever has to read it.
MAX_RELEASED_CHARS = 262_144

# Fixed refusal payload. It quotes nothing from the output it replaces — no
# released text, no exception message, no traceback, no source path.
EGRESS_NOTICE = json.dumps(
    {
        "status": "refused",
        "reason": "output_withheld",
        "detail": "The generated alert did not pass the output gate and was not released.",
    },
    ensure_ascii=False,
)


class PostProcessNode(FunctionNode):
    """Publish the composed alert, or withhold it and say so.

    Input state keys:
        formatted_output: str — the alert produced by the inner domain workflow

    Output state keys (partial dict):
        result:           str — the released alert, or the refusal notice
        formatted_output: str — replaced on the refusal path
        status:           AgentStatus
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> Dict[str, Any]:
        output = state.get("formatted_output") or ""
        if not isinstance(output, str):
            output = str(output)

        violations: List[str] = []

        _llm, _ = resolve_llm(None, state)
        _remarks = review_result(
            _llm,
            user_input=str(state.get("user_input") or ""),
            result=output,
            domain="FIN TradingAnomalyAlertAgent",
        )
        _review = render_review(_remarks)
        # Remarks are LLM text derived from the caller's raw words, so they pass through the
        # same gate the answer does -- appending after the gate would put unscanned text past
        # it. A tripped review is dropped on its own: withholding a correct answer because an
        # advisory remark quoted an identifier would let the review change the outcome, and
        # the whole design rests on it being unable to.
        if (
            _review
            and isinstance(output, str)
            and not credential_findings(output + _review)
            or len(output + _review) > MAX_RELEASED_CHARS
        ):
            output = output + _review

        findings = credential_findings(output)
        if findings:
            violations.append(f"credential:{','.join(findings)}")

        if len(output) > MAX_RELEASED_CHARS:
            violations.append("released_payload_too_large")

        if violations:
            emit_trace_event("post_process_withheld", {"violations": violations}, state)
            logger.error("PostProcessNode: output withheld (%s)", violations)
            return {
                "status": AgentStatus.ERROR.value,
                "formatted_output": EGRESS_NOTICE,
                "result": EGRESS_NOTICE,
                "risk_alert": None,
                "alert_severity": None,
                "error_log": ["PostProcessNode: output gate refused the alert"],
            }

        emit_trace_event(
            "post_process_complete",
            {"has_output": bool(output.strip()), "released_chars": len(output)},
            state,
        )
        logger.info("PostProcessNode: releasing alert (%d chars)", len(output))

        return {
            "result": output,
            "status": AgentStatus.SUCCESS.value,
        }
