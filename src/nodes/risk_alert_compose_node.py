"""AgentCore Platform v1.0

Node contract:
 - Extend FunctionNode; implement execute(self, state) -> dict
 - Return ONLY the fields this node changes (partial dict — never full state)
 - Return AgentStatus enum constants — never plain strings
 - Never import the platform SDK (CI gate-import-isolation is a grep)

FIN-C2-105 — RiskAlertComposeNode
Final inner domain node. Turns the findings produced by AnomalyAnalyzeNode into a
structured model-risk alert aligned to the Japanese FSA's AI model-risk-management
guidance (金商法 §156).

Responsibilities:
 - determine the overall severity across the findings
 - build the alert payload, quoting the evidence each rule actually produced
 - enforce the domain output invariant in _extra_security_gate_output()
 - serialise formatted_output for the outer graph's merge_output()
 - emit the S-4 domain audit event

THE OUTPUT INVARIANT, AND WHY THE GATE CONTAINS RATHER THAN RAISES
------------------------------------------------------------------
The DOMAIN invariant enforced here is: the alert carries computed findings and
inert identifiers, and never a raw financial value, and never reports the
platform's redaction sentinel as though it were an extracted value.

AgentBaseGraph.get_output() returns ``formatted_output or result`` — and it does
so on the error path too. So a gate that merely raises still ships the un-gated
answer inside the error envelope, and a gate that returns ERROR while leaving the
output fields populated does the same. This gate therefore returns ERROR *and*
replaces every output-bearing field with a closed-set refusal notice.

TWO LAYERS, DELIBERATELY NON-OVERLAPPING
----------------------------------------
The credential scan is NOT duplicated here. It lives at the caller boundary in
PostProcessNode, which takes the framework detector UNION a local pattern set.
Splitting the two invariants keeps each one falsifiable: deleting this gate lets
a forbidden financial key ship and the domain test fails, while deleting the
egress gate lets a credential ship and the credential test fails. Had both layers
checked both things, each would have contained the other's mutant and both E2E
tests would have looked decorative — more defence buying less assurance.
The framework's own @final S-3 credential scan still runs on this node's result
regardless, as the floor under both.

NOTE ON THE PRECISION GRID: this template renders no monetary aggregate. Raw
financial values (order_amount, position_delta, price) are dropped at ingest and
the alert quotes only computed ratios, counts and latencies, so a
round-to-the-nearest-1,000 grid is not applicable and is deliberately not
implemented — applying one here would corrupt the ratios and millisecond figures
the alert exists to report. The invariant enforced instead is the one above.
"""

import json
import logging
from typing import Any, ClassVar, Dict, List, Mapping, Set

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json
from src.services.service import MASK_SENTINEL

logger = logging.getLogger(__name__)

# Severity ordering: lower number = higher severity
SEVERITY_MAP: Dict[str, int] = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}
_SEVERITY_BY_RANK: Dict[int, str] = {rank: label for label, rank in SEVERITY_MAP.items()}

# Anomaly type → primary regulatory article
FSA_ARTICLE_MAP: Dict[str, str] = {
    "POLICY_VIOLATION": "金商法 §156 (algorithmic trading disclosure)",
    "POSITION_CONCENTRATION": "金商法 §156 (concentration risk)",
    "MODEL_DRIFT": "FSA AI MRM Q3-2026 §4.2 (model behaviour monitoring)",
    "LATENCY_SPIKE": "FSA AI MRM Q3-2026 §4.3 (execution integrity)",
    "COORDINATION_SIGNAL": "金商法 §156 (market manipulation safeguards)",
}

_DEFAULT_FSA_ARTICLE = "FSA AI MRM Q3-2026 §4.1 (general monitoring)"

# Keys that must never appear anywhere in the composed alert — the raw financial
# values this template exists to keep out of a shareable report.
FORBIDDEN_ALERT_KEYS = frozenset({"order_amount", "position_delta", "price", "raw_order", "notional"})

RECOMMENDED_ACTIONS: Dict[str, str] = {
    "CRITICAL": (
        "Immediate escalation to the Compliance Officer required. "
        "Suspend the affected trading algorithm pending investigation."
    ),
    "HIGH": ("Notify the Compliance Officer within 1 business hour. " "Flag the algorithm for enhanced monitoring."),
    "MEDIUM": ("Schedule a Compliance review within 1 business day. " "Increase anomaly monitoring frequency."),
    "LOW": "Log for the weekly Compliance review. Continue the standard monitoring cadence.",
}

CLEAN_ACTION = "No action required — no anomaly exceeded its threshold."

# Fixed refusal payload used when the output invariant is violated. It quotes
# nothing from the alert it replaces.
CONTAINMENT_NOTICE = json.dumps(
    {
        "status": "refused",
        "reason": "output_invariant_violated",
        "detail": "The composed alert did not satisfy the output contract and was withheld.",
    },
    ensure_ascii=False,
)


def _determine_severity(anomalies: List[Dict[str, Any]]) -> str:
    """Return the highest-priority severity across all findings."""
    best_rank = SEVERITY_MAP["LOW"]
    for anomaly in anomalies:
        rank = SEVERITY_MAP.get(str(anomaly.get("severity", "LOW")).upper(), SEVERITY_MAP["LOW"])
        best_rank = min(best_rank, rank)
    return _SEVERITY_BY_RANK[best_rank]


def _primary_fsa_article(anomalies: List[Dict[str, Any]]) -> str:
    """Return the article reference for the highest-severity finding."""
    best_rank = SEVERITY_MAP["LOW"] + 1
    primary = _DEFAULT_FSA_ARTICLE
    for anomaly in anomalies:
        rank = SEVERITY_MAP.get(str(anomaly.get("severity", "LOW")).upper(), SEVERITY_MAP["LOW"])
        if rank < best_rank:
            best_rank = rank
            primary = FSA_ARTICLE_MAP.get(str(anomaly.get("type", "")), _DEFAULT_FSA_ARTICLE)
    return primary


def _walk_strings(value: Any) -> List[str]:
    """Every string leaf in a JSON-like structure, keys included."""
    found: List[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                found.append(key)
            found.extend(_walk_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_walk_strings(item))
    return found


def _forbidden_keys_present(value: Any) -> List[str]:
    """Forbidden keys found ANYWHERE in the structure, not only at the top level."""
    hits: Set[str] = set()
    if isinstance(value, Mapping):
        hits |= {k for k in value if isinstance(k, str) and k in FORBIDDEN_ALERT_KEYS}
        for item in value.values():
            hits |= set(_forbidden_keys_present(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            hits |= set(_forbidden_keys_present(item))
    return sorted(hits)


class RiskAlertComposeNode(FunctionNode):
    """Compose the structured model-risk alert from the detected anomalies.

    Required state inputs:
        detected_anomalies  JSON string — findings from AnomalyAnalyzeNode
        anomaly_count       int
        trade_log_metadata  JSON string — period / algorithm id / provenance

    Partial-dict outputs:
        risk_alert        JSON string — the alert payload
        alert_severity    str  — CRITICAL / HIGH / MEDIUM / LOW
        fsa_article_ref   str  — primary regulatory article
        formatted_output  str  — JSON string consumed by merge_output()
        status            AgentStatus
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> Dict[str, Any]:
        anomalies: List[Dict[str, Any]] = from_json(state.get("detected_anomalies"), []) or []
        count: int = int(state.get("anomaly_count", 0) or 0)
        metadata: Dict[str, Any] = from_json(state.get("trade_log_metadata"), {}) or {}

        affected_agent_id = str(metadata.get("agent_id") or "unknown")
        period = metadata.get("period")
        assessed_at = str(period).split(" / ")[-1] if period else ""

        if count == 0 or not anomalies:
            overall_severity = "LOW"
            primary_article = _DEFAULT_FSA_ARTICLE
            recommended_action = CLEAN_ACTION
            fsa_refs: List[str] = [primary_article]
        else:
            overall_severity = _determine_severity(anomalies)
            primary_article = _primary_fsa_article(anomalies)
            seen: List[str] = []
            for anomaly in anomalies:
                article = FSA_ARTICLE_MAP.get(str(anomaly.get("type", "")), _DEFAULT_FSA_ARTICLE)
                if article not in seen:
                    seen.append(article)
            fsa_refs = seen or [primary_article]
            recommended_action = RECOMMENDED_ACTIONS[overall_severity]

        # Deterministic composite id — the same event always produces the same id,
        # so an alert can be reconciled across re-runs without a stored sequence.
        audit_trail_id = f"FIN-C2-105-{affected_agent_id}-{count}-{overall_severity}"

        risk_alert: Dict[str, Any] = {
            "anomalies": [
                {
                    # AnomalyAnalyzeNode emits the finding under key "type"; the
                    # published alert exposes it as "anomaly_type".
                    "anomaly_type": a.get("type", "UNKNOWN"),
                    "severity": a.get("severity", "LOW"),
                    # The rule's own evidence string. It used to be dropped, which
                    # left every published finding with an empty description.
                    "description": str(a.get("evidence", "")),
                    "threshold": a.get("threshold"),
                    "detected_at": assessed_at,
                }
                for a in anomalies
            ],
            "overall_severity": overall_severity,
            "affected_agent_id": affected_agent_id,
            "observation_period": period,
            "instrument_count": metadata.get("instrument_count"),
            "order_count": metadata.get("entry_count"),
            "fsa_article_refs": fsa_refs,
            "recommended_action": recommended_action,
            "audit_trail_id": audit_trail_id,
            "inspection_ready": True,
            "advisory": (
                "この検知結果はAIによる自動分析です。コンプライアンス担当者による確認・判断が必要です。"
                "本アラートのみに基づいて取引システムへの対処を行わないでください。"
            ),
        }
        for name in ("desk_code", "report_reference"):
            if metadata.get(name):
                risk_alert[name] = metadata[name]

        formatted_output = json.dumps(risk_alert, ensure_ascii=False)

        emit_trace_event(
            "risk_alert_composed",
            {"severity": overall_severity, "anomaly_count": count},
            state,
        )
        logger.info(
            "RiskAlertComposeNode: alert composed — severity=%s anomaly_count=%d",
            overall_severity,
            count,
        )

        return {
            "risk_alert": to_json(risk_alert),
            "alert_severity": overall_severity,
            "fsa_article_ref": primary_article,
            "formatted_output": formatted_output,
            "status": AgentStatus.SUCCESS.value,
        }

    # -----------------------------------------------------------------------
    # S-3 domain output gate (FunctionNode extension hook)
    # -----------------------------------------------------------------------

    def _extra_security_gate_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce the alert's output invariant, containing on violation.

        The hook receives the RESULT dict returned by execute() and must RETURN a
        dict. Returning None (the previous signature, ``-> None``) made
        BaseNode.__call__ raise ``AttributeError: 'NoneType' object has no
        attribute 'setdefault'`` on every single invocation — which is why this
        node had never produced an alert.
        """
        alert = from_json(result.get("risk_alert"), {})
        rendered = str(result.get("formatted_output") or "")
        if not alert and not rendered:
            return result

        violations: List[str] = []

        # Recursive: a forbidden key nested inside an anomaly item escapes a
        # top-level key scan entirely.
        forbidden = _forbidden_keys_present(alert)
        if forbidden:
            violations.append(f"forbidden_keys:{','.join(forbidden)}")

        if any(MASK_SENTINEL in s for s in _walk_strings(alert)) or MASK_SENTINEL in rendered:
            violations.append("redaction_sentinel_reported_as_a_finding")

        if not violations:
            return result

        # Containment: ERROR *and* every output-bearing field replaced. The notice
        # is truthy so it cannot re-open get_output()'s `formatted_output or result`
        # fallback onto the un-gated alert.
        emit_trace_event(
            "risk_alert_output_contained",
            {"violations": violations},
            {"node": self.__class__.__name__},
        )
        logger.error("RiskAlertComposeNode: output invariant violated (%s) — alert withheld", violations)
        return {
            "status": AgentStatus.ERROR.value,
            "risk_alert": None,
            "alert_severity": None,
            "fsa_article_ref": None,
            "formatted_output": CONTAINMENT_NOTICE,
            "result": None,
            "error_log": ["RiskAlertComposeNode: output invariant violated — alert withheld"],
        }
