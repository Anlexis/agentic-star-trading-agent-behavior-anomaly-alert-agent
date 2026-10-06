"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(self, state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Read inputs via state.get(...) — read-only
#  - Never import from mediator/, api/, or other agents
#
# FIN-C2-105 — AnomalyAnalyzeNode
# Inner domain node: deterministic rule/threshold anomaly detection over the
# sanitised trade log. Five anomaly classes:
#   1. POSITION_CONCENTRATION — single-instrument weight above threshold
#   2. MODEL_DRIFT            — confidence-bucket distribution shift vs the caller's baseline
#   3. LATENCY_SPIKE          — execution latency above the rolling-median multiple
#   4. POLICY_VIOLATION       — forbidden policy tags or order-count ceiling breach
#   5. COORDINATION_SIGNAL    — 2+ algorithm ids with correlated timestamps in a window
#
# No model call — a pure rule engine, so the same trade log always produces the
# same alert.
#
# WHERE THRESHOLDS COME FROM
# --------------------------
# BaseNode.__call__() invokes ``self.execute(state)`` with ONE argument. A node
# that declares ``execute(self, state, config=None)`` therefore never receives a
# config: every value read from it is dead, and the agent silently runs on
# hard-coded defaults while config/config.yaml documents behaviour it does not
# have. Thresholds are consequently injected through the CONSTRUCTOR from the
# graph's runtime config, and a caller may override them per request through the
# validated ``caller_thresholds`` state field.
#
# S-1: no order_amount / position_delta / price value ever appears in evidence.
# S-4: emit_trace_event — anomaly counts and type names only.

import logging
import math
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — defaults and the bounds every source must satisfy
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "concentration_threshold": 0.40,  # 40% single-instrument weight
    "latency_spike_multiplier": 3.0,  # 3x rolling median
    "latency_baseline_ms": 100.0,  # fallback median when the log has none
    "drift_bucket_delta": 0.20,  # max allowed bucket-share deviation
    "coordination_window_sec": 2.0,  # correlated-timestamp window
    "order_count_ceiling": 1000.0,  # max approved order count in one log
}

# Single source of truth for the accepted range of every tunable. Both the
# declared config (DomainWorkflowGraph._validate_config) and the caller-supplied
# override (IngestValidateNode) are checked against THIS table, so the two paths
# cannot drift into accepting different values.
ANOMALY_CONFIG_BOUNDS: Dict[str, Tuple[float, float]] = {
    "concentration_threshold": (0.0, 1.0),
    "latency_spike_multiplier": (1.0, 1_000.0),
    "latency_baseline_ms": (0.001, 3_600_000.0),
    "drift_bucket_delta": (0.0, 1.0),
    "coordination_window_sec": (0.0, 86_400.0),
    "order_count_ceiling": (1.0, 10_000_000.0),
}

DEFAULT_FORBIDDEN_POLICY_TAGS: List[str] = ["HALT", "BLOCKED", "RESTRICTED"]

# Anomaly type constants
_TYPE_POSITION_CONCENTRATION = "POSITION_CONCENTRATION"
_TYPE_MODEL_DRIFT = "MODEL_DRIFT"
_TYPE_LATENCY_SPIKE = "LATENCY_SPIKE"
_TYPE_POLICY_VIOLATION = "POLICY_VIOLATION"
_TYPE_COORDINATION_SIGNAL = "COORDINATION_SIGNAL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """KL(p||q) over the union of bucket keys. Returns inf when q has no mass where p does."""
    keys = set(p) | set(q)
    total = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        qk = q.get(k, 0.0)
        if pk <= 0.0:
            continue
        if qk <= 0.0:
            return float("inf")
        total += pk * math.log(pk / qk)
    return total


def _normalize_bucket_dist(raw: Dict[str, Any]) -> Dict[str, float]:
    """Normalise a bucket count/weight mapping to a probability distribution.

    Non-finite weights are dropped rather than propagated: an inf or NaN weight
    turns every later comparison False, which would switch the drift rule off.
    IngestValidateNode already refuses those, so reaching this branch means the
    node was driven directly — it still must not fail open.
    """
    if not raw:
        return {}
    values: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        values[str(key)] = max(0.0, number)
    total = sum(values.values())
    if total <= 0.0:
        return {k: 0.0 for k in values}
    return {k: v / total for k, v in values.items()}


def _check_position_concentration(trade_log: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
    """Detect any single instrument exceeding the concentration threshold."""
    anomalies: List[Dict[str, Any]] = []
    positions = trade_log.get("positions", {})
    if not isinstance(positions, dict) or not positions:
        return anomalies

    total_weight = sum(
        float(v)
        for v in positions.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
    )
    if total_weight <= 0.0:
        return anomalies

    for instrument, weight_raw in sorted(positions.items()):
        try:
            weight = float(weight_raw) / total_weight
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if not math.isfinite(weight):
            continue
        if weight > threshold:
            anomalies.append(
                {
                    "type": _TYPE_POSITION_CONCENTRATION,
                    "severity": "CRITICAL" if weight > 0.60 else "HIGH",
                    "evidence": (f"instrument={instrument} weight_fraction={weight:.4f} threshold={threshold:.4f}"),
                    "threshold": threshold,
                }
            )
    return anomalies


def _check_model_drift(
    trade_log: Dict[str, Any], metadata: Dict[str, Any], max_bucket_delta: float
) -> List[Dict[str, Any]]:
    """Detect a confidence-bucket distribution shift against the caller's baseline.

    The baseline travels in trade_log_metadata under ``confidence_score_baseline``
    — IngestValidateNode carries it there from the caller's ``baseline`` block.
    Without that carry this rule is structurally unreachable, which is exactly
    what it used to be.
    """
    anomalies: List[Dict[str, Any]] = []
    current_raw = trade_log.get("confidence_score_buckets", {})
    baseline_raw = metadata.get("confidence_score_baseline", {})
    if not current_raw or not baseline_raw:
        return anomalies

    current = _normalize_bucket_dist(current_raw)
    baseline = _normalize_bucket_dist(baseline_raw)

    max_delta = 0.0
    worst_bucket = ""
    for bucket in sorted(set(current) | set(baseline)):
        delta = abs(current.get(bucket, 0.0) - baseline.get(bucket, 0.0))
        if delta > max_delta:
            max_delta = delta
            worst_bucket = bucket

    if max_delta > max_bucket_delta:
        kl = _kl_divergence(current, baseline)
        severity = "HIGH" if kl > 0.5 else "MEDIUM"
        anomalies.append(
            {
                "type": _TYPE_MODEL_DRIFT,
                "severity": severity,
                "evidence": (
                    f"worst_bucket={worst_bucket} delta={max_delta:.4f} "
                    f"threshold={max_bucket_delta:.4f} kl_divergence={kl:.4f}"
                ),
                "threshold": max_bucket_delta,
            }
        )
    return anomalies


def _check_latency_spikes(
    trade_log: Dict[str, Any], spike_multiplier: float, baseline_ms: float
) -> List[Dict[str, Any]]:
    """Detect execution-latency spikes above spike_multiplier x the rolling median."""
    anomalies: List[Dict[str, Any]] = []
    latencies = trade_log.get("latency_ms_series", [])
    if not isinstance(latencies, (list, tuple)) or not latencies:
        return anomalies

    numeric = [
        float(v)
        for v in latencies
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
    ]
    if not numeric:
        return anomalies

    sorted_vals = sorted(numeric)
    mid = len(sorted_vals) // 2
    median = sorted_vals[mid] if len(sorted_vals) % 2 == 1 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

    effective_median = median if median > 0.0 else baseline_ms
    spike_threshold = effective_median * spike_multiplier

    spike_count = sum(1 for v in numeric if v > spike_threshold)
    if spike_count > 0:
        max_latency = max(numeric)
        severity = "CRITICAL" if max_latency > spike_threshold * 5 else "HIGH"
        anomalies.append(
            {
                "type": _TYPE_LATENCY_SPIKE,
                "severity": severity,
                "evidence": (
                    f"spike_count={spike_count} rolling_median_ms={effective_median:.2f} "
                    f"spike_threshold_ms={spike_threshold:.2f} multiplier={spike_multiplier}"
                ),
                "threshold": spike_threshold,
            }
        )
    return anomalies


def _check_policy_violations(
    trade_log: Dict[str, Any], forbidden_tags: List[str], order_count_ceiling: int
) -> List[Dict[str, Any]]:
    """Detect forbidden policy tags or an order-count ceiling breach."""
    anomalies: List[Dict[str, Any]] = []
    policy_tags = trade_log.get("policy_tags", [])
    if not isinstance(policy_tags, (list, tuple)):
        policy_tags = []

    forbidden_upper = {t.upper() for t in forbidden_tags}
    forbidden_found = sorted({tag for tag in policy_tags if isinstance(tag, str) and tag.upper() in forbidden_upper})
    if forbidden_found:
        anomalies.append(
            {
                "type": _TYPE_POLICY_VIOLATION,
                "severity": "CRITICAL",
                "evidence": (f"forbidden_tags_found={forbidden_found} forbidden_list={sorted(forbidden_tags)}"),
                "threshold": 0,
            }
        )

    order_count_raw = trade_log.get("order_count", 0)
    try:
        order_count = int(order_count_raw)
    except (TypeError, ValueError):
        order_count = 0

    if order_count > order_count_ceiling:
        anomalies.append(
            {
                "type": _TYPE_POLICY_VIOLATION,
                "severity": "HIGH",
                "evidence": f"order_count={order_count} ceiling={order_count_ceiling}",
                "threshold": order_count_ceiling,
            }
        )
    return anomalies


def _check_coordination_signals(trade_log: Dict[str, Any], window_sec: float) -> List[Dict[str, Any]]:
    """Detect 2+ algorithm ids trading within the correlation window."""
    anomalies: List[Dict[str, Any]] = []
    trades = trade_log.get("trades", [])
    if not isinstance(trades, (list, tuple)) or len(trades) < 2:
        return anomalies

    timestamped: List[Tuple[float, str]] = []
    for entry in trades:
        if not isinstance(entry, dict):
            continue
        try:
            ts = float(entry["timestamp_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(ts):
            continue
        timestamped.append((ts, str(entry.get("agent_id", "unknown"))))

    if len(timestamped) < 2:
        return anomalies

    timestamped.sort(key=lambda x: x[0])

    coordinated_groups: List[Set[str]] = []
    for i in range(len(timestamped)):
        group = {timestamped[i][1]}
        for j in range(i + 1, len(timestamped)):
            if timestamped[j][0] - timestamped[i][0] > window_sec:
                break
            group.add(timestamped[j][1])
        if len(group) >= 2 and not any(group <= existing for existing in coordinated_groups):
            coordinated_groups.append(group)

    if coordinated_groups:
        largest_group: Set[str] = set()
        for group in coordinated_groups:
            if len(group) > len(largest_group):
                largest_group = group
        largest = sorted(largest_group)
        anomalies.append(
            {
                "type": _TYPE_COORDINATION_SIGNAL,
                "severity": "HIGH",
                "evidence": (
                    f"correlated_algorithm_ids={largest} window_sec={window_sec} "
                    f"group_count={len(coordinated_groups)}"
                ),
                "threshold": window_sec,
            }
        )
    return anomalies


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class AnomalyAnalyzeNode(FunctionNode):
    """Deterministic rule/threshold anomaly detection over the sanitised trade log.

    Input state keys:
        validated_trade_log  JSON string — detection-ready aggregate from IngestValidateNode
        trade_log_metadata   JSON string — period / algorithm id / instrument count / baseline
        caller_thresholds    JSON string — per-request threshold overrides, already validated
        validation_errors    list[str]   — if non-empty, detection is skipped

    Output state keys (partial dict):
        detected_anomalies   JSON string of the finding list
        anomaly_count        int
        status               AgentStatus
        error_log            list[str] on the ERROR path
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        declared = (config or {}).get("anomaly") or {}
        self._thresholds: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
        for key in ANOMALY_CONFIG_BOUNDS:
            if key in declared:
                self._thresholds[key] = float(declared[key])
        tags = declared.get("forbidden_policy_tags")
        self._forbidden_tags: List[str] = (
            [str(t) for t in tags] if isinstance(tags, list) else list(DEFAULT_FORBIDDEN_POLICY_TAGS)
        )

    def execute(self, state: AgentState) -> Dict[str, Any]:
        errors: List[str] = state.get("validation_errors", []) or []
        if errors:
            logger.info("AnomalyAnalyzeNode: %d upstream validation error(s); detection skipped", len(errors))
            emit_trace_event("anomaly_analyze_skipped", {"upstream_error_count": len(errors)}, state)
            return {
                "detected_anomalies": to_json([]),
                "anomaly_count": 0,
                "status": AgentStatus.SUCCESS.value,
            }

        trade_log = from_json(state.get("validated_trade_log"), {})
        metadata = from_json(state.get("trade_log_metadata"), {})
        if not isinstance(trade_log, dict) or not isinstance(metadata, dict):
            emit_trace_event("anomaly_analyze_rejected", {"field": "validated_trade_log"}, state)
            return {
                "detected_anomalies": to_json([]),
                "anomaly_count": 0,
                "status": AgentStatus.ERROR.value,
                "error_log": ["AnomalyAnalyzeNode: validated_trade_log — malformed detection input"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("AnomalyAnalyzeNode: validated_trade_log — malformed detection input"),
            }

        thresholds = self._effective_thresholds(state)

        detected: List[Dict[str, Any]] = []
        detected.extend(_check_position_concentration(trade_log, thresholds["concentration_threshold"]))
        detected.extend(_check_model_drift(trade_log, metadata, thresholds["drift_bucket_delta"]))
        detected.extend(
            _check_latency_spikes(
                trade_log,
                thresholds["latency_spike_multiplier"],
                thresholds["latency_baseline_ms"],
            )
        )
        detected.extend(
            _check_policy_violations(trade_log, self._forbidden_tags, int(thresholds["order_count_ceiling"]))
        )
        detected.extend(_check_coordination_signals(trade_log, thresholds["coordination_window_sec"]))

        anomaly_count = len(detected)
        logger.info("AnomalyAnalyzeNode: detection complete — %d anomalies", anomaly_count)

        emit_trace_event(
            "anomaly_analyze_complete",
            {"anomaly_count": anomaly_count, "anomaly_types": sorted({a["type"] for a in detected})},
            state,
        )

        return {
            # ADR-005: structured fields travel as JSON strings for msgpack-safe checkpointing.
            "detected_anomalies": to_json(detected),
            "anomaly_count": anomaly_count,
            "status": AgentStatus.SUCCESS.value,
        }

    # ------------------------------------------------------------------

    def _effective_thresholds(self, state: AgentState) -> Dict[str, float]:
        """Declared config, then the caller's already-validated per-request overrides.

        Every value here has passed ANOMALY_CONFIG_BOUNDS — at construction time
        for the declared ones, in IngestValidateNode for the caller's — so no
        non-finite or out-of-range number can reach a comparison.
        """
        effective = dict(self._thresholds)
        overrides = from_json(state.get("caller_thresholds"), {})
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                if key in ANOMALY_CONFIG_BOUNDS and isinstance(value, (int, float)):
                    number = float(value)
                    low, high = ANOMALY_CONFIG_BOUNDS[key]
                    if math.isfinite(number) and low <= number <= high:
                        effective[key] = number
        return effective
