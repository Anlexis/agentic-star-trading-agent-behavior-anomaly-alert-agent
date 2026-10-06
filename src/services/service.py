"""AgentCore Platform v1.0"""

# Service layer for FIN-C2-105.
#
# Holds the caller-data validation primitives shared by the domain nodes:
# finite/bounded numeric parsing, inert-identifier locking, the injection
# screen, and the credential scan. Nodes call these; nothing here reaches an
# external system, holds credentials, or makes a routing decision.
#
# Why these live in one place: every node that touches caller data must apply
# the SAME rules. A per-node copy drifts, and a drifted screen is a bypass.

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from framework.security.credential_detector import detect_credentials_in_value

# ---------------------------------------------------------------------------
# Rejection type
# ---------------------------------------------------------------------------


class InputRejected(ValueError):
    """Caller input failed validation.

    Carries the offending FIELD NAME only. The rejected value is never stored
    on the exception, so it cannot reach a log line or an error envelope by
    accident.
    """

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


# ---------------------------------------------------------------------------
# Inert identifiers
# ---------------------------------------------------------------------------

# Every caller-supplied string that is rendered into the alert is locked to this
# alphabet. It admits real instrument / desk / algorithm codes (JGB10Y, 7203.T,
# US-T-10Y, algo_alpha) and admits nothing that could restructure the rendered
# report: no whitespace, no newline, no quote, no angle bracket, no brace.
# A newline in a rendered value is how a forged "step" gets manufactured inside
# an otherwise trustworthy document, so the exclusion is deliberate.
_INERT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,32}$")

# The platform's S-2 privacy filter replaces personal-data-shaped values with
# this sentinel BEFORE any template code runs. A masked value is not an
# extracted value: certifying it as a real identifier would put
# ``instrument_id: "[MASKED]"`` into a regulatory alert as though it were an
# instrument. Detected explicitly so the caller gets an actionable message.
MASK_SENTINEL = "[MASKED]"

# Absolute magnitude ceiling for any caller-supplied number. Well above every
# plausible trading-log value, well below the point where float arithmetic
# stops being meaningful.
MAX_ABS_NUMERIC = 1e12


def inert_identifier(value: Any, *, field: str) -> str:
    """Return ``value`` if it is an inert identifier, else raise InputRejected.

    Rejects the redaction sentinel with its own reason so the caller can tell
    "the platform redacted this before we saw it" from "you sent junk".
    """
    if not isinstance(value, str):
        raise InputRejected(field, f"must be a string, got {type(value).__name__}")
    if MASK_SENTINEL in value:
        raise InputRejected(
            field,
            "value was redacted by the platform privacy filter before the agent "
            "received it; identifier fields must not contain personal data",
        )
    if not _INERT_ID_RE.match(value):
        raise InputRejected(
            field,
            "must match [A-Za-z0-9_.:-]{1,32} (identifier codes only, no free text)",
        )
    return value


def finite_in_range(value: Any, *, field: str, low: float, high: float) -> float:
    """Parse a caller-supplied number, fail CLOSED on anything not finite/in range.

    ``float("nan")`` and ``float("inf")`` parse without error, and every
    comparison against NaN is False — so an unchecked non-finite threshold
    silently turns a detection rule OFF. Booleans are rejected because
    ``isinstance(True, int)`` is True in Python and ``float(True) == 1.0``.
    """
    if isinstance(value, bool):
        raise InputRejected(field, "must be a number, got a boolean")
    if isinstance(value, str):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise InputRejected(field, "must be a finite number") from None
    elif isinstance(value, (int, float)):
        parsed = float(value)
    else:
        raise InputRejected(field, f"must be a number, got {type(value).__name__}")

    if not math.isfinite(parsed):
        raise InputRejected(field, "must be finite (NaN and Infinity are refused)")
    if not (low <= parsed <= high):
        raise InputRejected(field, f"must be within [{low}, {high}]")
    if abs(parsed) > MAX_ABS_NUMERIC:
        raise InputRejected(field, "magnitude exceeds the supported range")
    return parsed


# ---------------------------------------------------------------------------
# Injection screen
# ---------------------------------------------------------------------------

# Chat-template control tokens as a CLASS, not as three literals. The framework
# detector scores `<|im_start|>` and `[INST]` high but returns nothing at all for
# `<<SYS>>`, which is exactly the variant a phrase-based screen misses.
_CONTROL_TOKEN_RE = re.compile(
    r"<\|[^|>\n]{1,40}\|>"  # <|im_start|>, <|im_end|>, any ChatML-style token
    r"|\[/?\s*(?:INST|SYS)\s*\]"  # [INST] [/INST] [SYS]
    r"|<</?\s*SYS\s*>>"  # <<SYS>> <</SYS>>
    r"|<\s*/?\s*(?:system|user|assistant)\s*>",
    re.IGNORECASE,
)

# Directive phrasing. Anchored on the verb+object pair rather than a bare verb so
# ordinary trading prose ("do not follow up with the desk") does not trip it.
_DIRECTIVE_RE = re.compile(
    r"(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+)?"
    r"(?:previous|prior|above|earlier|system)\s+(?:instruction|prompt|rule|context)",
    re.IGNORECASE,
)

# Markup strip used for the SECOND pass. A sanitizer is not a refusal: stripping
# `<b>` out of `ig<b>nore previous instructions` re-assembles a directive that
# the raw pass cannot see, and stripping `<|im_start|>` would remove the very
# token the first pass needs. So both passes run, and neither is allowed to
# replace the other.
_MARKUP_RE = re.compile(r"<[A-Za-z/][^<>\n]{0,60}>")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")


def _normalize(text: str) -> str:
    """NFKC-fold and drop zero-width characters before pattern matching."""
    return _ZERO_WIDTH_RE.sub("", unicodedata.normalize("NFKC", text))


def _screen_text(text: str, *, field: str) -> None:
    raw = _normalize(text)
    stripped = _MARKUP_RE.sub("", raw)
    for pass_name, candidate in (("raw", raw), ("post-sanitize", stripped)):
        if _CONTROL_TOKEN_RE.search(candidate):
            raise InputRejected(field, f"contains a chat-template control token ({pass_name} scan)")
        if _DIRECTIVE_RE.search(candidate):
            raise InputRejected(field, f"contains an instruction-override directive ({pass_name} scan)")


def screen_injection(value: Any, *, field: str, _depth: int = 0) -> None:
    """Depth-first injection screen over a parsed payload — KEYS included.

    Scanning the PARSED structure (not the raw JSON text) is what makes
    ``\\u003c|im_start|\\u003e``-style escapes ineffective: by the time this runs
    the escape has already been decoded into the string it denotes.
    """
    if _depth > 12:
        raise InputRejected(field, "structure is nested too deeply")
    if isinstance(value, str):
        _screen_text(value, field=field)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                _screen_text(key, field=f"{field}.<key>")
            screen_injection(item, field=f"{field}.{key}", _depth=_depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            screen_injection(item, field=f"{field}[{index}]", _depth=_depth + 1)


# ---------------------------------------------------------------------------
# Credential scan (framework floor UNION local patterns)
# ---------------------------------------------------------------------------

# Local patterns describe credential ASSIGNMENTS; the framework's patterns
# describe credential FORMATS. Neither set is a superset of the other, so the
# gate takes the UNION. Replacing these with the framework detector alone would
# make the gate NARROWER while looking like a tightening — `password=hunter2...`
# matches no framework pattern (verified against the shipped wheel).
_LOCAL_CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "assigned_secret",
        re.compile(
            r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|private[_-]?key|client[_-]?secret)"
            r"\s*[:=]\s*\S{6,}",
            re.IGNORECASE,
        ),
    ),
    ("pem_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("basic_auth_url", re.compile(r"https?://[^\s/@]{1,64}:[^\s/@]{1,64}@")),
    # Optional scheme word ("Basic", "Token", …) between the colon and the value.
    ("authorization_header", re.compile(r"authorization\s*[:=]\s*(?:\S+[ \t]+)?\S{8,}", re.IGNORECASE)),
]


def credential_findings(value: Any) -> list[str]:
    """Return the credential TYPES found in ``value`` — framework floor + local set.

    Only type names are returned; the matched text is never surfaced, so a
    finding can be logged and reported without re-publishing the secret.
    """
    types = {finding["type"] for finding in detect_credentials_in_value(value)}
    for name, pattern in _LOCAL_CREDENTIAL_PATTERNS:
        if _any_string_matches(value, pattern):
            types.add(name)
    return sorted(types)


def _any_string_matches(value: Any, pattern: re.Pattern[str]) -> bool:
    if isinstance(value, str):
        return bool(pattern.search(value))
    if isinstance(value, Mapping):
        return any(_any_string_matches(item, pattern) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_any_string_matches(item, pattern) for item in value)
    return False


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------


def bounded_sequence(value: Any, *, field: str, max_items: int) -> Sequence[Any]:
    """Return ``value`` as a list, refusing non-lists and over-long lists."""
    if not isinstance(value, (list, tuple)):
        raise InputRejected(field, f"must be a list, got {type(value).__name__}")
    if len(value) > max_items:
        raise InputRejected(field, f"exceeds the maximum of {max_items} entries")
    return list(value)


def contains_mask_sentinel(values: Iterable[Any]) -> bool:
    """True when any string leaf is (or embeds) the platform redaction sentinel."""
    return any(isinstance(v, str) and MASK_SENTINEL in v for v in values)


# ---------------------------------------------------------------------------
# Closed-set refusal notice
# ---------------------------------------------------------------------------

# Substituted for the agent's output whenever a submission is refused. It is a
# FIXED string: it quotes no caller text, no exception message, no traceback and
# no source path, and it is TRUTHY so it cannot re-open
# AgentBaseGraph.get_output()'s `formatted_output or result` fallback onto
# un-gated state. Every refusal path publishes this same value, so a caller sees
# one shape whether the submission was refused at the entry node or inside the
# domain workflow.
REFUSAL_NOTICE = json.dumps(
    {
        "status": "refused",
        "reason": "trade_log_rejected",
        "detail": "The submitted trade log could not be processed. No alert was produced.",
    },
    ensure_ascii=False,
)
