"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for FIN-C2-105.
# Entry points are adapters only — no business logic here.
# For platform-level routing, AgentGateway calls agent.invoke() directly and
# supplies the trust level itself; this module covers the standalone deployment.
#
# WHAT THIS ADAPTER IS RESPONSIBLE FOR
# ------------------------------------
# 1. TRUST. The manifest declares required_trust_level: VERIFIED_EXTERNAL, and the
#    framework's S-1 gate reads state["caller_trust_level"], which comes from the
#    InvocationContext this adapter builds. An adapter that never sets it leaves
#    every caller at ANONYMOUS, so the first domain node refuses the request and
#    the deployed agent cannot serve anything. The bearer token is therefore read
#    here. Both credentials are accepted, because the STG evidence harness
#    presents STG_INTERNAL_RUNNER_TOKEN for an INTERNAL entry contract and
#    INVOKE_AUTH_TOKEN otherwise.
#
# 2. THE input_context CHANNEL, which is sharper than it looks:
#      * unknown keys are DROPPED, not ignored. A validator that merely ignores
#        an undeclared key leaves it in state["input_context"], from where
#        InitializeNode._setup() returns it verbatim into its own result — and
#        the framework's @final S-3 gate scans every value of every result. A
#        credential-shaped string anywhere in input_context therefore makes the
#        FIRST node fail with a traceback, before any template code runs.
#        "My contract only declares inert identifiers" is not immunity; dropping
#        the undeclared keys is.
#      * what survives the drop is credential-screened with the framework's own
#        detect_credentials_in_value, so this refusal set matches the gate's block
#        set exactly and cannot drift from it. The request cannot succeed either
#        way, so a readable 400 naming the FIELD beats an opaque node-1 error.
#      * the scan is per field only so the offending field can be named. That is
#        exactly equivalent to scanning the whole mapping, because
#        detect_credentials_in_value(dict) is defined as the union over its
#        values — pinned as a property test in the suite.
#      * 400, not 422: pydantic owns 422 and answers it with a list of error
#        objects, so reusing it would make client handling ambiguous.
#      * the field NAME is caller data too, and is echoed only when it is itself
#        safe to print.

import hmac
import os
import re
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.security.credential_detector import detect_credentials_in_value
from shared.secrets import factory as secrets_factory
from src.graph.graph import Graph
from src.nodes.ingest_validate_node import CONTEXT_FIELDS

AGENT_NAME = "TradingAnomalyAlertAgent"
NAMESPACE = "fin"

# Submission ceiling for the trade-log payload itself. The adapter cap exists so
# an oversized body is refused before it is parsed, not after.
MAX_INPUT_CHARS = 262_144

# A field name is only echoed back when it looks like a field name.
_SAFE_FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")

app = FastAPI(title=f"FIN-C2-105 {AGENT_NAME}")

agent = Graph()
agent.compile()
agent.provision_secrets(secrets_factory(namespace=NAMESPACE, agent_name=AGENT_NAME))


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    input_context: Optional[Dict[str, Any]] = None


def _resolve_trust_level(request: Request) -> TrustLevel:
    """Trust level for this request: platform middleware first, bearer token second.

    Falls back to ANONYMOUS, which the S-1 gate then refuses — the failure mode of
    a misconfigured deployment is a refusal, never an unauthenticated success.
    """
    from_middleware = getattr(request.state, "trust_level", None)
    if isinstance(from_middleware, TrustLevel):
        return from_middleware

    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        return TrustLevel.ANONYMOUS

    internal_token = os.environ.get("STG_INTERNAL_RUNNER_TOKEN", "")
    if internal_token and hmac.compare_digest(presented, internal_token):
        return TrustLevel.INTERNAL

    external_token = os.environ.get("INVOKE_AUTH_TOKEN", "")
    if external_token and hmac.compare_digest(presented, external_token):
        return TrustLevel.VERIFIED_EXTERNAL

    return TrustLevel.ANONYMOUS


def _safe_field_label(name: str, index: int) -> str:
    """A printable label for a caller-supplied field name."""
    if _SAFE_FIELD_NAME_RE.match(name) and not detect_credentials_in_value(name):
        return f"input_context.{name}"
    return f"input_context field #{index}"


def build_input_context(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Whitelist, drop, and credential-screen the caller context.

    Raises HTTPException(400) naming the offending field — never its value, and
    never the matched text of a finding.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="input_context must be an object")

    context: Dict[str, Any] = {}
    for index, (name, value) in enumerate(raw.items(), start=1):
        if name not in CONTEXT_FIELDS:
            # Dropped, not ignored — see the module docstring.
            continue
        if detect_credentials_in_value(value):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{_safe_field_label(name, index)} looks like a credential and was refused. "
                    "Remove it and retry; this agent never accepts secrets on the context channel."
                ),
            )
        context[name] = value
    return context


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Dict[str, Any]:
    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"input exceeds the {MAX_INPUT_CHARS}-character submission limit",
        )

    input_context = build_input_context(req.input_context)

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=_resolve_trust_level(request),
            caller_id=getattr(request.state, "caller_id", ""),
        )
        result: Dict[str, Any] = agent.invoke(req.input, ctx=ctx, input_context=input_context)
        return result


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "agent": AGENT_NAME}
