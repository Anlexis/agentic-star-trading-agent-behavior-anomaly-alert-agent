"""AgentCore Platform v1.0"""

# FIN-C2-105 — caller-context bridge for the Cat 2 two-layer nested graph.
#
# WHY THIS EXISTS
# ---------------
# The framework's GraphNode.execute() calls
#     subgraph.invoke(user_input, session_id=..., ctx=...)
# and passes NO input_context. BaseGraph.invoke() therefore seeds the inner
# graph with `input_context: {}` — so caller context that reached the outer
# graph is silently absent inside the inner one, and every inner node reading it
# sees an empty mapping. Unit tests that build inner state by hand never notice,
# because they hand the node the context directly. Only an end-to-end invoke
# shows it, which is why the regression test for this lives in the E2E suite.
#
# The bridge is a ContextVar rather than a State field on purpose:
#   * GraphNode.extract_input() may only return the inner `user_input` string;
#     there is no return channel for extra state.
#   * BaseGraph.invoke() exposes exactly one subclass hook that runs while the
#     inner initial state is being built — _extra_initial_state() — and it takes
#     no arguments, so the value has to travel out of band.
#   * extract_input() and invoke() run in the same call stack, so a ContextVar is
#     correctly scoped: concurrent requests on separate threads/tasks each see
#     their own value and there is no shared mutable state.
#
# extract_input() stashes on EVERY call, including with an empty context, so a
# later request can never observe an earlier one's value.

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Dict, Iterator, Mapping

_CALLER_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("fin_c2_105_caller_context", default={})


def stash_caller_context(value: Mapping[str, Any] | None) -> None:
    """Bind the caller context for the enclosing node call (outer graph side)."""
    _CALLER_CONTEXT.set(dict(value or {}))


def current_caller_context() -> Dict[str, Any]:
    """Return the caller context stashed by the outer graph (inner graph side)."""
    return dict(_CALLER_CONTEXT.get())


@contextmanager
def caller_context(value: Mapping[str, Any] | None) -> Iterator[None]:
    """Scoped binding used by tests that drive the inner graph on its own."""
    token: Token[Dict[str, Any]] = _CALLER_CONTEXT.set(dict(value or {}))
    try:
        yield
    finally:
        _CALLER_CONTEXT.reset(token)
