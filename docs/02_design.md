# Template Design Specification

## Position in AgentCore Architecture

| Property | Value |
|---|---|
| Agent Class | `TradingAnomalyAlertAgent` (`src.graph.graph.TradingAnomalyAlertAgent`) |
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Inner graph | `DomainWorkflowGraph` — `BaseGraph`, reached through a `GraphNode` in the `main` slot |
| Category | Cat 2 (two-layer nested) |
| Generation mode | deterministic — no language model is invoked |

- **Three-Layer Separation**:
  - State: flat TypedDict composition (no Pydantic — msgpack incompatible)
  - Node: L1 inheritance (Template Method: `execute(self, state: dict) -> dict` override only)
  - Graph: composition (`register_nodes()` for node substitution)

## Architecture Overview

### Node Configuration

**Outer graph** (`TradingAnomalyAlertAgent`, AgentBaseGraph backbone):

| Node | Responsibility | Input State | Output State | Inherits/Overrides |
|------|---------------|-------------|--------------|-------------------|
| initialize | Framework defaults (schema_version, session_id, trust_level) | — | `schema_version`, `session_id` | InitializeNode (default) |
| pre_process | Caller-contract validation: refuses injection, non-finite numerics and over-large submissions before any domain code runs | `user_input` | `validated_input` | `FunctionNode.execute()` |
| main | Delegates to the inner `DomainWorkflowGraph`; seeds it with the caller context | `validated_input` | `formatted_output`, `risk_alert`, `alert_severity` | `GraphNode` |
| post_process | Output containment gate — on violation returns ERROR **and clears every output-bearing field** | `risk_alert`, `formatted_output` | `formatted_output` (gated) | `FunctionNode.execute()` |
| finalize | Framework defaults (response_metadata, total_time_ms) | — | `response_metadata` | FinalizeNode (default) |

**Inner graph** (`DomainWorkflowGraph`, BaseGraph — custom linear topology):

| Node | Responsibility | Input State | Output State |
|------|---------------|-------------|--------------|
| ingest_validate | Parses and bounds-checks the trade log; validates caller thresholds against `ANOMALY_CONFIG_BOUNDS` | `validated_input` | `validated_trade_log`, `trade_log_metadata`, `caller_thresholds`, `validation_errors` |
| anomaly_analyze | Detects concentration, latency-spike, drift, coordination and policy-breach anomalies | `validated_trade_log`, `caller_thresholds` | `detected_anomalies`, `anomaly_count` |
| risk_alert_compose | Composes the structured alert, assigns severity, applies the domain S-3 output gate | `detected_anomalies`, `trade_log_metadata` | `risk_alert`, `alert_severity`, `fsa_article_ref`, `formatted_output` |

> The inner graph receives the caller's `input_context` through the ContextVar bridge in
> `src/graph/context_bridge.py`: SDK 1.0.1+ `GraphNode` does not forward `input_context`, so the
> outer `extract_input` stashes it and the inner graph's `_extra_initial_state` seeds it.
> Without the bridge every declared context value is silently absent inside the inner layer.

### Data Flow

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                            ↓ (retry)
                                          pre_process
```

### State Definition

Structured fields travel as JSON **strings** (msgpack-safe checkpointing); `to_json()` /
`from_json()` in `src/schemas/state.py` are the only sanctioned producers and consumers.

| Field | Type | Purpose | Required |
|-------|------|---------|----------|
| `user_input` | `str` | Raw trade log as submitted by the caller | Yes |
| `validated_input` | `str` | Sanitised trade log written by PreProcessNode | Yes |
| `validated_trade_log` | `str` (JSON) | Detection-ready trade log | Yes |
| `validation_errors` | `list[str]` | Structural validation messages; empty means clean | Yes |
| `trade_log_metadata` | `str` (JSON) | Aggregated metadata (period, instrument/entry counts, baseline); carries `desk_code` / `report_reference` only when the caller supplied them, both inert identifiers | Yes |
| `caller_thresholds` | `str` (JSON) | Per-request overrides, already bounds-checked | No |
| `detected_anomalies` | `str` (JSON) | Anomaly list — `type`, `severity`, `evidence`, `threshold` | Yes |
| `anomaly_count` | `int` | Convenience count of `detected_anomalies` | Yes |
| `risk_alert` | `str` (JSON) | Structured alert payload read back by the S-3 gate | Yes |
| `alert_severity` | `str` | `CRITICAL` \| `HIGH` \| `MEDIUM` \| `LOW` | Yes |
| `fsa_article_ref` | `str` | Relevant regulatory article reference | No |
| `formatted_output` | `str` | Final released string consumed by PostProcessNode | Yes |

No monetary value from the submitted log is ever written into `detected_anomalies.evidence` or
into the released alert — evidence carries counts, type names and threshold labels only.

### Output Precision Grid — not applicable

This template renders **no monetary aggregate**. Raw financial values (`order_amount`,
`position_delta`, `price`) are dropped at ingest, and the alert quotes only computed ratios,
counts and latencies. A round-to-the-nearest-1,000 grid is therefore **deliberately not
implemented**: applying one would corrupt the very ratios and millisecond figures the alert
exists to report (a concentration of `0.42` is not a rounding candidate, and a latency of
`900 ms` is not a currency amount).

The invariant enforced in its place is the domain one — no forbidden financial key may appear
in the released alert — checked in `RiskAlertComposeNode`, with the credential and size checks
one layer out in `PostProcessNode`.

### Two output layers, deliberately non-overlapping

`RiskAlertComposeNode` checks the **domain** invariant; `PostProcessNode` checks **credentials
and released size**. Neither duplicates the other, so each stays falsifiable: disabling the
credential scan fails the credential tests, disabling the domain gate fails the domain tests.
Had both layers checked both things, each would have contained the other's mutant and both sets
of tests would have looked decorative — more defence buying less assurance. The framework's own
`@final` S-3 scan runs under both as the floor.

**State Constraints (mandatory):**
- Flat TypedDict only (primitives + JSON-serializable types)
- No JWT, API keys, credentials in State (checkpoint DB leakage)
- InvocationContext via `config["configurable"]` only (not in State)
- No Pydantic models, dataclass, arbitrary Python objects (msgpack incompatible)

## Framework Utilization

### Shared Components Used
- [x] InvocationContext (correlation_id, session_id, permissions, credential handle)
- [x] SecurityViolationError
- [x] S-2: `_extra_security_gate_input()` — domain-specific input check hook
      (runs after the default PII scan; implement any domain checks needed — e.g. PII scan
      on additional fields, consent flag validation, input size limits, business rule gates;
      omit if the default framework scan on `user_input` / `validated_input` / `llm_response`
      is sufficient; **MUST NOT override `_security_gate_input()`** — `TypeError` at class definition)
- [x] S-3: `_extra_security_gate_output()` — domain-specific output check hook
      (runs after the default credential scan; implement any domain checks needed — e.g.
      credential scan on nested fields, PII re-check on LLM output, content filtering,
      preservation verification; omit if the default scan on result string values is sufficient;
      **MUST NOT override `_security_gate_output()`** — `TypeError` at class definition)
- [x] S-4: `emit_trace_event()` — at least one domain-specific event inside each `execute()`
      (**mandatory**; do NOT emit `node_start` / `node_complete` / `node_error` —
      `BaseNode.__call__()` emits these automatically; duplicates corrupt audit trail)

> **S-2/S-3 gate behaviour by node type (ADR-017):**
> - `FunctionNode` subclass → framework `@final` gate always runs automatically;
>   extend via `_extra_security_gate_input()` / `_extra_security_gate_output()` only
> - `GraphNode` / `RemoteAgentNode` → deliberate no-op (upstream or remote node's gate already applied)
> - Custom `BaseNode` subclass → must implement `_security_gate_input()` and
>   `_security_gate_output()` directly (`@abstractmethod` — omission raises `TypeError` at instantiation)

### Composition Pattern

- **Pattern**: GraphNode (subgraph) — the `main` slot wraps the inner `DomainWorkflowGraph`
- **Composition target**: `src.graph.domain_workflow_graph.DomainWorkflowGraph`
- **Error propagation strategy**: handle — a refusal or a gate violation returns an ERROR
  envelope with every output-bearing field cleared, never a partially-released alert

## Import Isolation Confirmation
- [x] Template does not import the platform SDK (Level 0)
- [x] Import targets: `framework/` and `shared/` only (no `agents/base/` required)

Enforced by `tests/proof_of_boundary/test_import_isolation.py` (AST scan, 0 violations).

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | The workflow is a fixed pipeline — ingest, analyse, compose. There is no goal the agent has to plan its own way to, so the autonomous loop would add a non-determinism the regulatory use case cannot accept. |
| Composition pattern | Standalone | GraphNode (subgraph) | **GraphNode** | Keeps the domain pipeline behind the backbone's own validation and containment nodes, so the output gate sees every result the inner graph produces. |
| Threshold source | Hard-coded | `config/config.yaml` + per-request override | **Both, one bounds table** | Operators tune without a redeploy and callers override per request, while `ANOMALY_CONFIG_BOUNDS` stops either route reaching an unsafe value. |
| Structured state fields | Native dict / list | JSON strings | **JSON strings** | msgpack-safe checkpointing; a nested structure in a checkpointed field is a state-safety violation. |
| Over-large submission | Truncate | Refuse | **Refuse** | A truncated log yields an alert about a log nobody sent — worse than no answer. |
