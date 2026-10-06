# AI Trading Behavior Anomaly Alert Agent

AI agent for detecting anomalous AI trading-agent behaviour and raising risk alerts, built with Agentic Star.

> **Category**: Cat 2 (Domain-specific multi-step workflow)
> **Industry**: Finance
> **Template ID**: FIN-C2-105

## Overview

Reviews an algorithmic trading desk's own execution log and reports the behaviour that a model-risk
reviewer would have to explain. Given a period's orders it applies five deterministic rules —
single-instrument concentration, confidence-score drift against the desk's own baseline, execution
latency spikes, forbidden policy tags and order-count ceilings, and correlated timing across
algorithms — and returns a structured alert: the findings, their evidence, an overall severity, the
regulatory article each finding maps to, and the action a compliance officer should take.

There is no model call anywhere in the pipeline, so the same log always produces the same alert and
every finding can be traced back to the rule and threshold that produced it. Raw financial values
(order amounts, position deltas, prices) are dropped at ingestion and never appear in the output;
the alert quotes computed ratios, counts and identifiers only. The result is advisory: it is
explicitly labelled as automated analysis requiring human compliance review.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails during
graph compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Calling the agent

`POST /invoke` takes the trade log as a JSON-encoded string in `input`, and optional caller
provenance in `input_context`. Authentication is a bearer token; without one the request is
refused, because the manifest declares `required_trust_level: VERIFIED_EXTERNAL`.

```json
{
  "input": "{\"trade_log_entries\": [...], \"baseline\": {...}, \"thresholds\": {...}}",
  "session_id": "review-2026-06",
  "input_context": {"desk_code": "tokyo_rates", "report_reference": "mrm_q2_review"}
}
```

Each trade-log entry requires `order_id`, `timestamp`, `instrument_id` and `agent_id`, and may
carry `confidence_score_bucket`, `latency_ms` and `policy_tags`. Every identifier is validated
against `[A-Za-z0-9_.:-]{1,32}` and every number must be finite and within its declared range; a
submission that fails either check is refused with a message naming the field. `thresholds` may
override any of the detection thresholds in `config/config.yaml` for a single request.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent manifest (agent.yaml) and runtime parameters (config.yaml)
deploy/       local deployment recipe and a sample invoke payload
docs/         design and operational documentation
```

See `docs/` for the design and the test specification.

## Customising

1. Adjust `config/config.yaml` — detection thresholds, forbidden policy tags and submission
   ceilings all take effect at runtime, and a malformed value stops the graph rather than
   silently reverting to a default.
2. Replace the sample payload in `deploy/invoke_payload.json` with a log from your own desk.
3. Review the rule implementations in `src/nodes/anomaly_analyze_node.py` and the regulatory
   article mapping in `src/nodes/risk_alert_compose_node.py` for your jurisdiction.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
