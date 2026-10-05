# Prosper Challenge — Context Management

Voice AI for healthcare scheduling. An agent is a **graph of nodes** (Pipecat Flows), defined declaratively as JSON and compiled into a runnable voice pipeline.

- **Phase 1** — a UI to edit the node graph and place a test call.
- **Phase 2** — a context-management approach so a scheduling agent can reliably and cost-effectively navigate a large catalog of locations, doctors, and appointment types.

```
browser mic  ->  ElevenLabs STT  ->  OpenAI LLM  ->  ElevenLabs TTS  ->  browser
```

Pipecat's dev runner ships a **prebuilt browser client**, so the test-call UI comes for free — no frontend code to write yet.

## Quickstart

Requires **Python 3.11+**. Run from the repo root:

```bash
make install
make run
```

Open **`http://localhost:7860/builder`** to edit agents and place a test call (click **Test call**, then **Connect** in the call panel). The bare prebuilt client is still at `/client`. `Ctrl+C` to stop. (`make help` lists all targets.)\
\
Remember to update the `.env` file accordingly.

## Layout

| Path | Responsibility |
| --- | --- |
| `backend/bot.py` | The voice pipeline (WebRTC + ElevenLabs STT/TTS + OpenAI LLM). Loads an agent JSON via `AgentBuilder` and runs it. No graph logic lives here. |
| `backend/agent_builder/` | All agent-building code. `schema.py` = the declarative `AgentConfig` / `Node` / `Edge` contract; `builder.py` = `AgentBuilder`, which loads + validates the JSON and compiles it into a Pipecat Flows graph. |
| `backend/example_flow.json` | The example agent **as data** — a clinic scheduler. The starting point for the Phase 2 context-management work. |
| `backend/data/catalog.json` | A deliberately large, deliberately messy clinic catalog (locations, providers, appointment types, booking rules) for the Phase 2 work. See [`backend/data/README.md`](backend/data/README.md). |

| `backend/agents/` | Agent definitions (JSON). `scheduler.json` is the Phase 2 scheduling agent. The builder UI picks which one a test call runs. |
| `backend/catalog/` | Phase 2: catalog engine (matching + policy solver + mock availability), LLM tools and edge guards, STT vocabulary, Jev resolver (`jev.py`). |
| `backend/data/enrichment.json` | Offline enrichment: lay aliases, descriptions, confusable groups + clarifying questions. |
| `backend/builder_api.py`, `backend/ui/` | Builder UI backend + single-page editor with live call trace. |
| `backend/sim/` | Text-mode flow runner, full-catalog baselines, 15 scenarios, eval harness (`make eval`). |
| `backend/tests/` | Deterministic engine tests (`cd backend && .venv/bin/python -m pytest -q`). |

See **[solution.md](solution.md)** for the design, trade-offs and eval results, and **[docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md)** for how it evolved.