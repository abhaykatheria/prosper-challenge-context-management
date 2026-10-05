# Prosper Challenge — Context Management

A voice AI scheduling agent for a multi-specialty clinic, plus a builder UI to
edit it and place test calls.

- **Phase 1 — builder UI:** edit the agent's conversation graph in the browser and
  place a voice test call with a live trace (tool calls, guard decisions, tokens,
  cost, response time).
- **Phase 2 — context management:** the agent never sees the catalog (8 locations,
  50 providers, 82 appointment types) as text. A deterministic catalog engine
  answers narrow tool calls, booking rules are enforced in code, and a hybrid
  matcher (local fuzzy shortlist + [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev))
  works out what the caller means.

Against the obvious approach (the whole catalog in the prompt), on simulated
callers: **100% vs 57% bookings correct, 0 vs 8 rule-breaking bookings, ~0.9k vs
~11.1k prompt tokens per turn.** Design, all approaches compared, and results:
**[solution.md](solution.md)**. How it evolved: **[docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md)**.

```
browser mic → WebRTC → ElevenLabs STT → OpenAI LLM ⇄ catalog engine (tools + guards) → ElevenLabs TTS → browser
```

## Setup

Put your keys in `backend/.env` (copy `backend/.env.example`):

| Variable | Needed for |
|---|---|
| `OPENAI_API_KEY` | The agent's LLM, and the evaluations |
| `ELEVENLABS_API_KEY` | Speech-to-text and text-to-speech |
| `TYPESAFE_API_KEY` | Optional. The Jev-based matchers; without it they fall back to the local fuzzy matcher |

## Run locally

Requires **Python 3.11+**. From the repo root:

```bash
make install
make run
```

Open **http://localhost:7860/builder**, click **Test call**, then **Connect** in the
call panel and allow the microphone. `Ctrl+C` stops the server; `make help` lists all targets.

## Run with Docker

```bash
cp backend/.env.example backend/.env    # add your keys
docker compose up --build               # or: make docker → http://localhost:7860/builder
```

Voice calls use WebRTC, which needs **host networking**: native on Linux; on Docker
Desktop (Mac/Windows 4.34+) turn on *Settings → Resources → Network → Enable host
networking* and restart Docker Desktop. Without it, `make docker-bridge` serves the
builder UI, catalog search, tests and evals on port 7860, but calls may not connect.
Builder edits (`backend/agents/`) and call traces (`backend/logs/`) persist on the host.

## Using the builder

- **Graph:** click a step to edit its prompt, lookup tools, exits and guards; **Save**
  validates the agent. **Raw JSON** for direct edits.
- **Agent settings:** model, voice, and the **catalog matcher** (`hybrid` default;
  also `fuzzy`, `jev`, `cascade`).
- **Test call:** the trace beside the call shows each step change, tool call (labelled
  fuzzy / hybrid / prefetch), guard decision, tokens, cost and per-turn response
  time. Traces are also saved to `backend/logs/trace.jsonl`.
- **Catalog search:** type what a caller might say and see exactly what the tools return.

## Tests and evaluations

From `backend/` (or prefix with `docker compose run --rm scheduler`):

```bash
.venv/bin/python -m pytest -q                                   # 40 deterministic tests
.venv/bin/python -m sim.eval --agents ours_hybrid,naive --model gpt-4.1-mini   # simulated callers, 20 scenarios
.venv/bin/python -m sim.match_bench --bench holdout             # matcher accuracy on the frozen held-out set
```

`make eval` runs the agent against both full-catalog baselines. Every command
behind the numbers in `solution.md` is listed in its §8; raw results are in
`backend/sim/results/`.

## Layout

| Path | What it is |
| --- | --- |
| `backend/bot.py` | Voice pipeline and web server; loads the active agent, streams the live trace, measures response time. |
| `backend/agents/scheduler.json` | The scheduling agent, as data (steps, prompts, tools, guards, matcher). |
| `backend/agent_builder/` | Agent schema and compiler to Pipecat Flows: per-step tools, guarded exits, context reset, prefetch. |
| `backend/catalog/` | Catalog engine (`engine.py`: matching, policy solver, mock availability, booking), LLM tools and guards (`tools.py`), Jev client and matcher (`jev.py`), speech-to-text key terms (`vocab.py`). |
| `backend/data/catalog.json` | The clinic catalog provided with the challenge ([details](backend/data/README.md)). |
| `backend/data/enrichment.json` | Lay aliases, descriptions, and confusable groups with clarifying questions. |
| `backend/builder_api.py`, `backend/ui/` | Builder UI backend and single-page editor. |
| `backend/sim/` | Text-mode runner, full-catalog baselines, 20 scenarios, evaluation, matching benchmarks (dev + frozen held-out). |
| `backend/tests/` | Unit tests: policy exhaustiveness, matching, fallbacks, ambiguity logic. |
| `backend/example_flow.json` | The starter's example agent (kept for reference). |
| `Dockerfile`, `docker-compose.yml` | Container setup. |
| `solution.md`, `docs/` | Write-up and development log. |
