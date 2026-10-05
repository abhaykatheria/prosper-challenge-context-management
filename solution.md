# Solution — Voice Agent Builder + Catalog Context Management

## TL;DR

**The scheduling agent never sees the catalog as text.** A deterministic catalog
engine (retrieval + policy solver) answers narrow tool calls with small,
pre-validated results. Booking rules are enforced in code at graph edges
("guards"), so an invalid booking can't happen, and every refusal comes with
alternatives. Each conversation step sees only its own tools and a short summary
of known facts (~1k prompt tokens per turn). To understand what the caller means,
a local fuzzy matcher shortlists appointment types and
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (TypeSafe
AI's typed-decision model) picks within the shortlist and decides when to ask a
clarifying question.

| simulated callers, same LLM for all | **This design** | Catalog in prompt | Catalog in prompt + guards |
|---|---|---|---|
| Bookings correct | **100%** (30/30) | 57% | 73% |
| Bookings breaking a clinic rule | **0** | 8 | 0 |
| Prompt tokens per LLM turn | **~0.9k** | ~11.1k | ~11.1k |
| Cost per call, uncached / cached | **$0.0064 / $0.0058** | $0.0366 / $0.0106 | $0.0414 / $0.0120 |

Final build: 40/40 on 20 scenarios with 0 rule violations; **50/60** on a frozen
held-out set of caller phrasings written by a different model (vs 36.7 for "give
Jev the whole catalog", 22 for fuzzy alone); median **1.8 s** response time on live calls.

```bash
make install && make run                      # http://localhost:7860/builder → Test call
docker compose up --build                     # same, in Docker (voice needs host networking; see README)
cd backend && .venv/bin/python -m pytest -q   # 36 tests
```
How the design evolved, including dead ends: [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md).

## 1. The problem

Pasting 8 locations × 50 providers × 82 appointment types (~11k tokens) into the
prompt is expensive (re-sent on all ~15 turns of a call, and slower to first
token) and inaccurate: the model must cross-check five things in its head (who
does the type, where they work, which sites have the equipment, new-patient
rules, referrals). The catalog is built to break that:

| Trap | Example |
|---|---|
| Duplicate names | 2× Maria Garcia (Pediatrics / Family Medicine), 2× Michael Sato (one not accepting new patients), 7 Chens |
| Near-duplicate types | Annual Physical vs Annual Wellness Visit; MRI Brain / Spine / Knee; three kinds of follow-up |
| Types nobody offers | All eye exams, physical therapy, urology |
| Equipment ≠ bookable | Mission District has imaging but no radiologist: MRIs only at Downtown and Midtown |
| Confusable sites (on a phone line) | Mission **Bay** / Mission **District**; North **Beach** / North **Gate** |

## 2. Approaches compared

**Built and measured.** Agent-level numbers: 15 scenarios × 2, gpt-4.1-mini.
Matcher numbers: correct on the first try out of 60 held-out phrasings, mean of 3 runs.

| Approach | Result | Verdict |
|---|---|---|
| **Whole catalog in the prompt** | 57% correct, 8 rule-breaking bookings, ~11k tokens/turn | Fails exactly on the traps above; burns rate limit (~3 turns/min at 30k TPM). |
| **Whole catalog + guards** | 73% correct, 0 violations | Guards stop illegal bookings, but the model still picks the wrong Sato, wrong dentist, etc. Guards fix *safety*, not *accuracy*. |
| **Tools + guards + per-step context**, fuzzy matcher | 100% correct, 0 violations, ~0.9k tokens/turn | The core design. Its matcher is overfit to my own wording: 66/90 on my phrasings, **22/60** held-out. |
| **Jev with the whole catalog** | 36.7/60, never guesses on ambiguous requests, ~400 ms, $0.0009/lookup | Most cautious but asks the most needless questions, and **doesn't scale**: the catalog uses 21.8k of Jev's measured ~32.8k-token limit. |
| **Jev with the option list only** | 68/90 dev, commits on ambiguity | Less context, less accurate. |
| **Cascade:** fuzzy if confident, else Jev | 69/90 dev | Inherits fuzzy's confident-but-wrong picks ("a skin check"). |
| **Hybrid v1:** fuzzy shortlist → Jev picks, plus one global "is it specific?" question | 45/60, 4 guesses | The global yes/no score is noise with only a shortlist in view. |
| **Hybrid v2:** shortlist → Jev picks, plus a "which one, or not specified?" question per confusable group, checked against the caller's words | **50/60, 1 guess**, ~265 ms, $0.00016/lookup | **Chosen.** Most accurate on unseen data, few needless questions, and Jev context is ~3.8k tokens at any catalog size. |

**Considered, not built:**

| Approach | Why not |
|---|---|
| Embeddings / RAG over the catalog | Solves recall, not constraints: retrieved chunks don't say which (provider, site) pairs can do an MRI for a new patient. Fuzzy recall is already 81/82 in the top 5, so embeddings only pay off at much larger catalogs; they'd slot into the retrieval step. |
| A bigger, long-context LLM with the whole catalog | Not measured. Might narrow the accuracy gap, but pays the full-catalog cost and latency on every turn of a voice call, and still needs guards for safety. |
| LLM writes queries (text-to-SQL) over the catalog | Flexible, but adds a multi-step loop (latency) and a new error surface; the rules still need enforcing in code. |
| Two-stage lookup (specialty → type) | Jev's documented answer for more than 255 options; unnecessary at 82. It becomes the hybrid's escape path at scale. |
| Shard the context by city / region | All 8 sites are in one city; 34/50 providers work at several sites; the location is usually unknown when the service is chosen ("soonest anywhere"). Good outer filter for a multi-city system only. |
| LLM-written summaries between steps | Adds latency and cost per step and can drop facts; code-built summaries are exact and free. |
| One big step with every tool | Every turn carries every tool, and the model can skip steps (book before checking the referral). |
| Fine-tuning on the catalog | The catalog changes daily; it must be data, not weights. |

## 3. The chosen design

```
 caller ─► WebRTC ─► STT (ElevenLabs + catalog key terms) ─► LLM (gpt-4.1-mini) ─► TTS (ElevenLabs Flash) ─► caller
                                                               │ ▲  function call / short result
                          ┌─────────── current step: its lookup tools + guarded exits ───────────┐
                          ▼                                                                       ▼
   find_appointment_types · find_providers · find_locations · search_slots      select_service · hold_slot · book_appointment
                          │                                                     refuse → reason + alternatives
                          ▼                                                                       │
          Catalog engine (Python, in memory): hybrid resolver · policy solver · availability · booking ◄┘
                          └── catalog.json + enrichment.json (aliases, confusable groups + questions)
```

* **The agent is data** (`backend/agents/scheduler.json`), compiled by `AgentBuilder`
  into a Pipecat Flows graph: `greeting → identify_patient → find_service →
  offer_slots → confirm_booking → booked`, plus Q&A. New schema fields, all editable
  in the UI: per-step `tools`, `context: "reset"` (start the step from a code-built
  summary of known facts), `prefetch` (the server runs an obvious lookup as the step
  begins, saving an LLM round trip), guarded edges (`action`, `fallback_target`), and
  the agent's `resolver`.
* **Policy solver** lists every valid (provider, site) pair or returns blockers with
  repairs ("Dr. David Chen isn't taking new patients → Dr. Emily Chen is").
  Patient facts (new/existing, referral, language, age from DOB) are applied by
  every tool. Booking uses opaque slot handles and re-validates everything.
* **Enrichment** (`data/enrichment.json`): lay aliases, descriptions, and 18
  confusable groups with one curated question each, generated offline in
  production, never at call time. Each type's *discriminating words* (Fasting
  Blood Test ← fasting, cholesterol) are derived from it.
* **Hybrid resolver** (`catalog/tools.py`, `catalog/jev.py`): fuzzy ranks all types
  (~3 ms) and keeps the top 12 plus their confusable siblings → exact matches
  without look-alikes skip Jev → otherwise Jev picks within the shortlist and, in the
  same request, answers "which one, or not specified?" per group → the agent commits
  only if the caller used a word that singles the option out (or Jev is very sure),
  else asks the group's question. Category requests ("some kind of scan") become a
  question. Requests over 30k tokens are refused before sending, and any Jev error
  or timeout falls back to fuzzy.

## 4. Builder UI (Phase 1)

`/builder`, one vanilla-JS page on the bot's server (Pipecat's prebuilt call
client kept, per scope): agent list, a live Mermaid graph with a step editor
(prompt, tools, exits, guards, context mode, resolver), validation on save, a
catalog-search playground, and **Test call** with a live trace of steps, tool
calls, guard decisions, tokens, cost and per-turn response time.

## 5. How it was evaluated

The simulator runs the same compiled graph over text; a small LLM plays patients
with private facts; **code, not an LLM, scores** every booking against the expected
type/provider/site and re-checks every clinic rule using the patient's true facts.
There are 20 scenarios, one per trap. Matchers are measured separately on a 90-phrase
dev set (used for tuning) and a **60-phrase held-out set generated by gpt-4.1 and
committed before any tuning** (`f8aefa0`). Jev results are 3-run means because its
answers vary slightly.

End to end, matcher choice barely matters (fuzzy 40/40, hybrid 40/40, full-catalog
Jev 39/40), because the conversational LLM rewrites "my chest hurts on stairs" into
"cardiology consultation" before any lookup. The matcher earns its keep on the long
tail of odd phrasings and speech-to-text errors, which the held-out set measures.

## 6. Voice

Live calls: median **1.84 s** from the caller stopping to bot audio (25/31 turns
≤ 2.5 s); the slow turns are step changes, where the LLM runs twice. Prefetch cut
LLM calls per booking from 18 to 15. The catalog feeds 41 key terms (site names,
surnames, rare test names) to speech-to-text; ElevenLabs caps this at 50, which I
verified against the live API before shipping.

## 7. Known issues and next steps

Live test calls surfaced issues the simulations missed. **Fixed** (40 tests, 40/40
end to end after the fixes): adults were offered pediatric providers (the solver now
filters providers by age, and booking re-checks it from the DOB); a refused MRI
suggested a Mammogram (alternatives now exclude equipment-bound tests); one fact was
asked twice; a Hindi-speaking caller got mixed English/Hindi (the agent now replies in
the caller's language, carried across context resets, while tool arguments stay in
English). **Still open:** "stomach pain / food poisoning" matched a new-patient
consultation instead of a sick visit; aliases learned from real calls would fix this
class of miss. A missing `TYPESAFE_API_KEY` is now flagged at startup and in the builder.

Next: look patients up by name + DOB instead of asking; feed
lookup misses from real calls into an alias review queue; recalibrate the
resolver on real transcripts; run the evaluation in CI on every agent edit.

**Mocked:** availability (deterministic), patient records, confirmation numbers.
**Left out on purpose:** real EHR or calendar, auth, a database, reschedule/cancel
(new steps over the same tools), a React editor.

## 8. Reproducing the numbers

From `backend/`, keys in `.env` (`TYPESAFE_API_KEY` is optional; without it,
Jev-based matchers fall back to fuzzy). Raw results are in `sim/results/`.

```bash
.venv/bin/python -m sim.eval --agents ours,naive,naive_guarded --repeat 2 --model gpt-4.1-mini   # §2 agent rows
.venv/bin/python -m sim.match_bench                                                         # matchers, dev set
.venv/bin/python -m sim.match_bench --bench holdout --only fuzzy,jev_full,hybrid            # matchers, held-out
.venv/bin/python -m sim.eval --agents ours,ours_jev,ours_cascade,ours_hybrid --repeat 2 --model gpt-4.1-mini
```

Code layout is in the [README](README.md). Branches stack: `main` →
`feat/context-management-scheduler` → `feat/hybrid-resolver` → `feat/hybrid-ambiguity` (latest).
