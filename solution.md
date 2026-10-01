# Solution — Voice Agent Builder + Catalog Context Management

**TL;DR.** The scheduling agent never sees the catalog as text. A deterministic
**catalog engine** (fuzzy retrieval + a policy solver) sits behind a handful of
narrow tools; the LLM asks small questions and gets back small, pre-validated
answers. Booking policies are enforced **in code at graph edges** (guards), so an
invalid booking is unrepresentable rather than merely discouraged by a prompt,
and every refusal comes back with concrete repair options so the model
self-corrects in one turn. Each conversation stage sees only its own tools and a
compact state summary instead of the growing transcript.

Measured with a simulated-caller eval (15 adversarial scenarios × 2 runs,
deterministic policy-checked scoring, same model for every agent):

| | **This approach** | Full catalog in prompt | Full catalog + guards |
|---|---|---|---|
| Task success | **100%** (30/30) | 57% (17/30) | 73% (22/30) |
| Policy-violating bookings | **0** | 8 | 0 |
| Prompt tokens per LLM call | **~0.9k** | ~11.1k | ~11.1k |
| Cost / conversation (no caching) | **$0.0064** | $0.0366 | $0.0414 |
| Cost / conversation (with prompt caching) | **$0.0058** | $0.0106 | $0.0120 |

```
make install && make run      # builder + voice test call at http://localhost:7860/builder
make eval                     # simulated-caller eval: ours vs full-catalog baselines
cd backend && .venv/bin/python -m pytest -q    # deterministic engine tests
```

---

## 1. The problem, restated

Dumping 8 locations × 50 providers × 82 appointment types (~12k tokens) into the
prompt fails in two ways:

* **Cost/latency** — the whole catalog is re-sent on *every* LLM turn of a call
  (a booking call is 15–20 LLM turns). Cost scales with catalog size × turns, and
  time-to-first-token grows with prompt size, which matters on a voice call.
* **Accuracy** — the model must do a multi-way join in its head: *type ×
  provider.appointment_type_ids × provider.location_ids × location.capabilities ×
  new-patient rules × referral rules*. LLMs are bad at exhaustive constraint
  satisfaction over long lists, and the catalog is deliberately messy.

Things I found in the data that a naive approach gets wrong:

| Trap | Example |
|---|---|
| Duplicate names | 2× Dr. Maria Garcia (Peds MD / Family NP), 2× Linda Ramirez, 2× Michael Sato (one not accepting new patients), 7 Chens |
| Near-duplicate types | Annual Physical vs Annual Wellness Visit; Follow-up Visit / Consultation / Telehealth; MRI Brain/Spine/Knee |
| **Types with zero providers** | All 5 eye exams, both Physical Therapy types, Urology — the agent must say "not offered", not hallucinate a booking |
| **Equipment ≠ bookable** | Mission District has `imaging`, but no radiologist practises there, so MRIs are only bookable at Downtown & Midtown |
| Confusable locations (voice!) | Mission **Bay** vs Mission **District**; North **Beach** vs North **Gate** |
| Service bookable at one site only | Blood draw: one lab provider, whose only lab-capable site is Mission Bay |

## 2. Architecture

```
 caller speech ─► STT ─► LLM ◄──────────── node-scoped tools ─────────────┐
                          │                                               │
                          │ edge call (e.g. select_service)               │
                          ▼                                               │
                  ┌───────────────┐  refuse + alternatives   ┌────────────┴───────────┐
                  │  edge GUARD   │ ───────────────────────► │     CatalogEngine      │
                  │ (policy check)│ ◄─────────────────────── │ match · solve · slots  │
                  └──────┬────────┘                          │ book (re-validates)    │
                         │ ok: state.update(ids)             └────────────────────────┘
                         ▼                                       in-memory indexes,
                  next node (context: reset → state summary)     never in the prompt
```

### 2.1 Catalog engine (`backend/catalog/engine.py`) — no LLM involved

* **Matching.** Token-order-insensitive fuzzy matching (rapidfuzz) over type
  names **plus offline-authored lay aliases** ("checkup", "heart doctor", "cat
  scan", "my kid is sick"). Plain edit distance absorbs STT errors ("halter
  monitor" → Holter). Returns `match | ambiguous | not_offered | no_match`.
* **Disambiguation is data, not prompt engineering.** `data/enrichment.json`
  defines *confusable groups* with a crisp clarifying question each. When a
  query lands in a group, the tool returns the siblings + `suggested_question`
  + `differs_by` (which attributes actually distinguish them). The LLM asks one
  sharp question instead of reading out a list.
* **Policy solver.** `solve(type, provider?, location?, new_patient, referral,
  language)` enumerates exactly the valid (provider, location) pairs. When the
  set is empty it returns typed **blockers with repairs**:
  `not_accepting_new_patients → [other Chens in cardiology who are]`,
  `location_ineligible → [sites that can]`, `existing_patients_only → [the
  new-patient type in the same specialty]`, `not_offered → [related offered types]`.
* **Mock availability**, deterministic (hash-seeded) so evals are reproducible.
  A provider works at one site per day, so a multi-site provider can never be
  double-booked across locations; bookings made in one call disappear from the
  next call's availability.
* **Opaque slot handles.** `search_slots` returns `slot_id`s that encode
  type|provider|location|time. The LLM books by handle, so it can't mis-transcribe
  or mix ids across options.

### 2.2 Tools vs. guards (`backend/catalog/tools.py`)

* **Tools** (stay on node): `find_appointment_types`, `find_providers`,
  `find_locations`, `search_slots`. Results are small (typically 100–600 tokens)
  and shaped for speech.
* **Guards** (run on an edge *before* the transition): `select_service`,
  `hold_slot`, `book_appointment`. If a guard fails the conversation doesn't
  advance; the LLM receives the blocker + alternatives and repairs. `book`
  re-validates everything with the patient facts on record.
* **Facts are recorded once, applied everywhere.** New-vs-existing, referral and
  language live in call state; tools read them implicitly. The model can't
  "forget" to filter for new patients on the third search.

### 2.3 Stage-scoped context (`backend/agent_builder/builder.py`)

The agent is a Pipecat Flows graph (`backend/agents/scheduler.json`):
`greeting → identify_patient → find_service → offer_slots → confirm_booking →
booked`, plus `answer_questions` for Q&A. Two schema additions, both plain data
and editable in the UI:

* `node.tools` — only the tools that stage needs (smaller tool schemas, less
  opportunity for the wrong call).
* `node.context: "reset"` — on entry, drop the transcript and inject a
  **deterministic** summary of structured state (name, new patient, resolved
  type/provider/location ids, stated preferences). Unlike Pipecat's
  `RESET_WITH_SUMMARY` this costs no extra LLM call and can't drop a fact.
  Prompt size per turn stays roughly constant regardless of call length.
* `edge.action` / `edge.fallback_target` — the guard, and where to go if it
  fails (e.g. slot taken at confirm → back to `offer_slots`).

### 2.4 Voice-specific: latency and STT

* **Server-side prefetch** (`node.prefetch`). On entering `offer_slots` the
  server runs `search_slots` itself and hands the result to the LLM, instead of
  the LLM spending one inference deciding to call it and a second speaking the
  result. Same for `find_appointment_types` on the request the caller already
  stated. One inference ≈ 0.5–1s of dead air saved per lookup.
* **STT biasing from the catalog** (`catalog/vocab.py`). The catalog also tells
  the recogniser what rare words to expect: confusable location names, provider
  surnames, test names (Holter, DEXA, spirometry). ElevenLabs caps keyterms at 50
  (verified against the API), so terms are prioritised by confusion risk.
* **Fast models where it's safe**: `gpt-4.1-mini` (100% on the eval with this
  architecture, because the hard reasoning lives in code, not in the model) and
  `eleven_flash_v2_5` TTS.
* **Measured, not guessed**: a pipeline tap records per-turn voice-to-voice latency
  (caller stops → bot audio) with STT/LLM/TTS TTFB breakdown, shown live in the
  builder and persisted to `backend/logs/trace.jsonl`.

## 3. Evaluation (`backend/sim/`)

A text-mode runner (`sim/textflow.py`) executes the **same** compiled graph with
Pipecat Flows semantics (APPEND/RESET, handlers returning `(result, next_node)`),
recording token usage per LLM call. A small LLM plays the caller from a persona
with private facts. Scoring is **deterministic, not an LLM judge**: the final
booking must match the expected type/provider/location constraints *and* is
re-validated against every catalog policy using the patient's **true** facts.

Three agents, same model, same caller simulator:

* **ours** — retrieval tools + guards + stage-scoped context.
* **naive** — full `catalog.json` in the system prompt, tools to check a specific
  (type, provider, location) and book it. The status quo.
* **naive_guarded** — naive + server-side policy validation on booking. This
  ablation separates *"guards alone"* from *"retrieval + scoped context"*.

### Results (gpt-4.1-mini agent, gpt-4o-mini caller, 15 scenarios × 2 repeats)

| agent | pass rate | policy violations | avg prompt tok / call | max prompt tok | tok / conversation | cost / conv (cached) | cost / conv (uncached) | LLM calls / conv |
|---|---|---|---|---|---|---|---|---|
| **ours** | **100% (30/30)** | **0** | **906** | 2,693 | **14.8k** | **$0.0058** | **$0.0064** | 15.9 |
| naive | 57% (17/30) | 8 | 11,122 | 14,738 | 90.5k | $0.0106 | $0.0366 | 8.1 |
| naive_guarded | 73% (22/30) | 0 | 11,117 | 15,170 | 102.3k | $0.0120 | $0.0414 | 9.2 |

Rate-limit crashes are retried serially and never counted as failures. Raw
transcripts for every run are in `backend/sim/results/` (`baselines_final.json`,
`ours_final.json`).

**How the baselines fail** — exactly on the traps listed in §1:

* Booked *Dr. David Chen* (not accepting new patients) for a new patient, twice.
* Booked *Therapy Session* / *Echocardiogram* for new patients (existing-only types), 4×.
* Booked mole removal at Downtown, which has no surgery capability, twice.
* Booked an eye exam (no provider offers it) once.
* Told the caller MRIs are available at Mission District (equipment ≠ bookable), 4/4.
* With guards, the illegal bookings disappear (0 violations) but the *matching*
  errors remain: wrong Dr. Michael Sato, a non-Spanish-speaking provider, a
  pediatric visit for an adult, wrong Mandarin-speaking dentist. **Guards fix
  safety; retrieval + structured disambiguation fix accuracy** — you need both.

**Reading the cost numbers honestly.** Prompt caching discounts the baseline's
huge, repeated prefix by 75%, so with caching the cost gap is ~2× rather than
~6×. But caching only applies to an identical prefix within minutes, it doesn't
help accuracy, and the token volume still lands on rate limits: on gpt-4.1 at
30k TPM (this org's tier), the baseline can serve ~3 LLM turns per minute
across *all* concurrent calls; ours ~30. Our agent makes ~2× more LLM calls
(tool round-trips, node transitions); prefetch (§2.4) already removed ~15% of them.

**Bugs the eval caught in my own design** (the reason to build it):
1. The LLM answered "where can I get an MRI?" with `find_locations(service=imaging)`
   → said Mission District. Fixed by removing the equipment filter from the tool
   entirely and replacing it with "where is appointment type X bookable", so the
   wrong question can't be asked.
2. After adding prefetch, a new patient asking for "therapy" got a plain match on an
   existing-only type with no repair, and the LLM improvised a primary-care visit.
   Fixed by making *every* tool, not just guards, return blockers with their repairs.

## 3b. Experiment: "just give a model the whole catalog" with Jev

The obvious counter-argument to this design is: *a better model with the whole
catalog in context would be simpler.* For an LLM that's too slow and expensive
per turn (§3). [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
(TypeSafe AI) changes the economics: it doesn't generate text, it answers typed
questions (pick one of N options, with a probability per option), takes up to
~32k input tokens (measured: 32,973 accepted, more returns `400
max_tokens_exceeded`), answers in ~0.3–0.5s, and costs $0.042 per 1M input tokens.
So sending the **entire catalog on every lookup costs ~$0.0009**, and its
output can't name an appointment type that doesn't exist.

Jev can't hold the conversation, so I tried it where it fits: the
*"which appointment type does the caller mean?"* step (`catalog/jev.py`), as a
drop-in for the fuzzy matcher. The solver, repairs and guards are unchanged.

**Matching benchmark** (`sim/match_bench.py`, 90 labelled caller phrasings, most
deliberately *not* taken from my alias list: 20 direct, 44 indirect
descriptions, 10 STT garbles, 8 genuinely ambiguous, 5 not offered, 3 vague):

| resolver | correct | asked a clarifying Q (safe) | committed on an ambiguous request | wrong | ambiguous handled | p50 latency | Jev input tokens | $/lookup |
|---|---|---|---|---|---|---|---|---|
| fuzzy + aliases (+ plural stemming) | 66 | 10 | 3 | **11** | 4/8 | 4 ms | 0 | 0 |
| Jev, option list only | 68 | 16 | 4 | 2 | 2/8 | 285 ms | 6,930 | $0.0003 |
| Jev, full catalog † | 69 (68–70) | 20 | **0** | **1** | **7/8** | 405 ms | 21,805 | $0.0009 |
| cascade (fuzzy if confident, else full-catalog Jev) | 69 | 16 | 3 | 2 | 4/8 | 389 ms | 14,780 | $0.0006 |
| hybrid v1 (shortlist + global yes/no) † | 79.7 (79–80) | 6 | 2 (1–3) | 2 | 4–6/8 | 260 ms | 3,041 | $0.00013 |
| **hybrid v2 (per-group questions + word check)** † | **87 (87–87)** | **2** | **0** | **1** | **8/8** | **260 ms** | 3,838 | $0.00016 |

† Mean [range] over 3 runs: Jev's answers vary slightly from run to run, so single
runs differ by a few points. Other rows are single runs (fuzzy is deterministic).

What I learned:

1. **Jev understands people far better than lexical matching.** Fuzzy matching
   fails on descriptions ("I keep forgetting things" → no match; "my daughter has
   a fever" → the *adult* sick visit). Jev gets 43/44 of those on its top pick.
2. **Full context helps** (all STT garbles fixed; better on edge cases).
3. **Raw Jev is overconfident on ambiguity**: "I need an MRI" → *MRI - Brain*.
   Its choice probabilities are peaky. Fix: a second question in the **same
   call** (answered in parallel, no added latency), a yes/no *"does the request
   pin down exactly one type, or would a receptionist need to clarify?"*. A
   threshold sweep traded correct-on-first-try for safety; I chose 0.4 (zero
   overconfident commits, 7/8 ambiguous requests clarified) because an extra
   question costs ~3s of call time while a wrong MRI costs a wasted visit. That
   threshold was tuned on this set; production would tune it on held-out call
   transcripts.

**End to end** (20 scenarios × 2, including 5 new ones where the simulated
caller only describes symptoms):

| agent resolver | pass | violations | cost / conversation |
|---|---|---|---|
| fuzzy | 100% (40/40) | 0 | $0.0063 |
| Jev, full catalog | 98% (39/40) | 0 | $0.0075 |
| cascade | 98% (39/40) | 0 | $0.0065 (0.55 Jev calls/conv, ~500 ms each) |
| **hybrid** | **100% (40/40)** | **0** | $0.0066 (1.07 Jev calls/conv, ~330 ms, $0.00016 of it Jev) |

The isolated gap almost vanishes end to end, and the reason matters for the
design: **the conversational LLM already normalises language before the
lookup** ("my chest hurts on stairs" arrives as "cardiology consultation"), and
patient context does the rest. Building the experiment also exposed two gaps
that fuzzy *and* Jev shared and that are now fixed for both: neither used the
**patient's age** from the DOB already collected (a 4-year-old with a fever got
the adult sick visit), and the confirm step could say "you're booked" without
calling `book_appointment`.

**Why full-catalog context doesn't scale, and what does.** Our catalog already
uses 21.8k of Jev's ~32k window; about 1.5× this catalog would not fit. I looked
at sharding the context by region: all 8 sample locations are in one city (so a
city split is a no-op), 34/50 providers work at several locations (they repeat
across location shards), and the location is usually unknown when the service is
resolved, especially for "soonest anywhere". Region is still a sensible *outer*
filter for a multi-city system, but it doesn't solve the problem inside one region.

The measurement that settled it: **fuzzy matching's weakness is ranking, not
recall.** The right type is in fuzzy's top 5 for 80/82 benchmark phrasings (top
15: 81/82). So the **hybrid** resolver lets each component do what it's good at:

1. fuzzy ranks every type locally (~3 ms) and keeps a shortlist: top 12 plus their
   confusable siblings, so "which MRI?" can still be asked;
2. an exact match on a type with no confusable siblings skips Jev entirely;
3. otherwise Jev picks within the shortlist, with only those options and the
   patient facts in context (~3.8k tokens instead of 21.8k, the same at any
   catalog size);
4. **ambiguity is decided per confusable group, in the same call**: for every
   group touching the shortlist Jev also answers "which of these, or *not
   specified*?". Committing inside a group requires the group answer to agree
   with the main pick **and** either a word in the caller's request that singles
   it out (derived from the enrichment data: Fasting Blood Test ← fasting,
   cholesterol, lipid; Telehealth ← video, virtual; MRI-Knee ← knee) or a very sure
   group answer (p ≥ 0.9). Otherwise the agent asks the group's curated question;
5. if Jev answers "none of these": a request that names a category ("some kind of
   scan") gets that group's question; anything else gets one escape-hatch call
   over every option ("I keep forgetting things");
6. a size check refuses any request over 30k tokens before sending it, and any Jev
   error, timeout or oversize degrades to the fuzzy result (tested).

**How v2 came about.** v1 used one global yes/no ("is the request specific?")
next to the choice. Failure analysis of three v1 runs showed that, with only a
shortlist in view, that score is noise: clear and ambiguous requests both scored
0.22–0.33, so it caused both errors at once (asking about "a video appointment"
when Jev was certain it meant Telehealth, and committing on "a blood test").
Two smaller fixes came from the same analysis: plural folding in the fuzzy
normaliser ("medications" never matched "medication", so Medication Review
missed the shortlist), and category requests no longer dead-ending in "no match".

**Held-out check.** The dev set above was used for tuning, so before changing
anything I generated a second set of 60 phrasings with a different model
(gpt-4.1, from the catalog's own descriptions; labels are the generation target,
with a pre-registered audit of 5 underspecified items) and committed it frozen
(`sim/match_bench_holdout.json`, commit f8aefa0). Results, 3 runs each, no
tuning after the first reading:

| held-out (60) | correct | asked (safe) | committed on ambiguous | wrong | ambiguous handled | p50 |
|---|---|---|---|---|---|---|
| fuzzy | 22 | 8 | 6 | **24** | 2/15 | 7 ms |
| Jev, full catalog | 36.7 | 19 | 0 | 4.3 | 11–12/15 | 400 ms |
| hybrid v1 | 45 | 8 | 4 | 3 | 8/15 | 259 ms |
| **hybrid v2** | **50** | **5** | **1** | 4 | **12/15** | 265 ms |

Two lessons. First, **my fuzzy matcher was overfit to my own vocabulary**: 66/90
on phrasings I wrote, 22/60 on someone else's; the aliases are only as good as
the person who wrote them, which is exactly the gap a model like Jev fills.
Second, the v2 gains hold on unseen data. The remaining misses are hard STT
garbles ("general ex-am with a Dennis" for a dentist, "neural allergy" for
neurology), one wrong clarifying question, and two debatable labels ("some blood
work" read as a standard blood draw).

**End to end** with hybrid v2: **100% (40/40), 0 policy violations**, 0.93 Jev
calls per conversation, ~320 ms each, $0.00015 of Jev cost per call.

**Decision:** default to **hybrid v2**. On held-out data it gets the most right
on the first try (50/60 vs 36.7 for full-catalog Jev and 22 for fuzzy), asks the
fewest needless questions, almost never guesses on an ambiguous request, is ~35%
faster and ~6× cheaper than full-catalog Jev, and its Jev context doesn't grow
with the catalog. No matcher is perfect, so correctness doesn't rest on it alone:
the agent reads the appointment back before booking, and the policy guards
re-validate every booking. The resolver remains a per-agent setting in the
builder UI (fuzzy, jev, cascade, hybrid). Past Jev's 255-option limit, step 5's
escape call becomes Jev's documented two-stage choice (specialty, then type).

## 4. Trade-offs and alternatives considered

| Option | Why not (as the primary mechanism) |
|---|---|
| **Full catalog in prompt** (+ prompt caching) | Caching cuts price, not accuracy; the model still has to do the multi-way join in its head. It also consumes rate limit: at 30k TPM on gpt-4.1 the naive agent fits only ~3 LLM turns per minute *across all calls*. |
| **Embedding RAG over catalog chunks** | Solves recall, not constraints. Top-k chunks of providers don't tell you which (provider, site) pairs can do an MRI for a new patient. Retrieval of *entities* is the easy half; the join is the hard half, and the join belongs in code. I'd add embeddings as a **recall layer inside `match_appointment_types`** once the catalog has thousands of types or free-text descriptions; the tool contract wouldn't change. |
| **LLM-generated summary on node transitions** | Extra latency + cost per transition, and it can silently drop facts. Structured state is exact and free. |
| **One big node with all tools** | Works, but every turn carries every tool schema and the model can skip steps (e.g. book before confirming referral). The graph enforces order where order matters. |
| **Fine-tuning on the catalog** | Catalog changes daily (providers join, stop accepting patients); knowledge must be data, not weights. |

**Scalability.** Nothing in the prompt grows with the catalog. Tool results are
capped (≤4 types, ≤5 providers, ≤3 slots). The engine's in-memory indexes are
O(catalog) and the solver is O(providers-for-type × their-sites). At 10k+
providers the same interface moves onto SQL/Postgres (indexes on type,
location, capability) plus an embedding index for recall. Enrichment (aliases,
confusable groups) is generated **offline** per catalog version by an LLM and
reviewed by clinic staff; it's never generated at call time.

**Latency.** Tools are local, sub-millisecond Python; each stage's prompt is
~0.5–1.2k tokens vs ~10k, which directly lowers time-to-first-token on every turn.

## 5. Phase 1 — builder UI

`http://localhost:7860/builder` (vanilla JS page served by the same FastAPI app
as the bot, per the "keep the prebuilt client" scope decision):

* Agent list / create / duplicate / delete; agents are JSON files in `backend/agents/`.
* Live **Mermaid graph** of the node graph; click a node to edit its task
  prompt, context mode, tools (chips) and edges (target, guard, fallback,
  collected fields). Raw JSON tab for power edits. Saving validates with
  `AgentBuilder` (unknown targets/tools/actions are rejected).
* **Test call**: activates the agent and embeds Pipecat's prebuilt WebRTC client.
  Next to it, a **live trace** streams node transitions, tool calls with results,
  guard allow/refuse decisions, transcripts, and per-turn prompt tokens / cost.
* **Catalog search** tab: type what a caller might say and see exactly what the
  tools return (and how many tokens that is).

## 6. Scoping — what I built, mocked, or left out

* **Built:** Jev resolver experiment (benchmark, end-to-end eval, cascade,
  hybrid fuzzy-shortlist → Jev resolver as the default),
  catalog engine + tests, enrichment layer, tool/guard layer, graph
  schema extensions (tools, guards, context reset, prefetch), scheduling agent,
  text runner, simulated-caller eval with ablation, builder UI with live trace
  and latency metrics, catalog-driven STT biasing.
* **Mocked:** availability (deterministic hash-seeded), patient records (new vs
  existing and referral are self-reported), confirmation numbers.
* **Left out on purpose:** real EHR/calendar integration, auth, a database
  (JSON files are enough for one editor), reschedule/cancel flows (same tools,
  new nodes — the graph makes that additive), a React/React Flow editor (the
  Mermaid view + form editor covers the requirement for a fraction of the work),
  embedding retrieval (unnecessary at 82 types; the interface is ready for it).

## 7. What I'd do next

1. Patient lookup by name + DOB against a patient store (derive new/existing and
   referrals instead of asking).
2. Embedding recall layer + learned aliases from real call transcripts
   (misses in `find_appointment_types` → review queue → enrichment).
3. Voice-specific: spell-back confirmation for confusable locations; NATO-style
   confirmation of names.
4. Run the eval in CI on every agent edit from the builder ("this change dropped
   pass rate from 93% → 80%").
