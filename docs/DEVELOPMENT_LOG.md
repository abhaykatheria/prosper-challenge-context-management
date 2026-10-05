# Development log

How the solution evolved, in order: what was built, what each measurement showed,
and what it changed. The design and final numbers are in [`solution.md`](../solution.md);
this file is the story behind them, including the dead ends and the bugs each
experiment caught.

## 0. Reading the starter

* The repo shipped a Pipecat voice pipeline (WebRTC → ElevenLabs STT → OpenAI →
  ElevenLabs TTS), an `AgentBuilder` that compiles JSON into Pipecat Flows, an
  example 4-node agent, and `catalog.json` (8 locations, 50 providers, 82 types).
* Profiling the catalog before writing code found traps beyond the README: 8 types
  no provider offers (eye exams, physical therapy, urology); a location with
  imaging equipment but no radiologist; confusable location names (Mission Bay /
  Mission District); pediatric providers who can be booked for general visits.
* Scope decisions agreed up front: Phase 2 engine first (it carries the most
  weight), and the builder UI stays minimal around Pipecat's prebuilt call client.

## 1. Catalog engine and tools (commit `d5732be`)

* **Engine** (`catalog/engine.py`): fuzzy matching with lay aliases, a policy solver
  that lists every valid (provider, site) pair or returns blockers with repairs,
  deterministic mock availability, opaque slot handles, booking that re-validates.
* **Enrichment** (`data/enrichment.json`): 460 lay aliases, one-line descriptions,
  18 confusable groups with a clarifying question each.
* **Tools and guards** (`catalog/tools.py`): 4 lookups plus 3 guards on graph edges.
  Patient facts are recorded once in call state and applied by every tool.
* **Builder extensions:** `node.tools`, `node.context: "reset"` (deterministic state
  summary), `edge.action` / `edge.fallback_target`.
* **Agent:** `agents/scheduler.json`, 8 steps.
* Early matcher fixes: an exact alias like "flu shot" was being called ambiguous
  because `token_set_ratio` scores any subset as 100 ("shot" ⊂ "flu shot"); blending
  in `token_sort_ratio` fixed it.

## 2. The evaluation harness, and what it caught

* **Text runner** (`sim/textflow.py`) with Pipecat Flows semantics, so the same
  compiled graph runs without audio; a small LLM plays the patient; scoring is code
  that re-validates bookings against every rule using the patient's true facts.
* **Baselines:** the whole catalog in the prompt, with and without guards.
* **Rate limits shaped the method.** The account's 30k tokens/minute on gpt-4.1
  crashed most baseline runs (each baseline turn is ~11k tokens), a finding in its
  own right. Evaluations moved to gpt-4.1-mini (200k/minute) for every agent, with
  retries, and crashed runs retried serially rather than counted as failures.
* **Bugs it caught in my design:**
  * The LLM passed a location *name* where an ID belonged → crash. Fix: return IDs
    with names, and accept exact names in the engine with a correctable error otherwise.
  * "Where can I get an MRI?" was answered with sites that have *equipment* (Mission
    District). Fix: removed the equipment filter from the tool; it now only answers
    "where is this bookable".
* **First full result:** 100% vs 57% (catalog in prompt) vs 73% (with guards); 0 vs 8
  rule violations; ~0.9k vs ~11.1k prompt tokens per turn.
* **Builder UI** (`builder_api.py`, `ui/builder.html`): graph view, step editor,
  catalog search, embedded test call, live trace. Checked in Chrome.

## 3. First live call: "the latency is high"

The user's first test call felt slow and speech-to-text stumbled. The logs from
that call were lost in a server restart, so the fixes targeted known causes and
added measurement:

* **Prefetch** (`node.prefetch`): the server runs obvious lookups as a step begins,
  removing an LLM round trip each (18 → 15 LLM calls per booking).
* gpt-4.1 → **gpt-4.1-mini** for voice; ElevenLabs **Flash** TTS; shorter, more
  natural slot offers.
* **STT key terms from the catalog.** The first version sent 100 terms; testing the
  ElevenLabs connection directly showed the API rejects more than 50, which would
  have broken every call. Shipped 41 prioritised terms.
* **Latency tap:** per-turn time from the caller stopping to the bot's first audio,
  with STT / LLM / TTS breakdown, shown in the builder and persisted to
  `logs/trace.jsonl`.
* Two regressions from prefetch, caught by the evaluation: a prefetched match on an
  existing-patients-only type carried no repair, so the LLM improvised (fix: every
  tool returns blockers with repairs), and the simulated caller didn't know today's
  date (harness fix), which exposed slot offers that omitted the day of the month (prompt fix).

## 4. Jev: "throw in all the context" (commit `cc018f5`)

* Researched TypeSafe's Jev: no text generation; returns one of a fixed set of
  options with probabilities; ~0.3–0.5 s; $0.042 per 1M input tokens. Sending the
  whole catalog per lookup costs ~$0.0009.
* Built `catalog/jev.py` as a drop-in for the "which service?" step, and a 90-phrase
  **matching benchmark** with groups for direct, indirect, garbled, ambiguous,
  not-offered and vague requests.
* **Findings:** Jev understands descriptions far better (fuzzy 64 correct / 15 wrong
  vs Jev 76 / 1). Full-catalog context beat options-only. But raw Jev **commits on
  ambiguous requests** ("I need an MRI" → MRI - Brain). A second yes/no question in
  the same request ("would a receptionist need to clarify?") plus a threshold sweep
  fixed that (0.4 chosen: no committed guesses).
* **End to end the gap nearly vanished** (fuzzy 40/40, Jev 39/40), because the
  conversational LLM normalises wording before any lookup. Building the experiment
  exposed bugs shared by both matchers, now fixed: the patient's **age** was never
  used (a 4-year-old with a fever got the adult sick visit), and the confirm step
  could say "you're booked" without calling the booking function.
* **Cascade** resolver (fuzzy if confident, else Jev) became the default at this point.

## 5. Does Jev scale? Measuring its limit

* TypeSafe's docs don't state a context limit (Cloudflare's listing says 32k), so I
  measured it: 32,973 input tokens accepted, more returns `400 max_tokens_exceeded`.
  The catalog already used 21.8k (~66%); ~1.5× this catalog wouldn't fit.
* **Region sharding** was considered and measured: all 8 locations are in one city,
  34/50 providers work at several sites, and the location is usually unknown when
  the service is resolved. A good outer filter for a multi-city system, not a fix.

## 6. Hybrid resolver (commit `e9fe8a1`)

* **Key measurement:** fuzzy's weakness is ranking, not recall: the right type is in
  its top 5 for 80/82 phrasings. So fuzzy retrieves a shortlist and Jev picks within
  it, with ~3k tokens of context at any catalog size, plus a size guard and a fuzzy
  fallback on any Jev error.
* A regression caught before commit: moving the age filter into the shared ranking
  step made it consider *every* type, so "urgent care" for a child returned nothing.
  Fixed by limiting it to relevant candidates; test added.
* Jev turned out to be **non-deterministic run to run**, so results from here on
  are 3-run means. Hybrid: ~80/90 first-try correct vs 69 for full-catalog Jev, ~7×
  cheaper, 40/40 end to end; but it still committed on ~2/8 ambiguous requests.

## 7. Hybrid v2: fixing ambiguity, proved on held-out data (commits `f8aefa0`, `da9534b`)

* **Held-out set first:** 60 phrasings generated by gpt-4.1 from the catalog's
  descriptions, a pre-registered audit relabelling 5 underspecified items, committed
  before any tuning. Baseline on it: fuzzy 23/60 with 24 wrong, showing **my matcher
  was overfit to my own vocabulary**.
* **Failure analysis** of three v1 runs: the global yes/no score was noise with only
  a shortlist in view (0.22–0.33 for both clear and ambiguous requests); "medications"
  never matched "medication"; category requests ("some kind of scan") dead-ended.
* **Fixes:** one "which of these, or not specified?" question per confusable group in
  the same request; commit only when the caller used a word that singles the option
  out (derived from the enrichment data) or Jev is very sure; plural folding; category
  requests become that group's question (chosen by the caller's exact words, after a
  first version keyed on fuzzy's top guess picked the wrong group).
* **Results:** dev 87/90, 0 committed guesses, 8/8 ambiguous (identical across 3
  runs); held-out 50/60 (v1 45, full-catalog Jev 36.7, fuzzy 22); end to end 40/40.

## 8. Second round of live calls

Three real calls (one in Hindi) on the final build, read from `logs/trace.jsonl`:

* Median **1.84 s** per turn; slow turns (3.5–6 s) are step changes where the LLM
  runs twice. Both completed calls booked validly; ambiguity questions and repairs
  worked live. $0.019 for all three calls.
* **New problems the simulations missed:** adults offered and booked with a pediatric
  provider; inconsistent language handling; acute symptoms resolved to a new-patient
  consultation instead of a sick visit; a weak repair (Mammogram) for a refused MRI;
  one fact asked twice.

## 9. Fixing what the live calls found (branch `feat/demo-fixes`)

* **Age-aware providers:** the solver drops pediatric providers for adults (except for
  pediatric types); `book()` re-derives the age from the DOB, so even a pediatric slot
  offered earlier can't be booked for an adult.
* **Repairs:** alternatives for an existing-patients-only type are the specialty's
  entry visits, never equipment-bound tests; general new-patient visits only when the
  specialty has none. A first version always appended the general visits, and the
  regression run caught it (a new patient asking for therapy got a general consult
  instead of a psychiatric evaluation).
* **Language:** reply in the caller's language; tool arguments stay in English. The
  first version lost the language at the first context reset; it's now a recorded fact
  in the step summary.
* **No re-asking** of facts already in the summary; a missing Jev key is flagged at
  startup and in the builder, and falls back without a network call.
* Result: 40 tests; 40/40 end to end, 0 rule violations. Still open: aliases for acute
  symptoms ("food poisoning").

## Supporting material

* Architecture walkthrough with diagrams (written for newcomers), published as a
  claude.ai artifact during the session. It predates sections 4–8.
* Every evaluation run's raw transcripts and summaries: `backend/sim/results/`.
