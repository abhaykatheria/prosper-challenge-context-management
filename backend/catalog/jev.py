#
# Jev resolver — catalog matching with TypeSafe AI's Jev (a "System One" model).
#
# Jev doesn't generate text: given `state` (context) and Choice questions with a
# fixed option set, it returns a probability for every option. That makes it a
# natural replacement for the fuzzy matcher's "which of the 82 appointment types
# does this caller mean?" step:
#
#   * it can't invent an option (output is constrained to the schema), and
#   * it's cheap/fast enough to send the WHOLE catalog as context on every lookup
#     (~$0.042 per 1M input tokens, 70-500ms), which an LLM can't afford per turn.
#
# Two context modes, so the benchmark can measure what extra context buys:
#   "options" — only the option list (names + descriptions + lay examples)
#   "full"    — options + the entire catalog (types, providers, locations, policies)
#
# The policy solver, repairs and guards are unchanged: Jev only resolves *what the
# caller means*; whether it's bookable is still decided in code.
#

import asyncio
import json
import os
import time
from typing import Optional

import httpx

from .engine import CatalogEngine
from .models import CATALOG_PATH

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
PRICE_PER_M_INPUT = 0.042

NONE_OPTION = "none of these"
NOT_SPECIFIED = "not specified"
# Measured: 32,973 input tokens accepted, larger -> HTTP 400 max_tokens_exceeded.
MAX_INPUT_TOKENS = 30_000
CHARS_PER_TOKEN = 3.0  # calibrated on catalog.json: 45.6KB of JSON ~ 15.2k Jev tokens


class ContextTooLarge(ValueError):
    """Request would exceed Jev's input window; callers fall back to the local matcher."""


class JevClient:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0, cache: Optional[dict] = None):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self._client = httpx.AsyncClient(timeout=timeout)
        self.calls: list = []  # [{"latency": s, "input_tokens": n}]
        self.cache = cache  # optional {request_json: response}, used by the offline benchmark sweeps

    async def system_one(self, state, questions: dict) -> dict:
        body = {"model": MODEL, "state": state, "questions": questions}
        est = len(json.dumps(body)) / CHARS_PER_TOKEN
        if est > MAX_INPUT_TOKENS:  # fail fast instead of a guaranteed 400 round trip
            raise ContextTooLarge(f"~{est:.0f} input tokens exceeds Jev's {MAX_INPUT_TOKENS} budget")
        key = json.dumps(body, sort_keys=True) if self.cache is not None else None
        if key is not None and key in self.cache:
            self.calls.append({"latency": 0.0, "input_tokens": self.cache[key].get("usage", {}).get("input_tokens", 0)})
            return self.cache[key]
        for attempt in range(5):
            t0 = time.monotonic()
            r = await self._client.post(API_URL, json=body,
                                        headers={"Authorization": f"Bearer {self.api_key}"})
            if r.status_code in (429, 529):
                await asyncio.sleep(0.5 * 2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            self.calls.append({"latency": time.monotonic() - t0,
                               "input_tokens": data.get("usage", {}).get("input_tokens", 0)})
            if key is not None:
                self.cache[key] = data
            return data
        raise RuntimeError("Jev unavailable (rate limited / overloaded)")


_catalog_context: Optional[dict] = None


def catalog_context() -> dict:
    """The full catalog, compacted (no whitespace, short keys kept as-is)."""
    global _catalog_context
    if _catalog_context is None:
        _catalog_context = json.loads(CATALOG_PATH.read_text())
    return _catalog_context


class JevResolver:
    # A candidate needs this much probability mass to be offered to the caller.
    MIN_P = 0.12
    # Top option is "clear" above this, with no real runner-up.
    CLEAR_P = 0.75
    # Shortlist mode: a confusable-group answer this confident overrides the
    # missing-discriminator backstop (the caller described it unambiguously in other words).
    GROUP_SURE_P = 0.9

    def __init__(self, engine: CatalogEngine, client: Optional[JevClient] = None, context: str = "full",
                 specific_threshold: float = 0.4):
        self.e = engine
        self.specific_threshold = specific_threshold
        self.client = client or JevClient()
        self.context = context
        types = list(engine.c.appointment_types.values())
        self._type_by_option = {t.name: t for t in types}
        self._type_criteria = {
            t.name: {"specialty": t.specialty, "what": t.description or t.name,
                     "examples": t.aliases[:6]}
            for t in types
        }
        self._type_criteria[NONE_OPTION] = "The request doesn't match any appointment type above, or is too vague to tell."

        provs = list(engine.c.providers.values())
        self._prov_by_option = {f"{p.display} [{p.id}]": p for p in provs}
        self._prov_criteria = {
            f"{p.display} [{p.id}]": {
                "sites": [engine.c.locations[l].name for l in p.location_ids],
                "accepting_new_patients": p.accepting_new_patients,
                "languages": p.languages,
            }
            for p in provs
        }
        self._prov_criteria[NONE_OPTION] = "No provider in the list matches who the caller described."

    def _state(self, caller_said: str, extra: dict, with_catalog: bool = True) -> dict:
        state = {"caller_said": caller_said, **{k: v for k, v in extra.items() if v is not None}}
        if self.context == "full" and with_catalog:
            state["clinic_catalog"] = catalog_context()
        return state

    # ---- appointment types -------------------------------------------------
    async def match_appointment_types(self, query: str, new_patient: Optional[bool] = None,
                                      patient_age: Optional[int] = None,
                                      shortlist: Optional[list] = None) -> dict:
        """Resolve the caller's words to appointment types.

        shortlist=None: choose among ALL types (+ full catalog as context in "full" mode).
        shortlist=[...]: choose only among these (the hybrid resolver's retrieved
        candidates); no catalog in context, so the request stays ~1-2k tokens at any
        catalog size.
        """
        if shortlist:
            return await self._match_shortlist(query, new_patient, patient_age, shortlist)
        criteria = self._type_criteria
        state = self._state(query, {"new_patient": new_patient, "patient_age_years": patient_age})
        data = await self.client.system_one(
            state,
            {
                "appointment_type": {
                    "type": "choice",
                    "instructions": "Which of this clinic's appointment types is the caller asking for? "
                                    "Spread probability across options if the request could reasonably mean several.",
                    "criteria": criteria,
                },
                # Jev's choice probabilities are peaky: on a genuinely ambiguous request it
                # still commits to one option. A second question in the same call (answered
                # in parallel, no extra latency) asks directly whether a receptionist would
                # need to clarify.
                "is_specific": {
                    "type": "noul",
                    "instructions": "Does the caller's request pin down exactly one appointment type? Answer no "
                                    "if several similar appointment types would fit what they said and a careful "
                                    "receptionist would need to ask a clarifying question before booking.",
                },
            },
        )
        specific = data["answers"]["is_specific"]["noul"]
        probs = data["answers"]["appointment_type"]["probabilities"]
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        if ranked[0][0] == NONE_OPTION and ranked[0][1] >= 0.5:
            return {"status": "no_match", "query": query, "candidates": [],
                    "hint": "Ask the caller to describe the reason for the visit in other words."}
        cands = [(p, self._type_by_option[name]) for name, p in ranked
                 if name != NONE_OPTION and p >= self.MIN_P][:4]
        if not cands:
            return {"status": "no_match", "query": query, "candidates": []}

        types = [t for _, t in cands]
        clear = (cands[0][0] >= self.CLEAR_P and specific >= self.specific_threshold
                 and (len(cands) == 1 or cands[1][0] < self.MIN_P + 0.03))
        if not clear and len(types) == 1:
            # Not specific, but the choice question gave only one candidate: widen with
            # the low-probability runners-up so the confusable group can be found.
            types = [self._type_by_option[n] for n, p in ranked if n != NONE_OPTION and p >= 0.01][:4]
        question = None
        if not clear:
            group = self.e.confusable_group(types[0].id, {t.id for t in types})
            if group is None and len(types) == 1:
                # Jev says "not specific" but put all its mass on one option: use the
                # curated sibling group of that option to form the clarifying question.
                groups = sorted(self.e.c.group_by_type.get(types[0].id, []), key=lambda g: len(g["ids"]))
                group = groups[0] if groups else None
            if group:
                question = group["question"]
                types = [t for t in types if t.id in group["ids"]] + [
                    self.e.c.appointment_types[i] for i in group["ids"] if i not in {t.id for t in types}]
        out = self.e.finalize_type_match(types, clear or len(types) == 1, question, new_patient, patient_age)
        out["confidence"] = round(cands[0][0], 3)
        out["specific"] = round(specific, 3)
        return out

    async def _match_shortlist(self, query: str, new_patient: Optional[bool],
                               patient_age: Optional[int], shortlist: list) -> dict:
        """Hybrid mode: choose within fuzzy's shortlist, and decide ambiguity per confusable group.

        A single global "is this specific?" yes/no proved to be noise with only a shortlist
        in view (clear and ambiguous requests both scored 0.22-0.33). Ambiguity in this
        catalog is defined by the curated confusable groups, so for every group touching
        the shortlist the same call also asks "which of these, or not specified?". The
        answer is then cross-checked against the caller's actual words: committing inside a
        group requires either a term that singles the option out, or a very sure answer.
        """
        criteria = {t.name: self._type_criteria[t.name] for t in shortlist}
        criteria[NONE_OPTION] = self._type_criteria[NONE_OPTION]
        in_list = {t.id for t in shortlist}
        groups = [g for g in self.e.c.confusable_groups if in_list & set(g["ids"])]
        questions = {"appointment_type": {
            "type": "choice",
            "instructions": "Which of this clinic's appointment types is the caller asking for?",
            "criteria": criteria,
        }}
        for i, g in enumerate(groups):
            members = {self.e.c.appointment_types[t].name: self._type_criteria[self.e.c.appointment_types[t].name]
                       for t in g["ids"]}
            members[NOT_SPECIFIED] = "What the caller said fits more than one of these, or doesn't say which."
            questions[f"group_{i}"] = {
                "type": "choice",
                "instructions": "Based only on what the caller actually said (not on what is most common), "
                                "which one of these do they mean? Choose 'not specified' if their words "
                                "would fit more than one of them.",
                "criteria": members,
            }
        data = await self.client.system_one(
            self._state(query, {"new_patient": new_patient, "patient_age_years": patient_age}, with_catalog=False),
            questions)

        ranked = sorted(data["answers"]["appointment_type"]["probabilities"].items(), key=lambda kv: -kv[1])
        top_name, top_p = ranked[0]
        if top_name == NONE_OPTION:
            return {"status": "no_match", "query": query, "candidates": [], "confidence": round(top_p, 3)}
        top = self._type_by_option[top_name]

        verdicts = []  # (group, decision, chosen_type) for each of top's groups that was asked
        for i, g in enumerate(groups):
            if top.id not in g["ids"]:
                continue
            ans = data["answers"][f"group_{i}"]
            choice, p = ans["choice"], ans["probabilities"][ans["choice"]]
            if choice == NOT_SPECIFIED:
                verdicts.append((g, "ask", None))
            else:
                chosen = self._type_by_option[choice]
                said_it = self.e.mentions_discriminator(query, chosen.id, g)
                if chosen.id != top.id and p < self.GROUP_SURE_P:
                    verdicts.append((g, "ask", None))           # the two questions disagree
                elif not said_it and p < self.GROUP_SURE_P:
                    verdicts.append((g, "ask", None))           # nothing they said singles it out
                else:
                    verdicts.append((g, "commit", chosen))

        asks = [g for g, d, _ in verdicts if d == "ask"]
        if asks:
            g = min(asks, key=lambda g: len(g["ids"]))
            types = [self.e.c.appointment_types[i] for i in g["ids"]]
            out = self.e.finalize_type_match(types, False, g["question"], new_patient, patient_age)
        else:
            chosen = next((c for _, d, c in verdicts if d == "commit"), top)
            runner_up = ranked[1][1] if len(ranked) > 1 and ranked[1][0] != NONE_OPTION else 0.0
            if verdicts or (top_p >= self.CLEAR_P and runner_up < self.MIN_P + 0.03):
                out = self.e.finalize_type_match([chosen], True, None, new_patient, patient_age)
            else:  # no group to consult and the main choice is split: offer the contenders
                types = [self._type_by_option[n] for n, p in ranked if n != NONE_OPTION and p >= self.MIN_P][:4]
                out = self.e.finalize_type_match(types or [chosen], len(types) <= 1, None, new_patient, patient_age)
        out["confidence"] = round(top_p, 3)
        out["groups_asked"] = len(groups)
        return out

    # ---- providers ---------------------------------------------------------
    async def find_providers(self, description: str, new_patient: Optional[bool] = None) -> dict:
        data = await self.client.system_one(
            self._state(description, {"new_patient": new_patient}),
            {"provider": {
                "type": "choice",
                "instructions": "Which provider is the caller describing? Several providers can share a "
                                "name; if the description fits more than one, spread probability across them.",
                "criteria": self._prov_criteria,
            }},
        )
        ranked = sorted(data["answers"]["provider"]["probabilities"].items(), key=lambda kv: -kv[1])
        cands = [self._prov_by_option[n] for n, p in ranked if n != NONE_OPTION and p >= self.MIN_P][:5]
        if not cands:
            return {"status": "no_match", "query": description, "candidates": []}
        out = {"status": "match" if len(cands) == 1 else "multiple",
               "candidates": [self.e.describe_provider(p) for p in cands]}
        if len(cands) > 1:
            out["hint"] = "Several providers match; confirm which one using specialty, title or location."
        return out


_resolvers: dict = {}


def get_resolver(engine: CatalogEngine, context: str = "full") -> JevResolver:
    key = (id(engine), context)
    if key not in _resolvers:
        _resolvers[key] = JevResolver(engine, context=context)
    return _resolvers[key]


def cost_usd(input_tokens: int) -> float:
    return input_tokens * PRICE_PER_M_INPUT / 1e6
