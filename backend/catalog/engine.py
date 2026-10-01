#
# CatalogEngine — deterministic retrieval + policy solver over the catalog.
#
# The core idea of the context-management approach: the LLM never reasons over
# the raw catalog. It asks narrow questions ("what matches 'knee MRI'?", "who can
# do appt_065 at a site near me?") and gets back a *small, pre-validated* answer.
# All booking policies are enforced here, in code, so an invalid combination is
# unrepresentable rather than merely discouraged by the prompt.
#
#   free text  --match-->  candidate ids  --solve(policies)-->  valid options
#                                                           \-> blockers + repairs
#

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Optional

from rapidfuzz import fuzz

from .models import AppointmentType, Catalog, Provider

# ---------------------------------------------------------------------------
# text matching
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_HONORIFICS = {"dr", "doctor", "doc", "md", "do", "np", "pa"}
_STOPWORDS = {"a", "an", "the", "my", "i", "need", "want", "to", "for", "with", "appointment", "visit", "book", "get", "some", "of", "please"}


def norm(text: str) -> str:
    text = _PUNCT.sub(" ", text.lower().replace("-", " ").replace("/", " "))
    return " ".join(text.split())


def _content(text: str) -> str:
    return " ".join(t for t in norm(text).split() if t not in _STOPWORDS)


def _phrase_score(query: str, phrase: str) -> float:
    """0-100 similarity between a caller's phrase and a catalog phrase.

    Order-insensitive token matching handles "knee MRI" vs "MRI - Knee"; plain
    ratio catches STT misspellings ("Holter" vs "halter").
    """
    q, p = _content(query), _content(phrase)
    if not q or not p:
        return 0.0
    if q == p:
        return 100.0
    # token_set alone scores a subset as 100 ("flu shot" vs alias "shot"), so blend
    # in token_sort, which penalises the words the phrase doesn't account for.
    blended = 0.6 * fuzz.token_set_ratio(q, p) + 0.4 * fuzz.token_sort_ratio(q, p)
    return max(blended, fuzz.ratio(q, p))


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class Blocker:
    """Why a request can't be booked as-is, plus how to repair it."""

    code: str
    message: str
    alternatives: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.alternatives:
            d["alternatives"] = self.alternatives
        return d


@dataclass
class Slot:
    start: datetime
    appointment_type_id: str
    provider_id: str
    location_id: str

    @property
    def slot_id(self) -> str:
        # Opaque handle: the LLM books by slot_id, so it can never mix up a
        # provider/location/type combination while transcribing ids.
        return f"{self.appointment_type_id}|{self.provider_id}|{self.location_id}|{self.start:%Y%m%dT%H%M}"

    @classmethod
    def parse(cls, slot_id: str) -> "Slot":
        t, p, l, ts = slot_id.split("|")
        return cls(datetime.strptime(ts, "%Y%m%dT%H%M"), t, p, l)


def spoken_time(dt: datetime) -> str:
    hour = dt.strftime("%I").lstrip("0")
    minute = "" if dt.minute == 0 else f":{dt.minute:02d}"
    return f"{dt:%A, %B} {dt.day} at {hour}{minute} {dt:%p}"


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class CatalogEngine:
    SLOT_GRID_MIN = 30
    HORIZON_DAYS = 21

    def __init__(self, catalog: Catalog, today: Optional[date] = None):
        self.c = catalog
        self.today = today or date.today()
        self.booked: set = set()        # (provider_id, datetime) taken by bookings this session
        self.bookings: dict = {}         # confirmation -> booking dict

    # ---- appointment types -------------------------------------------------
    def match_appointment_types(
        self, query: str, new_patient: Optional[bool] = None, limit: int = 4,
        patient_age: Optional[int] = None,
    ) -> dict:
        scored = []
        for t in self.c.appointment_types.values():
            phrases = [t.name] + t.aliases
            s = max(_phrase_score(query, ph) for ph in phrases)
            # Specialty words ("cardiology", "derm") are weak evidence on their own.
            s = max(s, 0.8 * _phrase_score(query, t.specialty))
            scored.append((s, t))
        scored.sort(key=lambda x: -x[0])

        top = [(s, t) for s, t in scored if s >= 60][: limit + 6]
        if patient_age is not None:
            top = self._apply_age(top, patient_age)
        top = top[: limit + 2]
        if not top:
            return {"status": "no_match", "query": query, "candidates": [],
                    "hint": "Ask the caller to describe the reason for the visit in other words."}

        best = top[0][0]
        # Keep only the competitive candidates: close to the best score.
        close = [(s, t) for s, t in top if s >= best - 8][:limit]

        # Pull in confusable siblings when a match lands inside a curated group:
        # "annual checkup" hitting Annual Physical should also surface Annual
        # Wellness Visit, even if its lexical score is lower.
        question = None
        exact = best >= 99 and (len(close) == 1 or close[1][0] < 97)
        if exact:
            close = close[:1]
        else:
            best_group = self.confusable_group(close[0][1].id, {t.id for _, t in close})
            if best_group:
                question = best_group["question"]
                known = {t.id for _, t in close}
                close = [(s, t) for s, t in close if t.id in best_group["ids"]]
                for tid in best_group["ids"]:
                    if tid not in known:
                        close.append((best - 10, self.c.appointment_types[tid]))

        clear = len(close) == 1 or (close[0][0] - close[1][0] >= 8 and close[0][0] >= 90)
        out = self.finalize_type_match([t for _, t in close], clear, question, new_patient, patient_age)
        out["score"] = round(best)  # lexical confidence; the cascade resolver keys off it
        return out

    def _apply_age(self, ranked: list, age: int) -> list:
        """Adults never get pediatric types; for a child, drop an adult type whenever a
        pediatric type from the same confusable group is also a candidate."""
        if age >= 18:
            return [(s, t) for s, t in ranked if t.specialty != "Pediatrics"] or ranked
        peds = {t.id for _, t in ranked if t.specialty == "Pediatrics"}
        if not peds:
            return ranked
        shadowed = {i for p in peds for g in self.c.group_by_type.get(p, []) for i in g["ids"]} - peds
        return [(s, t) for s, t in ranked if t.id not in shadowed]

    def confusable_group(self, top_id: str, ids_in_play: set):
        """Smallest curated confusable group containing the top hit and >=1 other candidate."""
        for g in sorted(self.c.group_by_type.get(top_id, []), key=lambda g: len(g["ids"])):
            if len(ids_in_play & set(g["ids"])) >= 2:
                return g
        return None

    def finalize_type_match(self, types: list, clear: bool, question: Optional[str],
                            new_patient: Optional[bool], patient_age: Optional[int] = None) -> dict:
        """Shared by every resolver (fuzzy, Jev): turn ranked candidate types into the
        tool result, applying patient context, offered-ness, repairs and bookability."""
        # The patient's age (from the DOB already collected) separates "Sick Visit" from
        # "Pediatric Sick Visit" when the caller never says "child" - deterministically,
        # instead of asking a question we already know the answer to.
        if patient_age is not None and len(types) > 1:
            peds = [t for t in types if t.specialty == "Pediatrics"]
            keep = peds if patient_age < 18 else [t for t in types if t.specialty != "Pediatrics"]
            if keep and len(keep) < len(types):
                types = keep
                if len(types) == 1:
                    clear, question = True, None
        # Patient context narrows further: a new patient can't book existing-only types.
        if new_patient is True and len(types) > 1:
            allowed = [t for t in types if t.new_patients_allowed]
            if allowed:
                types = allowed
                clear = clear or len(types) == 1
        candidates = [self.describe_type(t) for t in types]
        if not any(self.c.providers_by_type.get(t.id) for t in types):
            return {"status": "not_offered", "candidates": candidates,
                    "related_offered": self._related_offered_types(types[0])}
        if clear:
            only = types[0]
            if not self.c.providers_by_type.get(only.id):
                return {"status": "not_offered", "candidates": candidates[:1],
                        "related_offered": self._related_offered_types(only)}
            if new_patient and not only.new_patients_allowed:
                # Same contract as the guards: a blocker always travels with its repair,
                # so the LLM never has to improvise an alternative.
                return {"status": "existing_patients_only", "candidates": candidates[:1],
                        "message": f"{only.name} is only for established patients.",
                        "alternatives": [self.describe_type(x) for x in self._new_patient_alternatives(only)]}
            # Where it's *actually* bookable (provider x site x capability), which is
            # not the same as "sites with the equipment".
            opts, _ = self.solve(only.id, new_patient=new_patient)
            cand = candidates[0]
            cand["bookable_at"] = [{"location_id": l.id, "name": l.name}
                                   for l in sorted({l.id: l for _, l in opts}.values(), key=lambda l: l.id)]
            cand["provider_count"] = len({p.id for p, _ in opts})
            return {"status": "match", "candidates": [cand]}
        out = {"status": "ambiguous", "candidates": candidates, "differs_by": self._differences(types)}
        if question:
            out["suggested_question"] = question
        return out

    def describe_type(self, t: AppointmentType) -> dict:
        d = {
            "appointment_type_id": t.id,
            "name": t.name,
            "specialty": t.specialty,
            "duration_min": t.duration_min,
            "requires_referral": t.requires_referral,
            "new_patients_allowed": t.new_patients_allowed,
        }
        if t.description:
            d["description"] = t.description
        if t.required_capability:
            d["only_at_sites_with"] = t.required_capability
        if not self.c.providers_by_type.get(t.id):
            d["offered"] = False
        return d

    @staticmethod
    def _differences(types: list) -> list:
        """Which attributes actually distinguish these candidates (drives the clarifying question)."""
        diffs = []
        for attr in ("specialty", "duration_min", "requires_referral", "new_patients_allowed", "required_capability"):
            if len({getattr(t, attr) for t in types}) > 1:
                diffs.append(attr)
        return diffs

    # ---- providers ---------------------------------------------------------
    def find_providers(
        self,
        name: Optional[str] = None,
        specialty: Optional[str] = None,
        language: Optional[str] = None,
        location_id: Optional[str] = None,
        appointment_type_id: Optional[str] = None,
        new_patient: Optional[bool] = None,
        limit: int = 5,
    ) -> dict:
        pool = list(self.c.providers.values())
        if name:
            q = " ".join(t for t in norm(name).split() if t not in _HONORIFICS)
            scored = []
            for p in pool:
                full = " ".join(t for t in norm(p.name).split() if t not in _HONORIFICS)
                last = norm(p.last_name)
                # "Chen" should match every Chen equally; "Emily Chen" should prefer Emily.
                s = max(fuzz.ratio(q, full), fuzz.ratio(q, last) if len(q.split()) == 1 else 0,
                        fuzz.token_sort_ratio(q, full))
                scored.append((s, p))
            best = max(s for s, _ in scored)
            if best < 70:
                return {"status": "no_match", "query": name, "candidates": []}
            pool = [p for s, p in scored if s >= max(70, best - 6)]
        if specialty:
            pool = [p for p in pool if _phrase_score(specialty, p.specialty) >= 75]
        if language:
            pool = [p for p in pool if any(norm(language) == norm(l) for l in p.languages)]
        if location_id:
            pool = [p for p in pool if location_id in p.location_ids]
        if appointment_type_id:
            pool = [p for p in pool if appointment_type_id in p.appointment_type_ids]
        if new_patient:
            pool = [p for p in pool if p.accepting_new_patients] or pool

        if not pool:
            return {"status": "no_match", "candidates": []}
        cands = [self.describe_provider(p) for p in pool[:limit]]
        status = "match" if len(pool) == 1 else "multiple"
        out = {"status": status, "candidates": cands}
        if len(pool) > limit:
            out["more"] = len(pool) - limit
        if status == "multiple" and name:
            out["hint"] = "Several providers match that name; confirm which one using specialty, title or location."
        return out

    def describe_provider(self, p: Provider) -> dict:
        return {
            "provider_id": p.id,
            "name": p.name,
            "title": p.title,
            "specialty": p.specialty,
            "locations": [self.c.locations[l].name for l in p.location_ids],
            "accepting_new_patients": p.accepting_new_patients,
            "languages": p.languages,
        }

    # ---- locations ---------------------------------------------------------
    def find_locations(self, query: Optional[str] = None, capability: Optional[str] = None,
                       appointment_type_id: Optional[str] = None) -> dict:
        locs = list(self.c.locations.values())
        if capability:
            locs = [l for l in locs if capability in l.capabilities]
        if appointment_type_id:
            # Where it's actually bookable (a provider who does it practises at a
            # site with the required capability) - not merely "has the equipment".
            opts, _ = self.solve(appointment_type_id)
            ok = {l.id for _, l in opts}
            locs = [l for l in locs if l.id in ok]
        if query:
            scored = sorted(
                ((max(_phrase_score(query, l.name), 0.85 * _phrase_score(query, l.address)), l) for l in locs),
                key=lambda x: -x[0],
            )
            best = scored[0][0] if scored else 0
            locs = [l for s, l in scored if s >= max(60, best - 5)]
        return {
            "status": "match" if len(locs) == 1 else ("multiple" if locs else "no_match"),
            "candidates": [self.describe_location(l) for l in locs],
        }

    def describe_location(self, l) -> dict:
        return {"location_id": l.id, "name": l.name, "address": f"{l.address}, {l.city}",
                "phone": l.phone, "hours": l.hours, "services": l.capabilities}

    # ---- the policy solver -------------------------------------------------
    def solve(
        self,
        appointment_type_id: str,
        provider_id: Optional[str] = None,
        location_id: Optional[str] = None,
        new_patient: Optional[bool] = None,
        has_referral: Optional[bool] = None,
        language: Optional[str] = None,
    ) -> tuple:
        """Enumerate every (provider, location) that satisfies ALL policies.

        Returns (options, blockers). When options is empty, blockers explain why
        and carry concrete alternatives, so the LLM can self-correct in one turn
        instead of guessing.
        """
        c = self.c
        # LLMs sometimes pass a display name where an id belongs; accept exact names,
        # reject anything else with a correctable error rather than crashing.
        location_id = self._resolve_id(location_id, c.locations)
        provider_id = self._resolve_id(provider_id, c.providers)
        if location_id and location_id not in c.locations:
            return [], [Blocker("unknown_location", f"Unknown location '{location_id}'. Use a location_id from find_locations.")]
        t = c.appointment_types.get(self._resolve_id(appointment_type_id, c.appointment_types))
        if not t:
            return [], [Blocker("unknown_type", f"Unknown appointment type id {appointment_type_id}. Call find_appointment_types first.")]

        blockers = []
        if not c.providers_by_type.get(t.id):
            return [], [Blocker("not_offered", f"{t.name} is not currently offered by any of our providers.",
                                self._related_offered_types(t))]

        if new_patient and not t.new_patients_allowed:
            blockers.append(Blocker(
                "existing_patients_only",
                f"{t.name} is only for established patients.",
                [self.describe_type(x) for x in self._new_patient_alternatives(t)],
            ))
        if t.requires_referral and has_referral is False:
            blockers.append(Blocker("referral_required", f"{t.name} requires a referral on file before booking."))
        if blockers:
            return [], blockers

        providers = [c.providers[p] for p in c.providers_by_type[t.id]]
        if provider_id:
            p = c.providers.get(provider_id)
            if not p:
                return [], [Blocker("unknown_provider", f"Unknown provider id {provider_id}.")]
            if t.id not in p.appointment_type_ids:
                return [], [Blocker(
                    "provider_does_not_offer",
                    f"{p.display} does not do {t.name}.",
                    [self.describe_provider(x) for x in providers if (not new_patient or x.accepting_new_patients)][:4],
                )]
            providers = [p]
        if new_patient:
            accepting = [p for p in providers if p.accepting_new_patients]
            if not accepting:
                alts = [self.describe_provider(x) for x in c.providers.values()
                        if x.accepting_new_patients and t.id in x.appointment_type_ids][:4]
                who = providers[0].display if provider_id else f"providers who do {t.name}"
                return [], [Blocker("not_accepting_new_patients",
                                    f"{who} {'is' if provider_id else 'are'} not accepting new patients.", alts)]
            providers = accepting
        if language:
            spoken = [p for p in providers if any(norm(language) == norm(l) for l in p.languages)]
            if not spoken:
                return [], [Blocker("language_unavailable", f"No provider for {t.name} speaks {language}.",
                                    [self.describe_provider(x) for x in providers][:4])]
            providers = spoken

        options, cap_blocked = [], []
        for p in providers:
            for lid in p.location_ids:
                loc = c.locations[lid]
                if location_id and lid != location_id:
                    continue
                if t.required_capability and t.required_capability not in loc.capabilities:
                    cap_blocked.append((p, loc))
                    continue
                options.append((p, loc))

        if not options:
            if location_id:
                loc = c.locations[location_id]
                if t.required_capability and t.required_capability not in loc.capabilities:
                    msg = f"{loc.name} does not have {t.required_capability.replace('_', ' ')}, which {t.name} requires."
                else:
                    msg = f"No eligible provider does {t.name} at {loc.name}."
                alt_opts, _ = self.solve(t.id, provider_id, None, new_patient, has_referral, language)
                alts = sorted({o[1].id for o in alt_opts})
                return [], [Blocker("location_ineligible", msg,
                                    [c.locations[a].brief() for a in alts])]
            return [], [Blocker("no_eligible_site",
                                f"None of the eligible providers practice at a site that supports {t.name}.")]
        return options, []

    @staticmethod
    def _resolve_id(value, table: dict):
        if not value or value in table:
            return value
        hits = [k for k, v in table.items() if norm(v.name) == norm(value)]
        return hits[0] if len(hits) == 1 else value

    def _new_patient_alternatives(self, t: AppointmentType) -> list:
        same = [x for x in self.c.appointment_types.values()
                if x.specialty == t.specialty and x.new_patients_allowed and self.c.providers_by_type.get(x.id)]
        if not same:
            same = [self.c.appointment_types[i] for i in ("appt_001", "appt_000")]
        return same[:3]

    def _related_offered_types(self, t: AppointmentType) -> list:
        return [
            {"name": x.name, "appointment_type_id": x.id}
            for x in self.c.appointment_types.values()
            if x.specialty == t.specialty and x.id != t.id and self.c.providers_by_type.get(x.id)
        ][:3]

    # ---- mock availability -------------------------------------------------
    @staticmethod
    def _h(*parts) -> int:
        return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)

    def provider_location_on(self, p: Provider, d: date) -> Optional[str]:
        """Which site the provider works at on a given day (one site per day, so
        a multi-site provider can never be double-booked across locations)."""
        if d.weekday() >= 5:
            return None
        if d.weekday() == self._h(p.id, "dayoff") % 7:  # ~5/7 providers get one weekday off
            return None
        locs = sorted(p.location_ids)
        return locs[(d.weekday() + self._h(p.id)) % len(locs)]

    def _slots_for(self, t: AppointmentType, p: Provider, lid: str, d: date) -> list:
        if self.provider_location_on(p, d) != lid:
            return []
        open_h, close_h = 8, 17
        m = re.search(r"(\d+):(\d+)-(\d+):(\d+)", self.c.locations[lid].hours)
        if m:
            open_h, close_h = int(m.group(1)), int(m.group(3))
        out = []
        cur = datetime.combine(d, time(open_h))
        end = datetime.combine(d, time(close_h))
        while cur + timedelta(minutes=t.duration_min) <= end:
            if cur.hour != 12:  # lunch
                taken = self._h(p.id, cur.isoformat()) % 100 < 70  # ~70% pre-booked
                if not taken and (p.id, cur) not in self.booked:
                    out.append(Slot(cur, t.id, p.id, lid))
            cur += timedelta(minutes=self.SLOT_GRID_MIN)
        return out

    def search_slots(
        self,
        appointment_type_id: str,
        provider_id: Optional[str] = None,
        location_id: Optional[str] = None,
        new_patient: Optional[bool] = None,
        has_referral: Optional[bool] = None,
        language: Optional[str] = None,
        earliest_date: Optional[str] = None,
        weekdays: Optional[list] = None,
        time_of_day: Optional[str] = None,
        limit: int = 3,
    ) -> dict:
        options, blockers = self.solve(appointment_type_id, provider_id, location_id,
                                       new_patient, has_referral, language)
        if blockers:
            return {"status": "blocked", "blockers": [b.to_dict() for b in blockers]}

        t = self.c.appointment_types[appointment_type_id]
        start = self.today + timedelta(days=1)
        if earliest_date:
            try:
                start = max(start, date.fromisoformat(earliest_date))
            except ValueError:
                pass
        wd = {w.lower()[:3] for w in (weekdays or [])}

        found = []
        for i in range(self.HORIZON_DAYS):
            d = start + timedelta(days=i)
            if wd and d.strftime("%a").lower() not in wd:
                continue
            for p, loc in options:
                for s in self._slots_for(t, p, loc.id, d):
                    if time_of_day == "morning" and s.start.hour >= 12:
                        continue
                    if time_of_day == "afternoon" and s.start.hour < 12:
                        continue
                    found.append(s)
            if len({(s.provider_id, s.start.date()) for s in found}) >= limit * 2:
                break
        found.sort(key=lambda s: s.start)

        # Diversify: soonest first, but don't offer three back-to-back slots with
        # the same provider on the same morning - give the caller real choices.
        picked, seen = [], set()
        for s in found:
            k = (s.provider_id, s.start.date())
            if k not in seen:
                picked.append(s)
                seen.add(k)
            if len(picked) == limit:
                break
        for s in found:
            if len(picked) == limit:
                break
            if s not in picked:
                picked.append(s)
        picked.sort(key=lambda s: s.start)

        if not picked:
            return {"status": "no_availability",
                    "message": f"No openings for {t.name} in the next {self.HORIZON_DAYS} days with those preferences."}
        out = {
            "status": "ok",
            "appointment_type": t.name,
            "eligible_providers": len({p.id for p, _ in options}),
            "eligible_locations": len({l.id for _, l in options}),
            "slots": [self.describe_slot(s) for s in picked],
        }
        if t.requires_referral and has_referral is None:
            out["note"] = f"{t.name} requires a referral; confirm the caller has one before booking."
        return out

    def describe_slot(self, s: Slot) -> dict:
        p, l = self.c.providers[s.provider_id], self.c.locations[s.location_id]
        return {"slot_id": s.slot_id, "when": spoken_time(s.start),
                "provider": f"{p.name}, {p.title}", "location": l.name}

    # ---- booking (re-validates everything) ---------------------------------
    def book(
        self,
        slot_id: str,
        patient_name: str,
        date_of_birth: str,
        new_patient: bool,
        has_referral: Optional[bool] = None,
    ) -> dict:
        try:
            s = Slot.parse(slot_id)
        except Exception:
            return {"status": "error", "message": "Invalid slot_id. Use a slot_id returned by search_slots."}
        t = self.c.appointment_types.get(s.appointment_type_id)
        if t and t.requires_referral and not has_referral:
            return {"status": "blocked",
                    "blockers": [Blocker("referral_required", f"{t.name} requires a referral; ask the caller to confirm they have one.").to_dict()]}
        options, blockers = self.solve(s.appointment_type_id, s.provider_id, s.location_id,
                                       new_patient, has_referral)
        if blockers:
            return {"status": "blocked", "blockers": [b.to_dict() for b in blockers]}
        p = self.c.providers[s.provider_id]
        if s.start not in [x.start for x in self._slots_for(t, p, s.location_id, s.start.date())]:
            return {"status": "slot_unavailable", "message": "That time was just taken. Search slots again."}

        self.booked.add((s.provider_id, s.start))
        conf = f"PX-{self._h(slot_id, patient_name) % 1_000_000:06d}"
        booking = {"confirmation": conf, "patient_name": patient_name, "date_of_birth": date_of_birth,
                   "appointment_type": t.name, **self.describe_slot(s),
                   "address": self.c.locations[s.location_id].brief()["address"]}
        self.bookings[conf] = booking
        return {"status": "booked", **booking}
