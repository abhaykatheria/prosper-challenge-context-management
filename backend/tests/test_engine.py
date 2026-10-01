"""Deterministic tests for the catalog engine: matching + policy guarantees.

Run:  cd backend && .venv/bin/python -m pytest -q
"""

import itertools
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import Catalog, CatalogEngine  # noqa: E402

CAT = Catalog.load()


@pytest.fixture
def e():
    return CatalogEngine(CAT, today=date(2026, 9, 28))


@pytest.mark.parametrize("query,expected", [
    ("knee MRI", "MRI - Knee"),
    ("MRI of my knee", "MRI - Knee"),
    ("halter monitor", "Holter Monitor Fitting"),  # STT misspelling
    ("flu shot", "Flu Shot"),
    ("cat scan", "CT Scan"),
    ("teeth cleaning", "Dental Cleaning"),
    ("checkup for my kid", "Well-Child Visit"),
    ("heart doctor", "Cardiology Consultation"),
])
def test_clear_matches(e, query, expected):
    r = e.match_appointment_types(query)
    assert r["status"] == "match"
    assert r["candidates"][0]["name"] == expected


@pytest.mark.parametrize("query,names", [
    ("mri", {"MRI - Brain", "MRI - Spine", "MRI - Knee"}),
    ("annual checkup", {"Annual Physical", "Annual Wellness Visit"}),
    ("follow up", {"Follow-up Visit", "Follow-up Consultation", "Telehealth Follow-up"}),
])
def test_ambiguous_asks(e, query, names):
    r = e.match_appointment_types(query)
    assert r["status"] == "ambiguous"
    assert {c["name"] for c in r["candidates"]} == names
    assert r.get("suggested_question")


@pytest.mark.parametrize("query", ["eye exam", "urology", "physical therapy"])
def test_not_offered(e, query):
    assert e.match_appointment_types(query)["status"] == "not_offered"


def test_bookable_at_is_not_equipment(e):
    # Mission District has imaging, but no radiologist practises there.
    r = e.match_appointment_types("knee mri")
    names = {x["name"] for x in r["candidates"][0]["bookable_at"]}
    assert names == {"Downtown Health Center", "Midtown Medical Group"}


def test_duplicate_names_return_all(e):
    r = e.find_providers(name="Maria Garcia")
    assert r["status"] == "multiple" and len(r["candidates"]) == 2


def test_solver_never_violates_policies(e):
    """Exhaustive: every option the solver returns satisfies every catalog rule."""
    for t, new, ref in itertools.product(CAT.appointment_types.values(), [True, False], [True, False]):
        options, _ = e.solve(t.id, new_patient=new, has_referral=ref)
        for p, loc in options:
            assert t.id in p.appointment_type_ids
            assert loc.id in p.location_ids
            if t.required_capability:
                assert t.required_capability in loc.capabilities
            if t.requires_referral:
                assert ref
            if new:
                assert t.new_patients_allowed and p.accepting_new_patients


def test_blockers_carry_repairs(e):
    _, b = e.solve("appt_020", provider_id="prov_000", new_patient=True, has_referral=True)
    assert b[0].code == "not_accepting_new_patients"
    assert {a["provider_id"] for a in b[0].alternatives} >= {"prov_046"}
    _, b = e.solve("appt_065", location_id="loc_000", has_referral=True)
    assert b[0].code == "location_ineligible"
    assert {a["location_id"] for a in b[0].alternatives} == {"loc_004", "loc_005"}


def test_booking_revalidates_and_prevents_double_booking(e):
    slot = e.search_slots("appt_002", new_patient=True)["slots"][0]["slot_id"]
    assert e.book(slot, "A", "1990-01-01", new_patient=True)["status"] == "booked"
    assert e.book(slot, "B", "1990-01-01", new_patient=True)["status"] == "slot_unavailable"
    mri = e.search_slots("appt_065", new_patient=False, has_referral=True)["slots"][0]["slot_id"]
    assert e.book(mri, "C", "1990-01-01", new_patient=True, has_referral=True)["status"] == "blocked"


def test_provider_one_site_per_day(e):
    for p in CAT.providers.values():
        for i in range(14):
            d = date(2026, 9, 28 + i) if i < 3 else date(2026, 10, i - 2)
            sites = {lid for lid in p.location_ids if e.provider_location_on(p, d) == lid}
            assert len(sites) <= 1


def test_new_patient_match_carries_repair(e):
    r = e.match_appointment_types("therapy", new_patient=True)
    assert r["status"] == "existing_patients_only"
    assert "Psychiatric Evaluation" in {a["name"] for a in r["alternatives"]}
    assert e.match_appointment_types("mental health")["candidates"][0]["name"] == "Psychiatric Evaluation"


def test_patient_age_separates_pediatric_types(e):
    assert e.match_appointment_types("sick visit", patient_age=4)["candidates"][0]["name"] == "Pediatric Sick Visit"
    assert e.match_appointment_types("sick visit", patient_age=40)["candidates"][0]["name"] == "Sick Visit"


def test_jev_outage_falls_back_to_fuzzy(e, monkeypatch):
    """A Jev error or timeout must degrade to the local matcher, never break the call."""
    import asyncio
    import catalog.jev as jev
    from catalog.tools import _find_appointment_types

    class Down:
        async def match_appointment_types(self, *a, **k):
            raise RuntimeError("Jev unavailable")

    monkeypatch.setattr(jev, "get_resolver", lambda engine, context="full": Down())
    r = asyncio.run(_find_appointment_types(e, {"request": "knee mri"}, {"_resolver": "jev"}))
    assert r["resolver"] == "fuzzy_fallback" and r["candidates"][0]["name"] == "MRI - Knee"
    # cascade: confident local match never calls Jev at all
    r = asyncio.run(_find_appointment_types(e, {"request": "knee mri"}, {"_resolver": "cascade"}))
    assert r["resolver"] == "fuzzy"


def test_age_filter_only_uses_relevant_candidates(e):
    assert e.match_appointment_types("urgent care", patient_age=8)["candidates"][0]["name"] == "Urgent Care Visit"


@pytest.mark.parametrize("query,gold", [
    ("my chest gets tight on stairs, seeing a heart specialist", "appt_020"),
    ("my daughter has had a fever since last night", "appt_017"),
    ("spy rometry test", "appt_079"),
    ("I need some kind of scan", "appt_066"),
])
def test_retrieval_recall_for_hybrid(e, query, gold):
    """The hybrid resolver relies on fuzzy *recall*: gold must be in the shortlist even
    when fuzzy's top-1 is wrong."""
    from catalog.tools import _shortlist
    assert gold in {t.id for t in _shortlist(e, e.rank_appointment_types(query))}


def test_hybrid_fast_path_and_oversize_fallback(e, monkeypatch):
    import asyncio
    import catalog.jev as jev
    from catalog.tools import _find_appointment_types

    class Never:
        async def match_appointment_types(self, *a, **k):
            raise AssertionError("fast path must not call Jev")

    monkeypatch.setattr(jev, "get_resolver", lambda engine, context="full": Never())
    r = asyncio.run(_find_appointment_types(e, {"request": "pacemaker check"}, {"_resolver": "hybrid"}))
    assert r["resolver"] == "fuzzy" and r["candidates"][0]["name"] == "Pacemaker Check"

    class Oversize:
        async def match_appointment_types(self, *a, **k):
            raise jev.ContextTooLarge("too big")

    monkeypatch.setattr(jev, "get_resolver", lambda engine, context="full": Oversize())
    r = asyncio.run(_find_appointment_types(e, {"request": "I need an MRI"}, {"_resolver": "hybrid"}))
    assert r["resolver"] == "fuzzy_fallback"


def test_jev_client_refuses_oversized_requests():
    import asyncio
    import catalog.jev as jev
    client = jev.JevClient(api_key="unused")
    with pytest.raises(jev.ContextTooLarge):
        asyncio.run(client.system_one("x" * 120_000, {}))


# ---- hybrid ambiguity logic -------------------------------------------------

def test_stemming_fixes_plural_recall(e):
    from catalog.tools import _shortlist
    ids = {t.id for t in _shortlist(e, e.rank_appointment_types("go over all my medications with my primary doctor"))}
    assert "appt_009" in ids  # Medication Review


def test_discriminators_from_enrichment(e):
    g = next(g for g in CAT.confusable_groups if set(g["ids"]) == {"appt_072", "appt_073"})
    assert {"fasting", "cholesterol"} <= e.discriminators("appt_073", g)
    assert not e.mentions_discriminator("a blood test", "appt_072", g)


def test_category_group_uses_callers_words(e):
    assert "appt_066" in e.category_group("I need some kind of scan")["ids"]
    assert e.category_group("I want to see a doctor") is None


class _FakeJev:
    """Answers every question deterministically: main choice -> `main`; each group
    question -> `group_answers.get(frozenset(options))`, else 'not specified'."""

    def __init__(self, main, group_answers=None):
        self.main, self.group_answers, self.calls = main, group_answers or {}, []

    async def system_one(self, state, questions):
        answers = {}
        for k, q in questions.items():
            opts = list(q["criteria"])
            if k == "appointment_type":
                pick, p = self.main
            else:
                pick, p = self.group_answers.get(frozenset(o for o in opts if o != "not specified"), ("not specified", 0.9))
            probs = {o: (p if o == pick else (1 - p) / (len(opts) - 1)) for o in opts}
            answers[k] = {"type": "choice", "choice": pick, "probabilities": probs, "confidence": p}
        return {"answers": answers, "usage": {"input_tokens": 0}}


def _hybrid_with(e, fake, text):
    import asyncio
    from catalog.jev import JevResolver
    from catalog.tools import _shortlist
    r = JevResolver(e, client=fake)
    sl = _shortlist(e, e.rank_appointment_types(text))
    return asyncio.run(r.match_appointment_types(text, shortlist=sl))


def test_confident_pick_without_distinguishing_words_asks(e):
    blood = frozenset({"Blood Draw / Lab Work", "Fasting Blood Test"})
    out = _hybrid_with(e, _FakeJev(("Blood Draw / Lab Work", 1.0), {blood: ("Blood Draw / Lab Work", 0.7)}), "a blood test")
    assert out["status"] == "ambiguous" and out["suggested_question"]


def test_distinguishing_word_commits(e):
    fu = frozenset({"Follow-up Visit", "Follow-up Consultation", "Telehealth Follow-up"})
    out = _hybrid_with(e, _FakeJev(("Telehealth Follow-up", 1.0), {fu: ("Telehealth Follow-up", 0.8)}),
                       "a video appointment to follow up with my doctor")
    assert out["status"] == "match" and out["candidates"][0]["name"] == "Telehealth Follow-up"


def test_group_says_not_specified_asks(e):
    out = _hybrid_with(e, _FakeJev(("MRI - Brain", 0.95)), "I need an MRI")
    assert out["status"] == "ambiguous"
    assert {c["name"] for c in out["candidates"]} >= {"MRI - Brain", "MRI - Knee", "MRI - Spine"}
