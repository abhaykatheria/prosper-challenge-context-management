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
