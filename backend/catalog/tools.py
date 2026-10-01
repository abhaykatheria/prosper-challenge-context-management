#
# LLM-facing tools over the CatalogEngine.
#
# Two kinds of functions a node can expose:
#   * TOOLS   — lookups that return data and keep the conversation on the same node
#               (find_appointment_types, search_slots, ...).
#   * ACTIONS — server-side guards attached to an edge. They run before the
#               transition; if they fail, the transition is refused and the LLM
#               gets a structured error with repair alternatives (self-correction).
#
# Both read the per-call `state` dict (Pipecat Flows' flow_manager.state), so facts
# collected once — new vs existing patient, referral — are applied automatically
# instead of relying on the LLM to re-pass them on every call.
#

from dataclasses import dataclass
from typing import Callable, Optional

from .engine import CatalogEngine
from .models import Catalog

_engine: Optional[CatalogEngine] = None


def get_engine() -> CatalogEngine:
    """Process-wide engine: bookings made in one test call are visible in the next."""
    global _engine
    if _engine is None:
        _engine = CatalogEngine(Catalog.load())
    return _engine


def reset_engine(engine: Optional[CatalogEngine] = None) -> None:
    global _engine
    _engine = engine


@dataclass
class ToolSpec:
    name: str
    description: str
    properties: dict
    required: list
    fn: Callable  # (engine, args, state) -> dict


def _bool(state: dict, key: str) -> Optional[bool]:
    v = state.get(key)
    return v if isinstance(v, bool) else None


# ---------------------------------------------------------------------------
# TOOLS (stay on node)
# ---------------------------------------------------------------------------

def _find_appointment_types(e: CatalogEngine, a: dict, s: dict) -> dict:
    return e.match_appointment_types(a["request"], new_patient=_bool(s, "new_patient"))


def _find_providers(e: CatalogEngine, a: dict, s: dict) -> dict:
    return e.find_providers(
        name=a.get("name"), specialty=a.get("specialty"), language=a.get("language"),
        location_id=a.get("location_id"), appointment_type_id=a.get("appointment_type_id"),
        new_patient=_bool(s, "new_patient"),
    )


def _find_locations(e: CatalogEngine, a: dict, s: dict) -> dict:
    return e.find_locations(a.get("query"), appointment_type_id=a.get("appointment_type_id"))


def _search_slots(e: CatalogEngine, a: dict, s: dict) -> dict:
    tid = a.get("appointment_type_id") or s.get("appointment_type_id")
    if not tid:
        return {"status": "error", "message": "No appointment type selected yet."}
    return e.search_slots(
        tid,
        provider_id=a.get("provider_id", s.get("provider_id")),
        location_id=a.get("location_id", s.get("location_id")),
        new_patient=_bool(s, "new_patient"),
        has_referral=_bool(s, "has_referral"),
        language=s.get("language"),
        earliest_date=a.get("earliest_date"),
        weekdays=a.get("weekdays"),
        time_of_day=a.get("time_of_day"),
    )


TOOLS = {
    t.name: t
    for t in [
        ToolSpec(
            "find_appointment_types",
            "Match what the caller needs (in their own words) to our appointment types. "
            "Returns 'match', 'ambiguous' (ask the suggested_question; never guess), "
            "'existing_patients_only' (offer the returned alternatives), 'not_offered', or 'no_match'. Call this instead of assuming a type exists.",
            {"request": {"type": "string", "description": "The caller's need in their words, e.g. 'knee MRI', 'yearly checkup', 'my kid is sick'."}},
            ["request"],
            _find_appointment_types,
        ),
        ToolSpec(
            "find_providers",
            "Look up providers by name and/or filters. Names are NOT unique: if several match, "
            "confirm which one using specialty, title or location before proceeding.",
            {
                "name": {"type": "string", "description": "Provider name as the caller said it, e.g. 'Dr. Chen'."},
                "specialty": {"type": "string"},
                "language": {"type": "string", "description": "Language the provider must speak."},
                "location_id": {"type": "string"},
                "appointment_type_id": {"type": "string", "description": "Only providers who do this appointment type."},
            },
            [],
            _find_providers,
        ),
        ToolSpec(
            "find_locations",
            "Look up clinic locations (address, phone, hours). Omit query to list all. To answer "
            "'where can I get X?', first resolve X with find_appointment_types, then pass its "
            "appointment_type_id here: that returns only sites where X is actually bookable.",
            {
                "query": {"type": "string", "description": "Location name, neighborhood or street as said by the caller."},
                "appointment_type_id": {"type": "string", "description": "Only locations where this appointment type can be booked."},
            },
            [],
            _find_locations,
        ),
        ToolSpec(
            "search_slots",
            "Find the soonest open slots for the selected appointment type, honoring every booking "
            "policy and the caller's provider/location/time preferences. Returns at most 3 slots, "
            "or 'blocked' with the reason and alternatives.",
            {
                "provider_id": {"type": "string", "description": "Only if the caller wants a specific provider."},
                "location_id": {"type": "string", "description": "Only if the caller wants a specific location."},
                "earliest_date": {"type": "string", "description": "YYYY-MM-DD; no earlier than this."},
                "weekdays": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['Tuesday','Thursday']"},
                "time_of_day": {"type": "string", "enum": ["morning", "afternoon"]},
            },
            [],
            _search_slots,
        ),
    ]
}


# ---------------------------------------------------------------------------
# ACTIONS (edge guards) — return (ok, result_for_llm, state_updates)
# ---------------------------------------------------------------------------

def _select_service(e: CatalogEngine, a: dict, s: dict):
    """Only let the conversation advance to scheduling once the choice is bookable."""
    a = {**a,
         "appointment_type_id": e._resolve_id(a.get("appointment_type_id"), e.c.appointment_types),
         "provider_id": e._resolve_id(a.get("provider_id"), e.c.providers),
         "location_id": e._resolve_id(a.get("location_id"), e.c.locations)}
    tid = a.get("appointment_type_id")
    t = e.c.appointment_types.get(tid or "")
    if not t:
        return False, {"status": "error", "message": "Unknown appointment_type_id; use an id returned by find_appointment_types."}, {}
    has_ref = a.get("has_referral", s.get("has_referral"))
    if t.requires_referral and has_ref is None:
        return False, {"status": "need_info", "message": f"{t.name} requires a referral. Ask whether they have one, then call again with has_referral."}, {}
    options, blockers = e.solve(tid, a.get("provider_id"), a.get("location_id"),
                                _bool(s, "new_patient"), has_ref, s.get("language"))
    if blockers:
        return False, {"status": "blocked", "blockers": [b.to_dict() for b in blockers]}, {}
    updates = {"appointment_type_id": tid, "appointment_type": t.name,
               "provider_id": a.get("provider_id"), "location_id": a.get("location_id")}
    if has_ref is not None:
        updates["has_referral"] = has_ref
    if a.get("provider_id"):
        updates["provider"] = e.c.providers[a["provider_id"]].display
    if a.get("location_id"):
        updates["location"] = e.c.locations[a["location_id"]].name
    return True, {"status": "ok", "appointment_type": t.name,
                  "eligible_providers": len({p.id for p, _ in options}),
                  "eligible_locations": len({l.id for _, l in options})}, updates


def _hold_slot(e: CatalogEngine, a: dict, s: dict):
    from .engine import Slot
    try:
        slot = Slot.parse(a.get("slot_id", ""))
    except Exception:
        return False, {"status": "error", "message": "Use a slot_id exactly as returned by search_slots."}, {}
    return True, {"status": "ok", **e.describe_slot(slot)}, {"slot_id": a["slot_id"], "slot": e.describe_slot(slot)}


def _book(e: CatalogEngine, a: dict, s: dict):
    r = e.book(s.get("slot_id", ""), s.get("full_name", ""), s.get("date_of_birth", ""),
               bool(s.get("new_patient")), _bool(s, "has_referral"))
    if r["status"] != "booked":
        return False, r, {}
    return True, r, {"booking": r}


ACTIONS = {
    "select_service": _select_service,
    "hold_slot": _hold_slot,
    "book_appointment": _book,
}
