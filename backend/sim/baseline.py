#
# Baseline agents for comparison: "dump the catalog into the prompt".
#
#   naive          — full catalog.json in the system prompt, tools to check a
#                    specific (type, provider, location) and book it. No policy
#                    enforcement: accuracy depends entirely on the LLM reading
#                    the catalog correctly.
#   naive_guarded  — same prompt, but booking re-validates policies server-side
#                    and returns the blocker. Isolates the value of guards alone
#                    from the value of retrieval + scoped context.
#

import json

from catalog.models import CATALOG_PATH
from catalog.tools import ToolSpec


def _check(e, a, s):
    t, p, l = a.get("appointment_type_id"), a.get("provider_id"), a.get("location_id")
    if t not in e.c.appointment_types or p not in e.c.providers or l not in e.c.locations:
        return {"status": "error", "message": "Unknown id."}
    from datetime import timedelta
    typ, prov = e.c.appointment_types[t], e.c.providers[p]
    slots = []
    for i in range(1, e.HORIZON_DAYS):
        slots += e._slots_for(typ, prov, l, e.today + timedelta(days=i))
        if len(slots) >= 3:
            break
    return {"status": "ok", "slots": [e.describe_slot(x) for x in slots[:3]]}


def _book_unguarded(e, a, s):
    from catalog.engine import Slot
    try:
        slot = Slot.parse(a["slot_id"])
    except Exception:
        return {"status": "error", "message": "Invalid slot_id."}
    s["booking"] = {"slot_id": a["slot_id"], **e.describe_slot(slot)}
    s["full_name"] = a.get("full_name")
    return {"status": "booked", "confirmation": "PX-000000", **e.describe_slot(slot)}


def _book_guarded(e, a, s):
    r = e.book(a["slot_id"], a.get("full_name", ""), a.get("date_of_birth", ""),
               bool(a.get("new_patient")), a.get("has_referral"))
    if r["status"] == "booked":
        s["booking"] = r
    return r


_BOOK_PROPS = {
    "slot_id": {"type": "string"},
    "full_name": {"type": "string"},
    "date_of_birth": {"type": "string"},
    "new_patient": {"type": "boolean"},
    "has_referral": {"type": "boolean"},
}

NAIVE_TOOLS = {
    "check_availability": ToolSpec(
        "check_availability",
        "Get the next open slots for a specific appointment type, provider and location.",
        {"appointment_type_id": {"type": "string"}, "provider_id": {"type": "string"},
         "location_id": {"type": "string"}},
        ["appointment_type_id", "provider_id", "location_id"], _check),
    "book_slot": ToolSpec("book_slot", "Book a slot returned by check_availability.",
                          _BOOK_PROPS, ["slot_id", "full_name", "date_of_birth", "new_patient"], _book_unguarded),
    "book_slot_guarded": ToolSpec("book_slot", "Book a slot returned by check_availability.",
                                  _BOOK_PROPS, ["slot_id", "full_name", "date_of_birth", "new_patient"], _book_guarded),
}


def naive_agent(guarded: bool = False, model: str = "gpt-4.1") -> dict:
    catalog = json.loads(CATALOG_PATH.read_text())
    persona = (
        "You are Ava, a scheduling assistant for Prosper Health. Replies are spoken aloud: one or two short "
        "sentences, no lists or ids. Help callers book appointments or answer questions using the clinic "
        "catalog below, honoring every policy in it. Collect the patient's name, date of birth, whether they "
        "are new, and a referral if required. Use check_availability then book_slot.\n\nCLINIC CATALOG (JSON):\n"
        + json.dumps(catalog)
    )
    return {
        "name": "Naive full-catalog agent" + (" (guarded)" if guarded else ""),
        "model": model,
        "persona": persona,
        "initial_node": "main",
        "nodes": [{
            "name": "main",
            "task_messages": [{"role": "developer", "content": "Greet the caller and help them."}],
            "tools": ["check_availability", "book_slot_guarded" if guarded else "book_slot"],
        }],
    }
