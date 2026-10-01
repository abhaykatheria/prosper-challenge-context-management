#
# Catalog models — typed, indexed view of catalog.json.
#
# Everything the scheduling agent knows about locations, providers and
# appointment types lives here, in memory, NOT in the LLM prompt. The LLM only
# ever sees small, query-shaped slices of it (see engine.py / tools.py).
#

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
ENRICHMENT_PATH = DATA_DIR / "enrichment.json"


@dataclass
class Location:
    id: str
    name: str
    address: str
    city: str
    phone: str
    hours: str
    capabilities: list

    def brief(self) -> dict:
        return {"location_id": self.id, "name": self.name, "address": f"{self.address}, {self.city}"}


@dataclass
class Provider:
    id: str
    name: str
    title: str
    specialty: str
    location_ids: list
    accepting_new_patients: bool
    languages: list
    appointment_type_ids: list

    @property
    def last_name(self) -> str:
        return self.name.split()[-1]

    @property
    def display(self) -> str:
        # "Dr. Maria Garcia" is ambiguous; "Dr. Maria Garcia, NP (Family Medicine)" is not.
        return f"{self.name}, {self.title} ({self.specialty})"


@dataclass
class AppointmentType:
    id: str
    name: str
    specialty: str
    duration_min: int
    requires_referral: bool
    new_patients_allowed: bool
    required_capability: Optional[str] = None
    # Enrichment (offline-authored, see data/enrichment.json)
    aliases: list = field(default_factory=list)
    description: str = ""


@dataclass
class Catalog:
    policies: list
    locations: dict        # id -> Location
    providers: dict        # id -> Provider
    appointment_types: dict  # id -> AppointmentType
    confusable_groups: list  # list[{"ids": [...], "question": str}]

    # ---- derived indexes (built once; O(catalog) memory) -------------------
    def __post_init__(self):
        # appointment type -> providers who can perform it
        self.providers_by_type: dict = {t: [] for t in self.appointment_types}
        for p in self.providers.values():
            for t in p.appointment_type_ids:
                self.providers_by_type.setdefault(t, []).append(p.id)
        self.group_by_type: dict = {}
        for g in self.confusable_groups:
            for t in g["ids"]:
                self.group_by_type.setdefault(t, []).append(g)

    @classmethod
    def load(
        cls,
        path: Union[str, Path] = CATALOG_PATH,
        enrichment_path: Union[str, Path, None] = ENRICHMENT_PATH,
    ) -> "Catalog":
        raw = json.loads(Path(path).read_text())
        enrich = {"appointment_types": {}, "confusable_groups": []}
        if enrichment_path and Path(enrichment_path).exists():
            enrich = json.loads(Path(enrichment_path).read_text())

        types = {}
        for a in raw["appointment_types"]:
            e = enrich["appointment_types"].get(a["id"], {})
            types[a["id"]] = AppointmentType(
                id=a["id"],
                name=a["name"],
                specialty=a["specialty"],
                duration_min=a["duration_min"],
                requires_referral=a["requires_referral"],
                new_patients_allowed=a["new_patients_allowed"],
                required_capability=a.get("required_capability"),
                aliases=e.get("aliases", []),
                description=e.get("description", ""),
            )
        return cls(
            policies=raw.get("policies", []),
            locations={l["id"]: Location(**l) for l in raw["locations"]},
            providers={p["id"]: Provider(**p) for p in raw["providers"]},
            appointment_types=types,
            confusable_groups=enrich.get("confusable_groups", []),
        )
