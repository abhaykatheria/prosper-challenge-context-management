#
# STT vocabulary biasing from the catalog.
#
# The same catalog that drives retrieval also tells the speech recogniser which
# rare words to expect: provider surnames, confusable location names, and
# medical terms that generic STT mangles ("Holter", "DEXA", "spirometry").
# Sent as ElevenLabs `keyterms`, which the API caps at 50 (verified: it rejects
# more), so terms are prioritised: confusable locations, then surnames, then the
# medical words generic STT gets wrong most. Common words need no biasing.
#

from .models import Catalog

MEDICAL_TERMS = [
    "MRI", "EKG", "EEG", "Holter", "echocardiogram", "DEXA", "spirometry", "audiogram",
    "colonoscopy", "endoscopy", "Pap smear", "OB/GYN", "ENT", "dermatology", "pulmonology",
    "gastroenterology", "endocrinology",
]


def stt_keyterms(catalog: Catalog, limit: int = 50) -> list:
    terms = ["Prosper Health"]
    terms += [l.name.replace(" Health Center", "").replace(" Family Clinic", "")
              .replace(" Specialty Center", "").replace(" Medical Group", "")
              .replace(" Community Clinic", "").replace(" Care Center", "")
              for l in catalog.locations.values()]
    # Surnames are what callers say ("Dr. Nguyen"); full names add little at 50 terms.
    terms += sorted({p.last_name for p in catalog.providers.values()})
    terms += MEDICAL_TERMS
    seen, out = set(), []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:limit]
