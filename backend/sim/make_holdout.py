#
# One-off generator for the held-out matching benchmark (sim/match_bench_holdout.json).
#
# Written by a *different* model (gpt-4.1) from the catalog's own descriptions, so the
# phrasings don't share my alias vocabulary, and labels come for free from what each
# item was generated for. Generated once and frozen BEFORE tuning the resolver; the
# hybrid resolver's thresholds were never fit to it.
#
#   python -m sim.make_holdout   (refuses to overwrite an existing file)
#

import asyncio
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from catalog import Catalog  # noqa: E402

OUT = Path(__file__).resolve().parent / "match_bench_holdout.json"
STYLES = {
    "direct": "names what they want plainly, in everyday words (not the clinic's official name)",
    "indirect": "describes their situation or symptoms WITHOUT naming the service, test or specialty",
    "stt_garbled": "is a speech-to-text transcript where one or two key words were misrecognised "
                   "phonetically (e.g. a medical word split or replaced by a sound-alike)",
}


async def ask(client, prompt: str) -> str:
    r = await client.chat.completions.create(
        model="gpt-4.1", temperature=0.9, seed=7,
        messages=[{"role": "user", "content": prompt + "\nReply with only the caller's words, no quotes."}])
    return r.choices[0].message.content.strip().strip('"')


async def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; the held-out set is frozen. Delete it deliberately to regenerate.")
    rng = random.Random(2026)
    c = Catalog.load()
    offered = [t for t in c.appointment_types.values() if c.providers_by_type.get(t.id)]
    not_offered = [t for t in c.appointment_types.values() if not c.providers_by_type.get(t.id)]
    client = AsyncOpenAI()
    jobs = []

    for i, t in enumerate(rng.sample(offered, 42)):
        style = list(STYLES)[i % 3]
        prompt = (f"A patient phones a clinic and needs this appointment:\n{t.name}: {t.description}\n"
                  f"Write the one sentence they say to the receptionist. It {STYLES[style]}.")
        jobs.append((prompt, {"group": style, "kind": "clear", "gold": [t.id]}))

    groups = [g for g in c.confusable_groups
              if sum(bool(c.providers_by_type.get(i)) for i in g["ids"]) >= 2]
    for g in rng.sample(groups, 10):
        opts = "\n".join(f"- {c.appointment_types[i].name}: {c.appointment_types[i].description}" for i in g["ids"])
        prompt = ("A patient phones a clinic wanting ONE of these appointments, but says it vaguely, naming "
                  "only the general kind of thing, so the receptionist can't tell which one they mean:\n"
                  f"{opts}\nWrite the one sentence they say. It must not contain any detail that would "
                  "single out one of the options.")
        jobs.append((prompt, {"group": "ambiguous", "kind": "ambiguous", "gold": list(g["ids"])}))

    for t in rng.sample(not_offered, 5):
        prompt = (f"A patient phones a clinic asking for: {t.name} ({t.description}). "
                  "Write the one sentence they say, in everyday words.")
        jobs.append((prompt, {"group": "not_offered", "kind": "not_offered", "gold": []}))

    for _ in range(3):
        jobs.append(("A patient phones a clinic to make an appointment but doesn't say what it's for. "
                     "Write the one sentence they say.", {"group": "vague", "kind": "vague", "gold": []}))

    texts = await asyncio.gather(*[ask(client, p) for p, _ in jobs])
    items = [{**meta, "text": text} for (_, meta), text in zip(jobs, texts)]
    OUT.write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} items to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
