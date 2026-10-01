#
# Matching benchmark: fuzzy matcher vs Jev (options only) vs Jev (full catalog).
#
#   python -m sim.match_bench                # all resolvers
#   python -m sim.match_bench --only fuzzy
#
# Each labelled phrase is scored by what the agent would DO with the result:
#   clear      -> correct if it resolves to a gold type; "asked" if it returns an
#                 ambiguity that contains the gold (safe, costs one extra question);
#                 wrong otherwise (wrong type, or claims not offered / no match).
#   ambiguous  -> correct if it asks (>=2 gold candidates); "overconfident" if it
#                 commits to one plausible option without asking; wrong otherwise.
#   not_offered-> correct only if it says not offered.
#   vague      -> correct if it doesn't commit (no_match or ambiguous).
#

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from catalog import Catalog, CatalogEngine  # noqa: E402
from catalog.jev import JevResolver, cost_usd  # noqa: E402

HERE = Path(__file__).resolve().parent


def grade(item: dict, out: dict) -> str:
    status = out["status"]
    ids = [c["appointment_type_id"] for c in out.get("candidates", [])]
    gold = set(item["gold"])
    kind = item["kind"]
    if kind == "clear":
        if status in ("match", "existing_patients_only") and ids and ids[0] in gold:
            return "correct"
        if status == "ambiguous" and gold & set(ids):
            return "asked"
        return "wrong"
    if kind == "ambiguous":
        if status == "ambiguous" and len(gold & set(ids)) >= 2:
            return "correct"
        if status in ("match", "existing_patients_only") and ids and ids[0] in gold:
            return "overconfident"
        return "wrong"
    if kind == "not_offered":
        return "correct" if status == "not_offered" else "wrong"
    if kind == "vague":
        return "correct" if status in ("no_match", "ambiguous") else "wrong"
    raise ValueError(kind)


async def run(resolver_name: str, items: list, engine: CatalogEngine, concurrency: int) -> list:
    if resolver_name == "fuzzy":
        out = []
        for it in items:
            t0 = time.perf_counter()
            r = engine.match_appointment_types(it["text"])
            out.append({**it, "result": r, "latency": time.perf_counter() - t0, "tokens": 0})
        return out

    if resolver_name == "cascade":
        from catalog.tools import _find_appointment_types
        sem = asyncio.Semaphore(concurrency)

        async def one_c(it):
            async with sem:
                state = {"_resolver": "cascade"}
                t0 = time.perf_counter()
                r = await _find_appointment_types(engine, {"request": it["text"]}, state)
                dt = time.perf_counter() - t0
                return {**it, "result": r, "latency": dt,
                        "tokens": 21805 if r.get("resolver") == "jev" else 0}

        return await asyncio.gather(*[one_c(it) for it in items])

    context = "full" if resolver_name == "jev_full" else "options"
    res = JevResolver(engine, context=context)
    sem = asyncio.Semaphore(concurrency)

    async def one(it):
        async with sem:
            t0 = time.perf_counter()
            r = await res.match_appointment_types(it["text"])
            dt = time.perf_counter() - t0
            return {**it, "result": r, "latency": dt, "tokens": res.client.calls[-1]["input_tokens"]}

    return await asyncio.gather(*[one(it) for it in items])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="fuzzy,jev_options,jev_full,cascade")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    items = json.loads((HERE / "match_bench.json").read_text())
    engine = CatalogEngine(Catalog.load(), today=date(2026, 9, 28))
    summary, all_rows = {}, {}

    for name in args.only.split(","):
        rows = await run(name, items, engine, args.concurrency)
        for r in rows:
            r["grade"] = grade(r, r["result"])
        all_rows[name] = rows
        by_group = defaultdict(Counter)
        for r in rows:
            by_group[r["group"]][r["grade"]] += 1
        lat = [r["latency"] for r in rows]
        tokens = sum(r["tokens"] for r in rows)
        g = Counter(r["grade"] for r in rows)
        summary[name] = {
            "correct": g["correct"], "asked": g["asked"], "overconfident": g["overconfident"],
            "wrong": g["wrong"], "n": len(rows),
            "by_group": {k: dict(v) for k, v in by_group.items()},
            "p50_ms": round(statistics.median(lat) * 1000, 1),
            "p95_ms": round(sorted(lat)[int(len(lat) * 0.95) - 1] * 1000, 1),
            "avg_input_tokens": round(tokens / len(rows)),
            "cost_per_lookup": cost_usd(tokens) / len(rows),
        }
        if args.verbose:
            for r in rows:
                if r["grade"] != "correct":
                    names = [c["name"] for c in r["result"].get("candidates", [])]
                    print(f"  {name:12} {r['grade']:13} {r['text'][:60]:60} -> {r['result']['status']} {names}")

    groups = ["direct", "indirect", "stt_garbled", "ambiguous", "not_offered", "vague"]
    print("\n| resolver | correct | asked (safe) | overconfident | wrong | " + " | ".join(groups) + " | p50 | p95 | input tok | $/lookup |")
    print("|---" * (10 + len(groups)) + "|")
    for name, s in summary.items():
        cells = []
        for gname in groups:
            c = s["by_group"].get(gname, {})
            n = sum(c.values())
            cells.append(f"{c.get('correct', 0)}/{n}")
        print(f"| {name} | {s['correct']}/{s['n']} | {s['asked']} | {s['overconfident']} | {s['wrong']} | "
              + " | ".join(cells)
              + f" | {s['p50_ms']}ms | {s['p95_ms']}ms | {s['avg_input_tokens']} | ${s['cost_per_lookup']:.5f} |")

    out = HERE / "results" / "match_bench.json"
    out.write_text(json.dumps({"summary": summary, "rows": {k: [{kk: vv for kk, vv in r.items()} for r in v]
                                                            for k, v in all_rows.items()}}, indent=1, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
