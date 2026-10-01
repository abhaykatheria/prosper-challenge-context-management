#
# Eval harness — simulated callers vs. agent variants, scored deterministically.
#
#   python -m sim.eval                      # all agents, all scenarios
#   python -m sim.eval --agents ours --only mri_knee_soonest --verbose
#
# A small LLM plays the patient (persona + private facts). The agent under test
# runs through the same AgentBuilder + TextFlowRunner as the product. Scoring is
# NOT an LLM judge: the final booking is checked against the scenario's expected
# type/provider/location AND re-validated against every catalog policy using the
# patient's true facts, so a "successful" booking that breaks a rule still fails.
#

import argparse
import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from agent_builder import AgentBuilder  # noqa: E402
from catalog import Catalog, CatalogEngine  # noqa: E402
from catalog.engine import Slot  # noqa: E402
from catalog.tools import TOOLS  # noqa: E402
from sim.baseline import NAIVE_TOOLS, naive_agent  # noqa: E402
from sim.textflow import TextFlowRunner  # noqa: E402

HERE = Path(__file__).resolve().parent
TODAY = date(2026, 9, 28)
CATALOG = Catalog.load()

SIM_PROMPT = """You are role-playing a patient phoning a clinic's scheduling line. Stay in character.
Goal: {persona}
Your facts (share only when asked): name {name}; date of birth {dob}; {status}; {referral}.
Today is {today}. Talk like a real caller: short, natural sentences, one thing at a time. Don't volunteer every fact at once.
When your goal is done (booked and confirmed, or clearly impossible), or the assistant says goodbye, reply with exactly: [END]"""


def agent_config(kind: str, model: str) -> dict:
    if kind in ("ours", "ours_jev", "ours_cascade"):
        cfg = json.loads((HERE.parent / "agents" / "scheduler.json").read_text())
        cfg["model"] = model
        cfg["resolver"] = {"ours": "fuzzy", "ours_jev": "jev", "ours_cascade": "cascade"}[kind]
        return cfg
    return naive_agent(guarded=(kind == "naive_guarded"), model=model)


def score(sc: dict, state: dict, transcript: list) -> dict:
    exp, pt = sc["expect"], sc["patient"]
    booking = state.get("booking")
    reasons = []
    if not exp["book"]:
        if booking:
            reasons.append("booked but should not have")
        said = " ".join(t["text"] for t in transcript if t["role"] == "assistant")
        for m in exp.get("mentions", []):
            if m.lower() not in said.lower():
                reasons.append(f"did not mention {m}")
        for m in exp.get("not_mentions", []):
            if m.lower() in said.lower():
                reasons.append(f"wrongly mentioned {m}")
        return {"pass": not reasons, "reasons": reasons}

    if not booking:
        return {"pass": False, "reasons": ["no booking made"]}
    s = Slot.parse(booking["slot_id"])
    if s.appointment_type_id not in exp["type_ids"]:
        reasons.append(f"wrong type {CATALOG.appointment_types[s.appointment_type_id].name}")
    if "provider_ids" in exp and s.provider_id not in exp["provider_ids"]:
        reasons.append(f"wrong provider {CATALOG.providers[s.provider_id].display}")
    if "location_ids" in exp and s.location_id not in exp["location_ids"]:
        reasons.append(f"wrong location {CATALOG.locations[s.location_id].name}")
    if "provider_language" in exp and exp["provider_language"] not in CATALOG.providers[s.provider_id].languages:
        reasons.append(f"provider doesn't speak {exp['provider_language']}")
    if exp.get("afternoon") and s.start.hour < 12:
        reasons.append("not an afternoon slot")
    # Policy re-validation with the patient's TRUE facts.
    checker = CatalogEngine(CATALOG, today=TODAY)
    _, blockers = checker.solve(s.appointment_type_id, s.provider_id, s.location_id,
                                pt["new_patient"], pt["has_referral"])
    reasons += [f"POLICY VIOLATION: {b.message}" for b in blockers]
    return {"pass": not reasons, "reasons": reasons,
            "booked": f"{CATALOG.appointment_types[s.appointment_type_id].name} / "
                      f"{CATALOG.providers[s.provider_id].display} / {CATALOG.locations[s.location_id].name} / {s.start}"}


async def run_one(client, kind: str, sc: dict, model: str, sim_model: str, max_turns: int) -> dict:
    engine = CatalogEngine(CATALOG, today=TODAY)
    builder = AgentBuilder.from_dict(agent_config(kind, model), tools={**TOOLS, **NAIVE_TOOLS},
                                     engine_factory=lambda: engine, today=TODAY)
    runner = TextFlowRunner(builder, client=client, model=model)
    pt = sc["patient"]
    sim_sys = SIM_PROMPT.format(
        today=f"{TODAY:%A, %B} {TODAY.day}, {TODAY.year}",
        persona=sc["persona"], name=pt["name"], dob=pt["dob"],
        status="you are a NEW patient here" if pt["new_patient"] else "you have been seen here before",
        referral="you have a referral" if pt["has_referral"] else "you do not have a referral")
    sim_msgs = [{"role": "system", "content": sim_sys}]

    t0 = time.time()
    agent_text = await runner.start()
    for _ in range(max_turns):
        sim_msgs.append({"role": "user", "content": agent_text})
        r = await client.chat.completions.create(model=sim_model, messages=sim_msgs, temperature=0.3)
        caller = (r.choices[0].message.content or "").strip()
        sim_msgs.append({"role": "assistant", "content": caller})
        if "[END]" in caller or runner.ended:
            break
        agent_text = await runner.send(caller)
    result = score(sc, runner.state, runner.transcript)
    u = runner.usage
    from catalog import jev as _jev
    jr = _jev._resolvers.pop((id(engine), "full"), None)
    jev_calls = jr.client.calls if jr else []
    jev_cost = _jev.cost_usd(sum(c["input_tokens"] for c in jev_calls))
    return {
        "agent": kind, "scenario": sc["id"], **result,
        "llm_calls": u.calls, "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
        "avg_prompt_tokens": round(u.prompt_tokens / max(u.calls, 1)),
        "max_prompt_tokens": u.max_prompt_tokens,
        "cost_usd": round(u.cost(model) + jev_cost, 4),
        "cost_uncached_usd": round(u.cost(model, use_cache=False) + jev_cost, 4),
        "jev_calls": len(jev_calls),
        "jev_avg_latency_ms": round(1000 * sum(c["latency"] for c in jev_calls) / len(jev_calls)) if jev_calls else 0,
        "jev_cost_usd": round(jev_cost, 5),
        "cached_tokens": u.cached_tokens,
        "turns": sum(1 for t in runner.transcript if t["role"] == "user"),
        "seconds": round(time.time() - t0, 1),
        "per_call": u.per_call,
        "transcript": runner.transcript,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="ours,naive,naive_guarded")
    ap.add_argument("--only", default="", help="comma-separated scenario ids")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--sim-model", default="gpt-4o-mini")
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    scenarios = json.loads((HERE / "scenarios.json").read_text())
    if args.only:
        keep = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in keep]
    client = AsyncOpenAI(max_retries=20)  # 429s back off using retry-after
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(kind, sc):
        async with sem:
            try:
                return await run_one(client, kind, sc, args.model, args.sim_model, args.max_turns)
            except Exception as ex:  # keep the batch going; report the failure
                return {"agent": kind, "scenario": sc["id"], "pass": False, "reasons": [f"crash: {ex!r}"],
                        "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "avg_prompt_tokens": 0,
                        "max_prompt_tokens": 0, "cost_usd": 0, "cost_uncached_usd": 0, "cached_tokens": 0, "turns": 0, "seconds": 0, "transcript": []}

    jobs = [guarded(k, sc) for k in args.agents.split(",") for sc in scenarios for _ in range(args.repeat)]
    results = list(await asyncio.gather(*jobs))

    # Rate-limit crashes say nothing about the agent: retry them one at a time.
    by_id = {sc["id"]: sc for sc in scenarios}
    for attempt in range(3):
        crashed = [i for i, r in enumerate(results) if any(x.startswith("crash") for x in r["reasons"])]
        if not crashed:
            break
        print(f"Retrying {len(crashed)} crashed runs serially (attempt {attempt + 1})...")
        for i in crashed:
            await asyncio.sleep(10)
            results[i] = await guarded(results[i]["agent"], by_id[results[i]["scenario"]])

    for r in results:
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"{r['agent']:14} {r['scenario']:26} {mark}  calls={r['llm_calls']:2} jev={r.get('jev_calls', 0)} "
              f"avg_prompt={r['avg_prompt_tokens']:6} cost=${r['cost_usd']:.4f}  {'; '.join(r['reasons'])}")
        if args.verbose:
            for t in r["transcript"]:
                print(f"      {t['role'][:5]:5} {t['text']}")

    print("\n| agent | pass rate | policy violations | avg prompt tok / call | max prompt tok | avg tok / conversation | cost / conv (cached) | cost / conv (uncached) | avg LLM calls |")
    print("|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for kind in args.agents.split(","):
        rs = [r for r in results if r["agent"] == kind]
        n = len(rs)
        calls = sum(r["llm_calls"] for r in rs)
        row = {
            "pass_rate": sum(r["pass"] for r in rs) / n,
            "avg_prompt_per_call": sum(r["prompt_tokens"] for r in rs) / max(calls, 1),
            "max_prompt": max(r["max_prompt_tokens"] for r in rs),
            "avg_tokens_per_conv": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs) / n,
            "avg_cost": sum(r["cost_usd"] for r in rs) / n,
            "avg_cost_uncached": sum(r["cost_uncached_usd"] for r in rs) / n,
            "policy_violations": sum(any("POLICY" in x for x in r["reasons"]) for r in rs),
            "crashes": sum(any(x.startswith("crash") for x in r["reasons"]) for r in rs),
            "avg_calls": calls / n,
        }
        summary[kind] = row
        print(f"| {kind} | {row['pass_rate']:.0%} ({sum(r['pass'] for r in rs)}/{n}) | {row['policy_violations']} | "
              f"{row['avg_prompt_per_call']:.0f} | {row['max_prompt']} | {row['avg_tokens_per_conv']:.0f} | "
              f"${row['avg_cost']:.4f} | ${row['avg_cost_uncached']:.4f} | {row['avg_calls']:.1f} |"
              + (f" ({row['crashes']} crashed)" if row["crashes"] else ""))

    out = HERE / "results" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "summary": summary, "results": results}, indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
