"""Multi-seed generalization: is 0.79265 a stable figure or one lucky draw?

eval/generalization.py builds a synthetic session set from a single seed. That
gives a point estimate, and a point estimate on 200 sessions carries sampling
noise we have not quantified. This runs the same construction across several
seeds and reports the spread.

Each draw is independent: different targets, same entropy stratification, same
40/40/15/5 scenario mix, same official evaluator.

Run: python -m eval.generalization_seeds
"""
import collections
import random
import statistics

from evaluator.local_evaluator import (evaluate, load_jsonl, catalog_index,
                                       materialize_hidden_fields, coarse_category)
from starter.agent import Agent

SEEDS = [20260830, 19, 4242]
N_SESSIONS = 200
SCENARIO_MIX = [("buying", 0.40), ("browsing", 0.40),
                ("intent_override", 0.15), ("boundary", 0.05)]

PUBLIC = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")

PROFILE = {
    "average_prior_rating": 4.0,
    "preference_tags": ["fit", "comfort"],
    "purchase_frequency": "3-4 prior purchases",
    "rating_style": "usually positive",
    "summary": "Prior purchases emphasize fit and comfort.",
}


def bucket(rarity):
    if rarity < 50:
        return "<50"
    if rarity < 500:
        return "50-500"
    if rarity < 5000:
        return "500-5k"
    return ">5k"


def target_rarity(agent, asin):
    stub = {"sample_id": "x", "scenario_type": "buying",
            "ground_truth": {"parent_asin": asin}}
    card, _ = materialize_hidden_fields(stub, PRODS)
    cons = set(card["hard_constraints"] + card["soft_preferences"])
    if len(cons) < 4:
        return None
    return min(agent._phrase_rarity(c) for c in cons)


def build_sessions(pool, public_buckets, seed):
    rng = random.Random(seed)
    total = sum(public_buckets.values())
    chosen = []
    for b, n in public_buckets.items():
        want = round(N_SESSIONS * n / total)
        available = list(pool[b])
        rng.shuffle(available)
        chosen.extend(available[:want])
    rng.shuffle(chosen)

    scenarios = []
    for name, share in SCENARIO_MIX:
        scenarios.extend([name] * round(len(chosen) * share))
    while len(scenarios) < len(chosen):
        scenarios.append("buying")
    rng.shuffle(scenarios)

    return [{"sample_id": f"synthetic_{seed}_{i:04d}",
             "scenario_type": scen,
             "difficulty_bucket": "synthetic",
             "category_bucket": "clothing",
             "ground_truth": {"parent_asin": asin},
             "user_profile": PROFILE}
            for i, (asin, scen) in enumerate(zip(chosen, scenarios))]


def main():
    agent = Agent()
    public_targets = {r["ground_truth"]["parent_asin"] for r in PUBLIC}

    print("profiling public targets...")
    public_buckets = collections.Counter()
    for r in PUBLIC:
        rarity = target_rarity(agent, r["ground_truth"]["parent_asin"])
        if rarity is not None:
            public_buckets[bucket(rarity)] += 1

    print("profiling candidate targets (this takes a minute)...")
    pool = collections.defaultdict(list)
    for asin in PRODS:
        if asin in public_targets or not coarse_category(CATS.get(asin, [])):
            continue
        rarity = target_rarity(agent, asin)
        if rarity is not None:
            pool[bucket(rarity)].append(asin)

    print("\nscoring public set...")
    pub = evaluate(Agent(), PUBLIC, IDS, CATS, PRODS)

    scores = []
    print()
    print(f"{'seed':>10}  {'HR':>6}  {'MRR':>6}  {'MTTC':>6}  {'score':>8}")
    print(f"{'public':>10}  {pub['hit_rate_at_10']:>6.3f}  {pub['mrr']:>6.3f}  "
          f"{pub['mttc']:>6.2f}  {pub['recommended_technical_score']:>8.5f}")

    for seed in SEEDS:
        sessions = build_sessions(pool, public_buckets, seed)
        r = evaluate(Agent(), sessions, IDS, CATS, PRODS)
        s = r["recommended_technical_score"]
        scores.append(s)
        print(f"{seed:>10}  {r['hit_rate_at_10']:>6.3f}  {r['mrr']:>6.3f}  "
              f"{r['mttc']:>6.2f}  {s:>8.5f}")

    print()
    print(f"  synthetic mean   {statistics.mean(scores):.5f}")
    print(f"  synthetic range  {min(scores):.5f} - {max(scores):.5f}")
    if len(scores) > 1:
        print(f"  std dev          {statistics.stdev(scores):.5f}")
    print(f"  gap from public  {statistics.mean(scores) - pub['recommended_technical_score']:+.5f}")


if __name__ == "__main__":
    main()