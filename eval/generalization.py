"""Generalization test: does our score hold on targets we never tuned against?

Every figure in RESULTS.md comes from the same 200 public sessions we developed
on. The private 800 use disjoint users and disjoint targets, so the obvious
question is whether 0.837 reflects a general capability or a fit to those 200.

This builds synthetic sessions from catalog products that never appear as public
targets, using the evaluator's own generation path — `materialize_hidden_fields`
derives the intent card from the target product, so a session record needs only a
target ASIN and a scenario type.

Two controls make the comparison fair:

  Scenario mix matches the official 40/40/15/5 split.

  Constraint entropy is stratified to match the public set. The official targets
  were drawn from a curated pipeline; sampling the catalog at random would give a
  population with different discriminability, and any score gap would be
  unattributable. Matching the entropy distribution isolates overfitting as the
  variable under test.

The official evaluator is used unmodified. Only the session file is synthetic,
and figures are reported separately from official scores.

Run: python -m eval.generalization
"""
import collections
import json
import random

from evaluator.local_evaluator import (evaluate, load_jsonl, catalog_index,
                                       materialize_hidden_fields, coarse_category)
from starter.agent import Agent

SEED = 20260830          # fixed so the synthetic set is reproducible
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
    """Rarity of the most distinctive constraint this target can disclose."""
    stub = {"sample_id": "x", "scenario_type": "buying",
            "ground_truth": {"parent_asin": asin}}
    card, _ = materialize_hidden_fields(stub, PRODS)
    cons = set(card["hard_constraints"] + card["soft_preferences"])
    if len(cons) < 4:
        return None
    return min(agent._phrase_rarity(c) for c in cons)


def main():
    rng = random.Random(SEED)
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
        if asin in public_targets:
            continue
        if not coarse_category(CATS.get(asin, [])):
            continue
        rarity = target_rarity(agent, asin)
        if rarity is not None:
            pool[bucket(rarity)].append(asin)

    print("\nentropy distribution")
    print(f"{'bucket':>8}  {'public':>7}  {'available':>10}")
    for b in ("<50", "50-500", "500-5k", ">5k"):
        print(f"{b:>8}  {public_buckets[b]:>7}  {len(pool[b]):>10}")

    # sample synthetic targets matching the public entropy profile
    total_public = sum(public_buckets.values())
    chosen = []
    for b, n in public_buckets.items():
        want = round(N_SESSIONS * n / total_public)
        available = pool[b]
        rng.shuffle(available)
        if len(available) < want:
            print(f"\nWARNING: only {len(available)} targets available in {b}, "
                  f"wanted {want}")
        chosen.extend(available[:want])
    rng.shuffle(chosen)

    scenarios = []
    for name, share in SCENARIO_MIX:
        scenarios.extend([name] * round(len(chosen) * share))
    while len(scenarios) < len(chosen):
        scenarios.append("buying")
    rng.shuffle(scenarios)

    sessions = [
        {"sample_id": f"synthetic_{i:04d}",
         "scenario_type": scen,
         "difficulty_bucket": "synthetic",
         "category_bucket": "clothing",
         "ground_truth": {"parent_asin": asin},
         "user_profile": PROFILE}
        for i, (asin, scen) in enumerate(zip(chosen, scenarios))
    ]

    with open("data/synthetic_set.jsonl", "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
    print(f"\nwrote data/synthetic_set.jsonl ({len(sessions)} sessions)")

    print("\nevaluating on public set...")
    pub = evaluate(Agent(), PUBLIC, IDS, CATS, PRODS)
    print("evaluating on synthetic set...")
    syn = evaluate(Agent(), sessions, IDS, CATS, PRODS)

    print()
    print(f"{'set':>10}  {'n':>4}  {'HR':>6}  {'MRR':>6}  {'MTTC':>6}  {'score':>8}")
    for name, r in (("public", pub), ("synthetic", syn)):
        print(f"{name:>10}  {r['sample_count']:>4}  {r['hit_rate_at_10']:>6.3f}  "
              f"{r['mrr']:>6.3f}  {r['mttc']:>6.2f}  "
              f"{r['recommended_technical_score']:>8.5f}")

    gap = (syn["recommended_technical_score"]
           - pub["recommended_technical_score"])
    print(f"\ndelta: {gap:+.5f}")
    if abs(gap) < 0.02:
        print("Within noise of the public score — no evidence of overfitting.")
    elif gap < 0:
        print("Synthetic score is materially lower. Either the components are "
              "fitted to the public set,\nor the synthetic population differs in "
              "a way the entropy stratification does not capture.")
    else:
        print("Synthetic score is higher — the synthetic population is easier "
              "than the public one\ndespite matched entropy.")


if __name__ == "__main__":
    main()