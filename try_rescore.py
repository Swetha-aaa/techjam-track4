"""Measure the exact-substring rescorer across candidate pool sizes.

Pool 0 is the current system with the rescorer off — the control row. Any pool
size that does not beat it is a rejection, and rejections go in RESULTS.md the
same as acceptances.

Run from the repo root:  python try_rescore.py
"""
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent

SESSIONS = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")

print(f"{'pool':>6}  {'HR@10':>6}  {'MRR':>6}  {'MTTC':>5}  {'score':>8}")

baseline = None
for pool in (0, 20, 50, 100, 200):
    cfg = {} if pool == 0 else {"exact_rescore": True, "rescore_pool": pool}
    r = evaluate(Agent(config=cfg), SESSIONS, IDS, CATS, PRODS)
    score = r["recommended_technical_score"]
    if pool == 0:
        baseline = score
        delta = ""
    else:
        delta = f"   {score - baseline:+.5f}"
    label = "off" if pool == 0 else str(pool)
    print(f"{label:>6}  {r['hit_rate_at_10']:6.3f}  {r['mrr']:6.3f}  "
          f"{r['mttc']:5.2f}  {score:8.5f}{delta}")