"""How discriminative is each session's constraint set, and does that explain our misses?"""
import json, collections
from evaluator.local_evaluator import (evaluate, load_jsonl, catalog_index,
                                       materialize_hidden_fields)
from starter.agent import Agent, terms

sessions = load_jsonl("data/public_set.jsonl")
ids, cats, prods = catalog_index("data/catalog.jsonl")

agent = Agent()
res = evaluate(agent, sessions, ids, cats, prods)

# rank achieved, by sample_id
rank_by_id = {s.get("sample_id"): s.get("best_rank") for s in res["sessions"]}

rows = []
for smp in sessions:
    sid = smp["sample_id"]
    card, _ = materialize_hidden_fields(smp, prods)
    constraints = card["hard_constraints"] + card["soft_preferences"]
    # rarest token across the whole constraint set
    rarity = min(
        (min((agent.df.get(t, 1) for t in terms(c)), default=agent.total_docs)
         for c in constraints),
        default=agent.total_docs)
    rows.append((sid, smp["scenario_type"], rarity, rank_by_id.get(sid)))

def bucket(r):
    if r < 50:    return "very rare (<50 docs)"
    if r < 500:   return "rare (50-500)"
    if r < 5000:  return "common (500-5k)"
    return "very common (>5k)"

agg = collections.defaultdict(lambda: {"n": 0, "hits": 0, "rank1": 0})
for sid, scen, rarity, rank in rows:
    b = agg[bucket(rarity)]
    b["n"] += 1
    if rank:
        b["hits"] += 1
        if rank == 1:
            b["rank1"] += 1

print(f"{'bucket':24} {'n':>4} {'HR':>7} {'rank1':>7}")
for b in ["very rare (<50 docs)", "rare (50-500)", "common (500-5k)", "very common (>5k)"]:
    d = agg.get(b)
    if not d or not d["n"]:
        continue
    print(f"{b:24} {d['n']:4} {d['hits']/d['n']:7.3f} {d['rank1']/d['n']:7.3f}")

print("\nmisses by rarity bucket:")
print(collections.Counter(bucket(r) for _, _, r, rank in rows if not rank))