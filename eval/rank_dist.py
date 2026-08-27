# save as check_ranks.py in the project root
import json, collections
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent

sessions = load_jsonl("data/public_set.jsonl")
ids, cats, prods = catalog_index("data/catalog.jsonl")
r = evaluate(Agent(), sessions, ids, cats, prods)

ranks = collections.Counter()
misses = []
for s in r["sessions"]:
    rank = s.get("best_rank")
    if rank:
        ranks[rank] += 1
    else:
        misses.append((s.get("sample_id"), s.get("scenario_type")))

print("rank distribution:", dict(sorted(ranks.items())))
print("hits:", sum(ranks.values()), "misses:", len(misses))
print("miss scenarios:", collections.Counter(m[1] for m in misses))