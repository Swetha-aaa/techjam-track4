"""Does the simulator ever emit a negative constraint?"""
import re
from evaluator.local_evaluator import load_jsonl, catalog_index
from starter.agent import Agent

NEG = re.compile(r"\bnot\b|\bno\b|\bwithout\b|\bavoid\b|\bexclude\b|"
                 r"\bdon't want\b|\brather not\b|\bexcept\b", re.I)

sessions = load_jsonl("data/public_set.jsonl")
ids, cats, prods = catalog_index("data/catalog.jsonl")
agent = Agent()

hits, total = [], 0
for s in sessions:
    agent.reset(s["sample_id"], None)
    for turn in range(1, 11):
        try:
            msg = agent  # placeholder; see note
        except Exception:
            break
    total += 1

print("run the evaluator loop instead — see note")