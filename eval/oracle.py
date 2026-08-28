"""Upper bound: an agent given all four constraints immediately.

Not a submission — a measurement of how much headroom exists. If the oracle
also plateaus, the remaining misses are a property of the benchmark rather than
of our retrieval.

Session IDs are random UUIDs, so we cannot key on them. The evaluator processes
samples in file order and calls reset() exactly once per session, so we consume
the cheat sheet in the same order.
"""
from evaluator.local_evaluator import (evaluate, load_jsonl, catalog_index,
                                       materialize_hidden_fields, coarse_category)
from starter.agent import Agent

SESSIONS = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")

CHEAT = []
for smp in SESSIONS:
    card, _ = materialize_hidden_fields(smp, PRODS)
    tgt = smp["ground_truth"]["parent_asin"]
    CHEAT.append((card["hard_constraints"] + card["soft_preferences"],
                  coarse_category(CATS.get(tgt, []))))


class OracleAgent(Agent):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._i = 0

    def reset(self, session_id, user_profile):
        super().reset(session_id, user_profile)
        phrases, category = CHEAT[self._i]
        self._i += 1
        self.s[session_id]["phrases"] = list(phrases)
        self.s[session_id]["category"] = category

    def respond(self, session_id, user_message, turn, top_k):
        st = self.s[session_id]
        st["msgs"].append(user_message)
        return {"message": "", "ask_attribute": None,
                "recommendations": self._recommend(st, top_k),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


def main():
    print("running ours...")
    ours = evaluate(Agent(), SESSIONS, IDS, CATS, PRODS)
    print("running oracle...")
    oracle = evaluate(OracleAgent(), SESSIONS, IDS, CATS, PRODS)

    print()
    for name, r in (("ours", ours), ("oracle", oracle)):
        print(f"  {name:8} HR {r['hit_rate_at_10']:.3f}  MRR {r['mrr']:.3f}  "
              f"MTTC {r['mttc']:.2f}  score {r['recommended_technical_score']:.5f}")

    gap = ours["recommended_technical_score"] / oracle["recommended_technical_score"]
    print(f"\n  we are at {gap:.1%} of the achievable ceiling")

    print("\n  per-scenario oracle HR:")
    for s, m in oracle["scenario_metrics"].items():
        print(f"    {s:16} {m['hit_rate_at_10']:.3f}")


if __name__ == "__main__":
    main()