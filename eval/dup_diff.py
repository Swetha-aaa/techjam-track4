"""Which sessions does clause duplication actually affect, and are they the long ones?"""
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent

SESSIONS = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")


def main():
    off = evaluate(Agent(config={"clause_duplication": 0}), SESSIONS, IDS, CATS, PRODS)
    on = evaluate(Agent(config={"clause_duplication": 6}), SESSIONS, IDS, CATS, PRODS)

    changed, unchanged = [], []
    for a, b in zip(off["sessions"], on["sessions"]):
        ra, rb = a.get("best_rank"), b.get("best_rank")
        ta, tb = a.get("first_hit_turn"), b.get("first_hit_turn")
        (changed if (ra, ta) != (rb, tb) else unchanged).append((a, b, ta, tb))

    print(f"changed: {len(changed)}   unchanged: {len(unchanged)}\n")

    def mean_turn(rows, idx):
        vals = [r[idx] or 11 for r in rows]
        return sum(vals) / len(vals) if vals else 0

    print(f"mean first-hit turn, sessions that changed:   {mean_turn(changed, 2):.2f} "
          f"-> {mean_turn(changed, 3):.2f}")
    print(f"mean first-hit turn, sessions that did not:   {mean_turn(unchanged, 2):.2f}")

    print("\nchanged sessions (turn off -> on, rank off -> on):")
    for a, b, ta, tb in changed[:25]:
        print(f"  turn {str(ta):>4} -> {str(tb):<4}   "
              f"rank {str(a.get('best_rank')):>4} -> {b.get('best_rank')}")


if __name__ == "__main__":
    main()