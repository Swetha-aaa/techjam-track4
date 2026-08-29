"""Diagnostic: how many constraint phrases does a session hold, and at which turn?

This is what makes `clause_duplication` work. Clauses are ordered by rarity, so
duplicating the first N boosts the N most distinctive phrases and leaves the
common tail single-weighted. That is only differential weighting when a call
holds more than N phrases — uniform duplication is provably inert, since FTS5's
BM25 sums per-clause contributions and doubling every clause scales all scores
equally (verified in eval/dup_probe.py).

So the question this answers is: at which turns do calls exceed 6 phrases?

Run: python -m eval.phrase_counts
"""
import collections

from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent

SESSIONS = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")

BOOST_N = 6      # the clause_duplication value we are explaining


def main():
    agent = Agent()
    records = []
    original = agent._search

    def spy(phrases, msgs, k, category=None):
        # len(msgs) is the turn number: respond() appends before searching
        records.append((len(phrases), len(msgs)))
        return original(phrases, msgs, k, category)

    agent._search = spy
    evaluate(agent, SESSIONS, IDS, CATS, PRODS)

    counts = [n for n, _ in records]
    dist = collections.Counter(counts)

    print("Phrases held at query time, all turns of all 200 sessions\n")
    print(f"{'phrases':>8}  {'calls':>6}")
    for n in sorted(dist):
        print(f"{n:>8}  {dist[n]:>6}")
    print(f"\nmax {max(counts)}   mean {sum(counts) / len(counts):.2f}")

    over = sum(1 for n in counts if n > BOOST_N)
    print(f"calls holding more than {BOOST_N} phrases: {over} "
          f"({over / len(counts):.1%})")
    print("These are the only calls where clause duplication is asymmetric, "
          "and therefore the only ones where it can change the ranking.\n")

    grid = collections.defaultdict(collections.Counter)
    for nph, turn in records:
        grid[turn][nph] += 1

    print("Breakdown by turn\n")
    print(f"{'turn':>5}  {'calls':>6}  {'>%d phrases' % BOOST_N:>12}  "
          f"{'share':>7}  {'max':>4}  {'mean':>6}")
    for turn in sorted(grid):
        row = grid[turn]
        total = sum(row.values())
        big = sum(v for k, v in row.items() if k > BOOST_N)
        mean = sum(k * v for k, v in row.items()) / total
        print(f"{turn:>5}  {total:>6}  {big:>12}  {big / total:>6.1%}  "
              f"{max(row):>4}  {mean:>6.2f}")


if __name__ == "__main__":
    main()