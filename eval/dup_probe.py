"""Isolate why clause duplication changes rankings.

Uniform duplication of a homogeneous query is provably inert: FTS5's BM25 sums
per-clause contributions, so doubling every clause scales all scores equally and
the ordering is unchanged. Yet duplication measurably improves the score
(+0.013), and eval/dup_diff.py shows sessions changing at turns 1-3 — where
eval/phrase_counts.py confirms no call holds more than 6 phrases, so every
clause is duplicated and the query should scale uniformly.

Something must be asymmetric even then. Candidates tested here:

  Part 1 — clause shape. Single-token phrases emit ONE clause per rep; multi-
  token phrases emit TWO (conjunction + adjacency). A mixed query might not
  scale uniformly.

  Part 2 — the category clause. The real query is
  `(categories:"x" AND categories:"y") AND (A OR A OR B OR B ...)`. The category
  terms appear once while the content clauses appear twice, so duplication
  shifts the balance between category evidence and constraint evidence. This
  asymmetry is present on every call that has a category, regardless of how many
  phrases are held.

Run: python -m eval.dup_probe
"""
from starter.agent import Agent, terms

CASES = {
    "all multi-token": ["Triple Moon Pentagram Symbol", "95% Cotton, 5% Spandex"],
    "all single-token": ["cotton", "Imported"],
    "mixed": ["cotton", "95% Cotton, 5% Spandex"],
    "mixed, three phrases": ["cotton", "Imported", "95% Cotton, 5% Spandex"],
}

CATEGORY_CASES = {
    "Women Bodysuits": ["cotton", "95% Cotton, 5% Spandex"],
    "Accessories Belts": ["leather", "100% Leather", "Buckle closure"],
    "Jewelry Necklaces": ["Material:alloy", "Triple Moon Pentagram Symbol"],
    "Shoes Slippers": ["polyester", "Imported", "Rubber sole"],
}

agent = Agent()
CAP = agent.cfg["max_phrase_tokens"]
RANK = agent.cfg["rank"]


def build(phrases, dup):
    """Rebuild the agent's clause construction, with duplication controllable."""
    clauses = []
    for i, ph in enumerate(sorted(phrases, key=agent._phrase_rarity)):
        toks = terms(ph)[:CAP]
        if not toks:
            continue
        conj = "(" + " AND ".join(f'"{t}"' for t in toks) + ")"
        adj = ('"' + " ".join(toks) + '"') if len(toks) > 1 else None
        for _ in range(2 if i < dup else 1):
            clauses.append(conj)
            if adj:
                clauses.append(adj)
    return "(" + " OR ".join(clauses) + ")", len(clauses)


def build_with_category(category, phrases, dup):
    """As the real agent builds it: category ANDed onto the OR expression."""
    expr, n = build(phrases, dup)
    cat_toks = terms(category)[:4]
    if not cat_toks:
        return expr, n
    cat = "(" + " AND ".join(f'categories : "{t}"' for t in cat_toks) + ")"
    return cat + " AND " + expr, n


def top10(expr):
    rows = agent.conn.execute(
        f"SELECT parent_asin FROM products WHERE products MATCH ? "
        f"ORDER BY {RANK} LIMIT 10", (expr,)).fetchall()
    return [r[0] for r in rows]


def compare(label, expr_off, expr_on, n_off, n_on):
    a, b = top10(expr_off), top10(expr_on)
    same = a == b
    print(f"{label}")
    print(f"  clauses  {n_off} -> {n_on}")
    print(f"  result   {'SAME' if same else 'DIFFERENT  <-- asymmetry here'}")
    if not same:
        print(f"    dup=0  {a}")
        print(f"    dup=6  {b}")
    print()
    return same


def main():
    print("=" * 64)
    print("PART 1 — clause shape, no category clause\n")
    for name, phrases in CASES.items():
        eo, no = build(phrases, 0)
        en, nn = build(phrases, 6)
        compare(f"{name}\n  phrases  {phrases}", eo, en, no, nn)

    print("=" * 64)
    print("PART 2 — with the category clause, as the real agent builds it\n")
    differing = 0
    for cat, phrases in CATEGORY_CASES.items():
        eo, no = build_with_category(cat, phrases, 0)
        en, nn = build_with_category(cat, phrases, 6)
        if not compare(f"{cat}\n  phrases  {phrases}", eo, en, no, nn):
            differing += 1

    print("=" * 64)
    if differing:
        print(f"{differing}/{len(CATEGORY_CASES)} category cases differ while the "
              f"category-free cases match.\nThe category clause is the asymmetry: "
              f"its terms appear once while the content\nclauses appear twice, so "
              f"duplication shifts weight from 'sits in the right\ncategory' "
              f"toward 'matches the stated constraints'.")
    else:
        print("No case differs. The asymmetry lies outside this construction —\n"
              "check the fallback query path in Agent._search next.")


if __name__ == "__main__":
    main()