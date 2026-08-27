"""Run configs through the evaluator and regenerate RESULTS.md."""
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent
from starter.agent_baseline import Agent as BaselineAgent

SESSIONS = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")

SCENARIOS = ["buying", "browsing", "intent_override", "boundary"]


# ---------------------------------------------------------------- hand-maintained
# Add one row per milestone. This is the story the writeup tells.
PROGRESSION = """
## Progression

| Stage                                | HR@10 | MRR   | MTTC | Score   |
|--------------------------------------|-------|-------|------|---------|
| Organizer BM25 baseline              | 0.125 | 0.068 | 9.81 | 0.10671 |
| + FTS phrase extraction, ask "other" | 0.730 | 0.547 | 4.87 | 0.65161 |
| + BM25 field reweighting             | 0.775 | 0.566 | 4.42 | 0.68886 |
"""

SWEEP_NOTE = """
## BM25 field weight sweep

| title | cats | feat | det  | store | desc | Score   |
|-------|------|------|------|-------|------|---------|
| 6.0   | 4.0  | 2.5  | 2.5  | 1.5   | 1.0  | 0.65161 |
| 2.0   | 3.0  | 6.0  | 6.0  | 1.0   | 1.5  | 0.66395 |
| 0.5   | 1.5  | 10.0 | 10.0 | 0.5   | 1.0  | 0.68322 |
| 0.1   | 0.5  | 15.0 | 15.0 | 0.1   | 0.5  | 0.68886 |
| 0.0   | 2.0  | 15.0 | 15.0 | 0.0   | 0.5  | 0.68851 |
| 0.0   | 0.0  | 1.0  | 1.0  | 0.0   | 0.0  | 0.61858 |

Constraints are drawn from `features` and `details`, so weighting those heavily
helps (+0.037). But zeroing the remaining fields costs 0.07 — `categories` and
`description` act as tiebreakers when features/details matches are ambiguous.
Selected: `0.1 / 0.5 / 15 / 15 / 0.1 / 0.5`.
"""
# ------------------------------------------------------------------------------

IDF_NOTE = """
## IDF phrase filtering (tested, rejected)

| Threshold | Score   | Filtering active |
|-----------|---------|------------------|
| 0.15      | 0.62081 | yes, aggressive  |
| 0.30      | 0.65298 | yes, mild        |
| 0.35      | 0.68886 | no               |
| 0.40      | 0.68886 | no               |
| 1.00      | 0.68886 | no               |

Every threshold that removed phrases lowered the score, monotonically. BM25's
ranking function already contains an IDF term, so token rarity is handled
internally; filtering on top discards conjunctive signal the ranker was using
correctly. Component disabled (`COMMON_THRESHOLD = 1.0`). The document-frequency
index is retained for constraint-entropy analysis.
"""


def run(agent):
    return evaluate(agent, SESSIONS, IDS, CATS, PRODS)


def fmt_table(headers, rows):
    """Build a markdown table with columns padded to equal width."""
    cols = [headers] + rows
    widths = [max(len(str(r[i])) for r in cols) for i in range(len(headers))]
    out = ["| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return out


def main():
    results = {}
    print("running baseline...")
    results["BM25 baseline (organizer)"] = run(BaselineAgent())
    print("running ours...")
    results["Ours"] = run(Agent())

    lines = ["# Results (200 public dev sessions)", ""]

    main_rows = [[name,
                  f"{r['hit_rate_at_10']:.3f}",
                  f"{r['mrr']:.3f}",
                  f"{r['mttc']:.2f}",
                  f"{r['recommended_technical_score']:.5f}"]
                 for name, r in results.items()]
    lines += fmt_table(["Config", "HR@10", "MRR", "MTTC", "Score"], main_rows)

    lines += ["", "## Per-scenario (ours)", ""]
    sm = results["Ours"]["scenario_metrics"]
    scen_rows = [[s,
                  str(sm[s]["sample_count"]),
                  f"{sm[s]['hit_rate_at_10']:.3f}",
                  f"{sm[s]['mrr']:.3f}",
                  f"{sm[s]['mttc']:.2f}"]
                 for s in SCENARIOS]
    lines += fmt_table(["Scenario", "n", "HR@10", "MRR", "MTTC"], scen_rows)

    lines += ["", PROGRESSION.strip()]
    lines += ["", SWEEP_NOTE.strip()]
    lines += ["", IDF_NOTE.strip()]

    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print()
    for name, r in results.items():
        print(f"  {name:32} {r['recommended_technical_score']:.5f}")
    print("\nwrote RESULTS.md")


if __name__ == "__main__":
    main()