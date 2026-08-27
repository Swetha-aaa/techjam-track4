"""Run config variants through the evaluator and regenerate RESULTS.md."""
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent
from starter.agent_baseline import Agent as BaselineAgent

SESSIONS = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")

SCENARIOS = ["buying", "browsing", "intent_override", "boundary"]

# --- ablation configs --------------------------------------------------------
# Each entry is a config override applied to the full system. Rows are generated
# automatically — never transcribe these numbers by hand.
CONFIGS = {
    "Ours (full)":          {},
    "- category filter":    {"use_category_filter": False},
    "- field reweighting":  {"rank": "bm25(products,0.0,6.0,4.0,2.5,2.5,1.5,1.0)"},
    "+ IDF filtering":      {"common_threshold": 0.15},
    "+ rerank (weight 2.0)": {"use_rerank": True, "fts_weight": 2.0},
    "+ rerank (weight 10.0)": {"use_rerank": True, "fts_weight": 10.0},
    "- robust extraction":  {"robust_extraction": False},
}
# -----------------------------------------------------------------------------

# --- hand-maintained prose ---------------------------------------------------
COMPONENT_NOTES = """
## Component notes

**Category hard-filter (+0.118).** Turn 1 always names the product category
(`Women Bodysuits`, `Accessories Belts`). ANDing it against the `categories`
column before phrase scoring narrows the pool from 50,000 to a few hundred, so
rare-phrase signal no longer competes with matches from unrelated categories.
The single largest component in the system — every scenario clears 0.90 HR@10
with it enabled. See the `- category filter` row above.

**Field reweighting (+0.025).** Constraints are drawn from `features` and
`details`, so those columns carry the evidence. Full sweep below.

**Structure-based extraction (-0.003).** Costs a fraction on the public set and
buys invariance to a rewording of the simulator's message templates. Details in
the drift-robustness section below.

**IDF filtering (rejected).** Hurts at every threshold that does anything.
Notably it costs less once the category filter is active (0.783 vs 0.621
without), since the pool is already narrow. Still net negative. Details below.

**Semantic reranking (rejected).** MiniLM embeddings over the candidate pool
never beat lexical ordering at any blend weight. Details below.
"""

PROGRESSION = """
## Progression

| Stage                                | HR@10 | MRR   | MTTC | Score   |
|--------------------------------------|-------|-------|------|---------|
| Organizer BM25 baseline              | 0.125 | 0.068 | 9.81 | 0.10671 |
| + FTS phrase extraction, ask "other" | 0.730 | 0.547 | 4.87 | 0.65161 |
| + BM25 field reweighting             | 0.775 | 0.566 | 4.42 | 0.68886 |
| + category hard-filter               | 0.915 | 0.634 | 3.04 | 0.80695 |
| + structure-based extraction         | 0.915 | 0.624 | 3.05 | 0.80382 |
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
Selected: `0.1 / 0.5 / 15 / 15 / 0.1 / 0.5`. Measured before the category filter
was added, so absolute values differ from the current system.
"""

IDF_NOTE = """
## IDF phrase filtering (tested, rejected)

| Threshold | Score   | Filtering active |
|-----------|---------|------------------|
| 0.15      | 0.62081 | yes, aggressive  |
| 0.30      | 0.65298 | yes, mild        |
| 0.35      | 0.68886 | no               |
| 1.00      | 0.68886 | no               |

Every threshold that removed phrases lowered the score, monotonically. BM25's
ranking function already contains an IDF term, so token rarity is handled
internally; filtering on top discards conjunctive signal the ranker was using
correctly. Disabled via `common_threshold = 1.0`. The document-frequency index
is retained for constraint-entropy analysis. Measured before the category
filter was added.
"""

ENTROPY_NOTE = """
## Constraint entropy analysis

Each session's simulated customer is built from four phrases lifted verbatim from
the target product's own `features` and `details`. How *rare* those phrases are in
the catalog determines whether the session is solvable at all.

| Rarest constraint matches | n  | HR@10 | Rank-1 rate |
|---------------------------|----|-------|-------------|
| < 50 products             | 59 | 1.000 | 0.797       |
| 50 - 500                  | 40 | 0.975 | 0.700       |
| 500 - 5,000               | 92 | 0.870 | 0.304       |
| > 5,000                   |  9 | 0.556 | 0.222       |

Recall is perfect when any constraint is distinctive: 59/59 on sessions whose
rarest phrase matches under 50 products. 16 of our 17 misses fall in the two
common buckets, where every disclosed phrase (`Imported`, `100% Cotton`,
`Pull-On closure`) matches thousands of products.

The gap between HR and rank-1 rate in the 500-5k bucket (0.870 vs 0.304) is not
a retrieval failure — the target reaches the top 10, but lexical scoring cannot
separate it from equally-matching distractors. We hypothesised a semantic
reranker would close this gap; it did not (see below). The deficit appears to be
information-theoretic rather than algorithmic: when every disclosed phrase
matches thousands of products, nothing in the transcript identifies the target.
"""

RERANK_NOTE = """
## Semantic reranking (tested, rejected)

MiniLM-L6-v2 embeddings pre-computed over all 50,000 products, used to reorder
the FTS candidate pool. `fts_weight` controls how much of the original lexical
ordering is preserved — low values mean near-pure semantic ordering.

Score rises monotonically as semantic influence is reduced (0.602 at weight 0.1,
0.790 at 2.0, 0.797 at 5.0, 0.805 at 10.0), converging on the no-rerank baseline
without ever exceeding it. Shrinking the candidate pool to 15 did not help
either (0.801).

The reason is structural: the simulated customer quotes the target product's own
`features` and `details` verbatim, so an exact lexical match is near-certain
evidence of identity. Embedding similarity dilutes that into topical proximity —
it cannot distinguish the product a phrase was copied from and a product that
merely sounds similar. Reranking marginally improved recall at weight 2.0
(0.920 vs 0.915) but cost 0.067 MRR.

Component disabled. The submitted agent has no model dependency and runs on the
Python standard library alone.
"""

DRIFT_NOTE = """
## Template drift robustness

Our extraction originally keyed on three exact lead-in strings from the
simulator. To test whether that would survive a rewording in the private
evaluation set, we built a separate harness (`eval/drift_test.py`) that rephrases
every message template the simulator emits. The official evaluator is not
modified; drift figures come from this separate harness and are reported as such.

| Extraction      | Official templates | Rephrased templates |
|-----------------|--------------------|---------------------|
| Fixed-template  | 0.80695            | 0.66007             |
| Structure-based | 0.80382            | 0.80382             |

Structure-based extraction keys on the colon delimiter that separates lead-in
from constraint, plus intent markers (`ignore`, `actually`, `no strong feelings`)
rather than exact strings. It is invariant to the rewrite, at a cost of 0.003 on
the official templates — a trade we accept, since the 200 public sessions are the
set we tuned against and the 800 private sessions are the ones that count.
"""

NOTE_BLOCKS = (PROGRESSION, COMPONENT_NOTES, SWEEP_NOTE, IDF_NOTE,
               ENTROPY_NOTE, RERANK_NOTE, DRIFT_NOTE)
# -----------------------------------------------------------------------------


def run(agent):
    return evaluate(agent, SESSIONS, IDS, CATS, PRODS)


def fmt_table(headers, rows):
    """Markdown table with columns padded to equal width."""
    cols = [headers] + rows
    widths = [max(len(str(r[i])) for r in cols) for i in range(len(headers))]
    out = ["| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return out


def metrics_row(name, r):
    return [name,
            f"{r['hit_rate_at_10']:.3f}",
            f"{r['mrr']:.3f}",
            f"{r['mttc']:.2f}",
            f"{r['recommended_technical_score']:.5f}"]


def main():
    results = {}

    print("running organizer baseline...")
    results["BM25 baseline (organizer)"] = run(BaselineAgent())

    for name, override in CONFIGS.items():
        print(f"running {name}...")
        results[name] = run(Agent(config=override))

    lines = ["# Results (200 public dev sessions)", ""]
    lines += fmt_table(["Config", "HR@10", "MRR", "MTTC", "Score"],
                       [metrics_row(n, r) for n, r in results.items()])

    lines += ["", "## Per-scenario (full system)", ""]
    sm = results["Ours (full)"]["scenario_metrics"]
    lines += fmt_table(["Scenario", "n", "HR@10", "MRR", "MTTC"],
                       [[s, str(sm[s]["sample_count"]),
                         f"{sm[s]['hit_rate_at_10']:.3f}",
                         f"{sm[s]['mrr']:.3f}",
                         f"{sm[s]['mttc']:.2f}"] for s in SCENARIOS])

    for block in NOTE_BLOCKS:
        lines += ["", block.strip()]

    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print()
    for name, r in results.items():
        print(f"  {name:28} {r['recommended_technical_score']:.5f}")
    print("\nwrote RESULTS.md")


if __name__ == "__main__":
    main()