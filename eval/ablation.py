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
    "Ours (full)":            {},
    "- category filter":      {"use_category_filter": False},
    "- clause duplication":   {"clause_duplication": 0},
    "- phrase adjacency":     {"phrase_adjacency": False},
    "- field reweighting":    {"rank": "bm25(products,0.0,6.0,4.0,2.5,2.5,1.5,1.0)"},
    "- robust extraction":    {"robust_extraction": False},
    "+ IDF filtering":        {"common_threshold": 0.15},
    "+ rerank (weight 2.0)":  {"use_rerank": True, "fts_weight": 2.0},
    "+ rerank (weight 10.0)": {"use_rerank": True, "fts_weight": 10.0},
}
# -----------------------------------------------------------------------------

# --- hand-maintained prose ---------------------------------------------------
COMPONENT_NOTES = """
## Component notes

**Category hard-filter.** Turn 1 always names the product category
(`Women Bodysuits`, `Accessories Belts`). ANDing it against the `categories`
column before phrase scoring narrows the pool from 50,000 to a few hundred, so
rare-phrase signal no longer competes with matches from unrelated categories.
The single largest component in the system.

**Phrase-adjacency clauses (+0.020).** Each constraint is compiled twice — once
as a token conjunction and once as a contiguous phrase match. A product where the
tokens appear adjacently satisfies both clauses and outranks one where they are
merely scattered across the record. Details below.

**Clause duplication (+0.013).** Every clause is emitted twice, which halves the
category clause's relative weight in the BM25 score since its terms appear only
once. The category filter has already gated the pool, so this shifts ranking
influence from "sits in the right category" to "matches the stated constraints."
Details below.

**Field reweighting (+0.025).** Constraints are drawn from `features` and
`details`, so those columns carry the evidence. Full sweep below.

**Structure-based extraction (-0.004).** Costs a fraction on the public set and
buys invariance to a rewording of the simulator's message templates. Details in
the drift-robustness section below.

**IDF filtering (rejected).** Hurts at every threshold that does anything.
Notably it costs less once the category filter is active, since the pool is
already narrow. Still net negative. Details below.

**Semantic reranking (rejected).** MiniLM embeddings over the candidate pool
never beat lexical ordering at any blend weight. Details below.

The pattern across all seven: mechanisms that sharpen the discriminative signal
(adjacency, duplication, field weighting) help. Mechanisms that filter, discard
or dilute evidence (IDF filtering, semantic reranking, override phrase-removal)
consistently do not.
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
| + phrase-adjacency clauses           | 0.935 | 0.644 | 2.85 | 0.82394 |
| + clause duplication                 | 0.950 | 0.654 | 2.70 | 0.83719 |
"""

ADJACENCY_NOTE = """
## Phrase-adjacency clauses

Constraint phrases were originally compiled into token conjunctions:
`95% modal, 5% spandex` became `("95" AND "modal" AND "spandex")`, which matches
any product containing all three tokens anywhere in its record. But the phrase was
copied *contiguously* out of the target's `details`, so contiguity is itself
evidence.

We now emit both clause forms per constraint — the token conjunction and an FTS5
contiguous phrase match. A product where the tokens appear adjacently satisfies
two clauses rather than one, and BM25 ranks it accordingly.

Worth +0.020, improving all three metrics simultaneously (HR 0.915 → 0.935,
MRR 0.624 → 0.644, MTTC 3.05 → 2.85). Browsing reached 0.975 HR@10 and boundary
reached 1.000.
"""

DUPLICATION_NOTE = """
## Clause duplication

Each constraint's clauses are emitted twice into the OR expression. Worth +0.013,
and the mechanism took three attempts to identify correctly.

| N   | HR@10 | MRR   | MTTC | Score   |
|-----|-------|-------|------|---------|
| 0   | 0.935 | 0.644 | 2.85 | 0.82394 |
| 1   | 0.940 | 0.633 | 2.83 | 0.82352 |
| 2   | 0.940 | 0.640 | 2.81 | 0.82599 |
| 3   | 0.945 | 0.646 | 2.76 | 0.83102 |
| 4   | 0.950 | 0.653 | 2.73 | 0.83649 |
| 6   | 0.950 | 0.654 | 2.70 | 0.83719 |
| 10  | 0.950 | 0.654 | 2.70 | 0.83719 |

**It is not rarity weighting.** Clauses are ordered by rarity and the first N are
duplicated, so we assumed the gain came from boosting the most distinctive
constraints. But `eval/phrase_counts.py` shows no retrieval call holds more than
six phrases before turn 6, while `eval/dup_diff.py` shows sessions changing at
turns 1-3. At those turns every clause is duplicated, so rarity ordering cannot
be the operative variable — consistent with N=6 and N=10 scoring identically.

**It is not uniform amplification either.** FTS5's BM25 sums per-clause
contributions, so doubling every clause of a homogeneous query scales all scores
equally and leaves the ranking untouched. We verified this directly
(`eval/dup_probe.py`, Part 1): four query shapes — all single-token, all
multi-token, and two mixed — return byte-identical top-10 lists under duplication.

**The asymmetry is the category clause.** The full query is
`(categories:"x" AND categories:"y") AND (A OR A OR B OR B ...)`. The category
terms appear once; the content clauses appear twice. Duplication therefore halves
the category's relative contribution to the BM25 score. Part 2 of the probe
confirms it: three of four category-anchored queries reorder under duplication,
while every category-free query does not.

This is a sensible thing to want. The category filter has already served its
purpose as a hard gate — every surviving candidate is in the right category, so
letting category terms also drive the *ranking* wastes scoring weight on a
dimension that no longer discriminates. Duplication demotes it toward being a
pure filter and lets constraint evidence dominate the ordering.

Retained at 6. Any value ≥4 produces the same effect, since the parameter's real
function is to duplicate everything rather than to select a subset.
"""

TOKENCAP_NOTE = """
## Phrase token cap sweep

Each constraint is truncated to at most N tokens before compilation into a
conjunctive clause.

| Cap | Score   |
|-----|---------|
| 8   | 0.82319 |
| 12  | 0.82394 |
| 16  | 0.82394 |
| 20  | 0.82394 |
| 30  | 0.81894 |

Flat across 12–20, falling off on both sides. Below 12 the truncation discards
discriminative detail from longer constraints. Above 20 the opposite problem
appears: a 25-token marketing sentence compiled as a conjunction requires every
token to be present, which over-constrains the query and eliminates the target.
Retained at 12 — identical score to 16 and 20, with marginally cheaper queries.
Measured at the 0.82394 configuration.
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
is retained for constraint-entropy analysis and for ordering clauses by rarity.
Measured before the category filter was added.
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
rarest phrase matches under 50 products. 16 of the 17 misses at the time of this
measurement fell in the two common buckets, where every disclosed phrase
(`Imported`, `100% Cotton`, `Pull-On closure`) matches thousands of products.

The gap between HR and rank-1 rate in the 500-5k bucket (0.870 vs 0.304) is not
a retrieval failure — the target reaches the top 10, but lexical scoring cannot
separate it from equally-matching distractors. We hypothesised a semantic
reranker would close this gap; it did not (see below). The deficit appears to be
information-theoretic rather than algorithmic: when every disclosed phrase
matches thousands of products, nothing in the transcript identifies the target.

Measured at the 0.80382 configuration.
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
merely sounds similar.

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
rather than exact strings. It is invariant to the rewrite, at a small cost on the
official templates — a trade we accept, since the 200 public sessions are the set
we tuned against and the 800 private sessions are the ones that count.

Measured at the 0.80382 configuration.
"""

OVERRIDE_NOTE = """
## Override handling (three strategies tested)

`intent_override` sessions replace a previously disclosed constraint mid-session.
The organizers' own guidance describes a strong agent as one that *replaces* the
superseded value rather than appending the new one. We tested that.

| Strategy                        | Score   |
|---------------------------------|---------|
| Keep all, prepend new (current) | 0.80382 |
| Drop most recent phrase         | 0.79858 |
| Drop least-similar phrase       | 0.80196 |

Both removal strategies underperform. The evaluator does not reveal which
constraint was superseded, so any removal heuristic sometimes discards a phrase
that is still true of the target — and the cost of losing a valid constraint
exceeds the cost of retaining a stale one, since BM25 scores conjunctive matches
higher. Prepending the new value is sufficient: it dominates the query without
requiring us to guess what to delete.

The residual MTTC gap on these sessions is largely structural — the evaluator
ignores hits before the override turn fires, so no agent can converge earlier
than turn 3 or 4 on them.

Measured at the 0.80382 configuration.
"""

ORACLE_NOTE = """
## Oracle ceiling

To establish what "good" means on this benchmark, we built an agent that reads
the target product directly, receives all four constraints on turn 1, and
searches with the same retrieval pipeline. It is not a submission — it measures
the maximum achievable score.

| Agent  | HR@10 | MRR   | MTTC | Score   |
|--------|-------|-------|------|---------|
| Ours   | 0.915 | 0.624 | 3.05 | 0.80382 |
| Oracle | 0.905 | 0.785 | 2.31 | 0.86205 |

**We reached 93.2% of the achievable ceiling at this measurement.**

The oracle's hit rate is *lower* than ours (0.905 vs 0.915), despite perfect
information. Roughly 19 sessions are unsolvable in principle: every phrase the
customer can disclose matches thousands of catalog products, so the transcript
never identifies one item. This confirms the constraint-entropy analysis by
construction rather than by inference — the remaining misses are a property of
the benchmark, not a deficiency in retrieval.

The oracle's advantage is concentrated in MRR and MTTC, both of which reflect the
cost of *eliciting* constraints across turns rather than being handed them. Since
the evaluator discloses at most two constraints per turn and ignores hits before
the override turn fires, that gap is largely structural.

Measured at the 0.80382 configuration; both figures move together, since the
oracle shares the retrieval pipeline.
"""

PARAPHRASE_NOTE = """
## Paraphrase sensitivity

The official simulator lifts constraint strings verbatim from the target
product's `features` and `details`, so the customer says things like
`Material:alloy` and `Stretchy fabric: 95% modal, 5% spandex`. A real shopper
would not. We built a separate harness (`eval/paraphrase.py`) that rewrites each
disclosed constraint into natural phrasing before it reaches the agent — a
hand-written dictionary of common constraints plus seven structural rules,
covering 76.4% of the 800 constraint instances. The remainder are long marketing
sentences that are already natural prose and were left unchanged. The official
evaluator is not modified; these figures are reported separately.

| Condition   | Retrieval | HR@10 | MRR   | MTTC | Score   |
|-------------|-----------|-------|-------|------|---------|
| Verbatim    | exact     | 0.915 | 0.624 | 3.05 | 0.80382 |
| Verbatim    | semantic  | 0.920 | 0.555 | 3.02 | 0.78609 |
| Paraphrased | exact     | 0.715 | 0.497 | 4.75 | 0.63156 |
| Paraphrased | semantic  | 0.720 | 0.475 | 4.72 | 0.62801 |

Two findings.

**Our score depends substantially on verbatim disclosure.** Exact-match retrieval
falls from 0.804 to 0.632 when the customer paraphrases — a 21% relative drop.
This quantifies how much of our result rests on the simulator's design rather
than on general retrieval capability.

**Semantic reranking does not recover the loss.** We expected the embedding layer
to earn its place once the lexical signal degraded. It did not: 0.628 versus
exact match's 0.632, marginally better on recall and worse on rank quality — the
same pattern as the verbatim condition. The limitation is not that queries are
unnaturally phrased but that the *documents* are. Product records are marketing
copy and specification fields, which embed poorly regardless of how the query is
worded. Closing this gap would require purpose-built product representations
rather than embedding raw catalog text, which we did not attempt.

Measured at the 0.80382 configuration.
"""

NOTE_BLOCKS = (PROGRESSION, COMPONENT_NOTES,
               ADJACENCY_NOTE, DUPLICATION_NOTE, TOKENCAP_NOTE, SWEEP_NOTE,
               IDF_NOTE, RERANK_NOTE, OVERRIDE_NOTE,
               ENTROPY_NOTE, ORACLE_NOTE, DRIFT_NOTE, PARAPHRASE_NOTE)
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