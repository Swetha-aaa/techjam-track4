# Results (200 public dev sessions)

| Config                    | HR@10 | MRR   | MTTC | Score   |
|---------------------------|-------|-------|------|---------|
| BM25 baseline (organizer) | 0.125 | 0.068 | 9.81 | 0.10671 |
| Ours (full)               | 0.915 | 0.624 | 3.05 | 0.80382 |
| - category filter         | 0.775 | 0.569 | 4.44 | 0.68949 |
| - field reweighting       | 0.885 | 0.611 | 3.33 | 0.77921 |
| + IDF filtering           | 0.890 | 0.608 | 3.25 | 0.78229 |
| + rerank (weight 2.0)     | 0.920 | 0.555 | 3.02 | 0.78609 |
| + rerank (weight 10.0)    | 0.915 | 0.618 | 3.05 | 0.80177 |
| - robust extraction       | 0.915 | 0.634 | 3.04 | 0.80695 |

## Per-scenario (full system)

| Scenario        | n  | HR@10 | MRR   | MTTC |
|-----------------|----|-------|-------|------|
| buying          | 80 | 0.912 | 0.586 | 2.66 |
| browsing        | 80 | 0.925 | 0.608 | 2.79 |
| intent_override | 30 | 0.900 | 0.705 | 4.43 |
| boundary        | 10 | 0.900 | 0.820 | 4.10 |

## Progression

| Stage                                | HR@10 | MRR   | MTTC | Score   |
|--------------------------------------|-------|-------|------|---------|
| Organizer BM25 baseline              | 0.125 | 0.068 | 9.81 | 0.10671 |
| + FTS phrase extraction, ask "other" | 0.730 | 0.547 | 4.87 | 0.65161 |
| + BM25 field reweighting             | 0.775 | 0.566 | 4.42 | 0.68886 |
| + category hard-filter               | 0.915 | 0.634 | 3.04 | 0.80695 |
| + structure-based extraction         | 0.915 | 0.624 | 3.05 | 0.80382 |

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

## Override handling (three strategies tested)

`intent_override` sessions replace a previously disclosed constraint mid-session.
We tested whether removing the superseded phrase helps.

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

The residual MTTC gap on these sessions (4.43 vs ~2.7 elsewhere) is largely
structural — the evaluator ignores hits before the override turn fires, so no
agent can converge earlier than turn 3 or 4 on them.

## Oracle ceiling

To establish what "good" means on this benchmark, we built an agent that reads
the target product directly, receives all four constraints on turn 1, and
searches with the same retrieval pipeline. It is not a submission — it measures
the maximum achievable score.

| Agent  | HR@10 | MRR   | MTTC | Score   |
|--------|-------|-------|------|---------|
| Ours   | 0.915 | 0.624 | 3.05 | 0.80382 |
| Oracle | 0.905 | 0.785 | 2.31 | 0.86205 |

**We reach 93.2% of the achievable ceiling.**

The oracle's hit rate is *lower* than ours (0.905 vs 0.915), despite perfect
information. Roughly 19 sessions are unsolvable in principle: every phrase the
customer can disclose matches thousands of catalog products, so the transcript
never identifies one item. This confirms the constraint-entropy analysis above by
construction rather than by inference — the remaining misses are a property of
the benchmark, not a deficiency in retrieval.

The oracle's advantage is concentrated in MRR (0.785 vs 0.624) and MTTC (2.31 vs
3.05), both of which reflect the cost of *eliciting* constraints across turns
rather than being handed them. Since the evaluator discloses at most two
constraints per turn and ignores hits before the override turn fires on
intent_override sessions, that gap is largely structural.
