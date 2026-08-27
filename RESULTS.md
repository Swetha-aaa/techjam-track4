# Results (200 public dev sessions)

| Config                    | HR@10 | MRR   | MTTC | Score   |
|---------------------------|-------|-------|------|---------|
| BM25 baseline (organizer) | 0.125 | 0.068 | 9.81 | 0.10671 |
| Ours (full)               | 0.915 | 0.634 | 3.04 | 0.80695 |
| - category filter         | 0.775 | 0.566 | 4.42 | 0.68885 |
| - field reweighting       | 0.885 | 0.621 | 3.33 | 0.78218 |
| + IDF filtering           | 0.890 | 0.610 | 3.24 | 0.78318 |

## Per-scenario (full system)

| Scenario        | n  | HR@10 | MRR   | MTTC |
|-----------------|----|-------|-------|------|
| buying          | 80 | 0.912 | 0.586 | 2.66 |
| browsing        | 80 | 0.925 | 0.608 | 2.79 |
| intent_override | 30 | 0.900 | 0.771 | 4.37 |
| boundary        | 10 | 0.900 | 0.820 | 4.10 |

## Component notes

**Category hard-filter (+0.118).** Turn 1 always names the product category
(`Women Bodysuits`, `Accessories Belts`). ANDing it against the `categories`
column before phrase scoring narrows the pool from 50,000 to a few hundred, so
rare-phrase signal no longer competes with matches from unrelated categories.
The single largest component in the system — every scenario clears 0.90 HR@10
with it enabled. See the `- category filter` row above.

**Field reweighting (+0.025).** Constraints are drawn from `features` and
`details`, so those columns carry the evidence. Full sweep below.

**IDF filtering (rejected).** Hurts at every threshold that does anything.
Notably it costs less once the category filter is active (0.783 vs 0.621
without), since the pool is already narrow. Still net negative. Details below.

## Progression

| Stage                                | HR@10 | MRR   | MTTC | Score   |
|--------------------------------------|-------|-------|------|---------|
| Organizer BM25 baseline              | 0.125 | 0.068 | 9.81 | 0.10671 |
| + FTS phrase extraction, ask "other" | 0.730 | 0.547 | 4.87 | 0.65161 |
| + BM25 field reweighting             | 0.775 | 0.566 | 4.42 | 0.68886 |
| + category hard-filter               | 0.915 | 0.634 | 3.04 | 0.80695 |

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
