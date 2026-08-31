# Results (200 public dev sessions)

| Config                    | HR@10 | MRR   | MTTC | Score   |
|---------------------------|-------|-------|------|---------|
| BM25 baseline (organizer) | 0.125 | 0.068 | 9.81 | 0.10671 |
| Ours (full)               | 0.970 | 0.694 | 2.53 | 0.86267 |
| - category filter         | 0.800 | 0.645 | 4.18 | 0.72988 |
| - exact rescore           | 0.950 | 0.662 | 2.69 | 0.83995 |
| - clause duplication      | 0.970 | 0.692 | 2.54 | 0.86194 |
| - phrase adjacency        | 0.965 | 0.705 | 2.58 | 0.86256 |
| - field reweighting       | 0.950 | 0.680 | 2.71 | 0.84486 |
| - turn-1 tail capture     | 0.970 | 0.685 | 2.54 | 0.85991 |
| - ask rotation            | 0.970 | 0.694 | 2.53 | 0.86267 |
| - robust extraction       | 0.970 | 0.694 | 2.53 | 0.86267 |
| + IDF filtering           | 0.935 | 0.693 | 2.81 | 0.83920 |
| + category as content     | 0.960 | 0.675 | 2.57 | 0.85122 |
| + rerank (weight 2.0)     | 0.950 | 0.568 | 2.69 | 0.81181 |
| + rerank (weight 10.0)    | 0.950 | 0.651 | 2.69 | 0.83657 |

## Per-scenario (full system)

| Scenario        | n  | HR@10 | MRR   | MTTC |
|-----------------|----|-------|-------|------|
| buying          | 80 | 0.938 | 0.594 | 2.34 |
| browsing        | 80 | 1.000 | 0.724 | 2.12 |
| intent_override | 30 | 0.967 | 0.851 | 3.90 |
| boundary        | 10 | 1.000 | 0.783 | 3.20 |

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
| + turn-1 tail capture                | 0.950 | 0.662 | 2.69 | 0.83995 |
| + exact-substring rescoring          | 0.970 | 0.694 | 2.53 | 0.86267 |

## Component notes

**Category hard-filter.** Turn 1 always names the product category
(`Women Bodysuits`, `Accessories Belts`). ANDing it against the `categories`
column before phrase scoring narrows the pool from 50,000 to a few hundred, so
rare-phrase signal no longer competes with matches from unrelated categories.
The single largest component in the system.

**Exact-substring rescoring (+0.023).** The top 20 FTS candidates are reordered
by how many disclosed constraints appear as literal substrings of their
features/details text. The second largest component, and the one that corrected a
claim we had previously made about the benchmark's ceiling. Details below.

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

**Turn-1 tail capture (+0.003).** Some sessions disclose their first constraint
with no colon at all — `I'm looking for Accessories Belts. Buckle closure`. We
were discarding it. Details below.

**Ask rotation.** Once `other` has been refused twice, we cycle the named
attributes instead of setting `ask_attribute` to null. Details below.

**Structure-based extraction.** Buys invariance to a rewording of the
simulator's message templates. Details in the drift-robustness section below.

**IDF filtering (rejected).** Hurts at every threshold that does anything.
Details below.

**Category as scored content (rejected).** Costs 0.013 — the mirror image of
clause duplication. Details below.

**Semantic reranking (rejected).** MiniLM embeddings over the candidate pool
never beat lexical ordering at any blend weight. Details below.

The pattern across every component tested: mechanisms that sharpen the
discriminative signal (exact rescoring, adjacency, duplication, field weighting,
tail capture) help. Mechanisms that filter, discard or dilute evidence (IDF
filtering, semantic reranking, override phrase-removal, category as content)
consistently do not.

## Exact-substring rescoring

BM25 scores token overlap. That is a proxy for the thing we actually care about,
which is whether a disclosed phrase was copied character-for-character out of one
particular product's record. After the FTS query returns a candidate pool, we
count how many disclosed constraints appear as literal substrings of each
candidate's `features` + `details` text and sort by that count, with the original
FTS ordering as the tiebreak.

| Pool | HR@10 | MRR   | MTTC | Score   | Delta    |
|------|-------|-------|------|---------|----------|
| off  | 0.950 | 0.662 | 2.69 | 0.83995 |          |
| 20   | 0.970 | 0.694 | 2.53 | 0.86267 | +0.02272 |
| 50   | 0.970 | 0.688 | 2.52 | 0.86097 | +0.02103 |
| 100  | 0.970 | 0.686 | 2.52 | 0.86053 | +0.02059 |
| 200  | 0.970 | 0.691 | 2.52 | 0.86169 | +0.02174 |

Worth +0.023, the second largest component in the system after the category
filter, improving all three metrics simultaneously.

**It corrected a claim we had made.** Before this component our hit rate was
0.950, identical to the oracle's at the time, and we concluded that recall was
solved to the limit of the benchmark and that the ten remaining misses were
unsolvable in principle. That was wrong. Four of those sessions were retrievable
all along; BM25 could not rank the target into the top ten against distractors
with similar token profiles. The information-theoretic argument still holds for
the six that remain, and the oracle still misses exactly those six, but the
earlier claim was stronger than the evidence supported. We had inferred a ceiling
from an agreement between two systems that shared a ranking weakness.

**Flat above pool 20.** All four pool sizes gain roughly the same amount, which
is what should happen: the target is almost always within the first twenty
candidates, so widening the pool only adds products that score zero substring
hits and are therefore sorted below anything scoring at least one. Retained at 20
as the cheapest setting, and the one with the best MRR.

**It degrades safely.** Ties break on the original FTS index, so a pool with no
substring hits anywhere comes back in exactly the order retrieval produced. On a
session where the customer paraphrases — or on a private set that discloses
constraints differently — the component cannot do worse than the retrieval it
reorders.

The mechanism is the same finding that motivates the whole system, applied one
stage later. Retrieval uses the verbatim property to *find* candidates; this uses
it to *order* them.

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
MRR 0.624 → 0.644, MTTC 3.05 → 2.85). Measured at the 0.82394 configuration.

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

**It is a property of our query construction, not of the benchmark.** The
category is already a hard gate, so the principled alternative is to filter
outside the `MATCH` expression entirely — a join against an unindexed column, or
a precomputed rowid set — after which category contributes zero to ranking by
construction and duplication becomes a no-op. We kept the in-match form because
the mechanism is now measured from both directions and the behaviour is
understood. The +0.013 should be read as recovering weight we spent, not as free
gain.

Retained at 6. Any value >= 4 produces the same effect, since the parameter's
real function is to duplicate everything rather than to select a subset. Sweep
measured at the 0.83719 configuration.

## Turn-1 tail capture

Most sessions disclose their first constraint after a colon, which our structural
extractor reads. Some do not:

    I'm looking for Accessories Belts. Buckle closure
    I'm looking for Underwear Undershirts. Imported
    I'm looking for Bras Everyday Bras. Date First Available: March 19, 2021

The constraint trails the category sentence with no delimiter, and we were
discarding it. On the belt session above that discarded phrase was the most
discriminative thing the customer ever disclosed: `Buckle closure` matches 1,585
catalog products, against 7,503 for `leather` and 15,300 for `Imported`.

Capturing the text after the first sentence is worth +0.003 overall, but the gain
is concentrated where the shape occurs: `intent_override` MRR rises from 0.763 to
0.822, with no other scenario moving. That is the signature of a targeted fix
rather than a tuning artifact.

This was found by reading a single session end to end with
`eval/transcript.py --miss`, not from any aggregate metric. Every other figure in
this document is a mean over 200 sessions, and a mean cannot show you that turn 1
extracted nothing.

## Ask rotation

Setting `ask_attribute` to null makes the evaluator reply:

    Those options are not quite right yet. Ask me about one specific attribute.

We originally treated two refusals of `other` as exhaustion and went silent from
then on. That was self-defeating twice over. The prompt above is a request to
name an attribute, not a statement that nothing remains — and our extractor was
ingesting the prompt itself as a constraint, so every subsequent query carried
several copies of it.

Now the filler is recognised and discarded, and after two refusals we cycle
`material -> color -> style -> size -> use_case -> brand -> budget`. On the
public set this changes nothing, because the affected turns occur only in
sessions already lost. It is retained for the private set, where an agent whose
query degrades with every wasted turn is strictly worse than one that holds
steady.

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

Flat across 12-20, falling off on both sides. Below 12 the truncation discards
discriminative detail from longer constraints. Above 20 the opposite problem
appears: a 25-token marketing sentence compiled as a conjunction requires every
token to be present, which over-constrains the query and eliminates the target.
Retained at 12 — identical score to 16 and 20, with marginally cheaper queries.
Measured at the 0.82394 configuration.

## BM25 field weight sweep

| title | cats | feat | det  | store | desc | Score   |
|-------|------|------|------|------|-------|---------|
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
correctly. Disabled via `common_threshold = 1.0`. The document-frequency index is
retained for constraint-entropy analysis and for ordering clauses by rarity.
Measured before the category filter was added.

## Category as scored content (tested, rejected)

The category filter is applied as a column-scoped hard AND. We tested whether
also emitting it as an unscoped content clause inside the OR expression — so that
category terms contribute to ranking, not just to filtering — would help.

| Config                   | HR@10 | MRR   | MTTC | Score   |
|--------------------------|-------|-------|------|---------|
| Category as filter only  | 0.950 | 0.654 | 2.69 | 0.83718 |
| Category also as content | 0.945 | 0.622 | 2.77 | 0.82385 |

It costs 0.013, almost exactly what clause duplication gains. That is the same
finding arrived at from the opposite direction: duplication helps by halving the
category's relative weight in the score, and this change raises that weight. Once
the pool has been gated to a single category, every surviving candidate matches
the category equally, so category terms carry no discriminative information —
scoring them only dilutes the constraint evidence that does discriminate.

The category is most useful as a filter and least useful as a ranking signal.
Measured at the 0.83718 configuration.

## Semantic reranking (tested, rejected)

MiniLM-L6-v2 embeddings pre-computed over all 50,000 products, used to reorder
the FTS candidate pool. `fts_weight` controls how much of the original lexical
ordering is preserved — low values mean near-pure semantic ordering.

Score rises monotonically as semantic influence is reduced (0.602 at weight 0.1,
0.790 at 2.0, 0.797 at 5.0, 0.805 at 10.0), converging on the no-rerank baseline
without ever exceeding it. Shrinking the candidate pool to 15 did not help either
(0.801).

The reason is structural: the simulated customer quotes the target product's own
`features` and `details` verbatim, so an exact lexical match is near-certain
evidence of identity. Embedding similarity dilutes that into topical proximity —
it cannot distinguish the product a phrase was copied from and a product that
merely sounds similar.

The exact-substring rescorer documented above is the constructive version of this
same observation, and it is worth +0.023 where the embedding layer was worth
nothing. The difference is that one measures literal provenance and the other
measures resemblance.

Component disabled. The submitted agent has no model dependency and runs on the
Python standard library alone.

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

Measured at the 0.80382 configuration.

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

The gap between HR and rank-1 rate in the 500-5k bucket (0.870 vs 0.304) is not a
retrieval failure — the target reaches the top 10, but lexical scoring cannot
separate it from equally-matching distractors. We hypothesised a semantic
reranker would close this gap; it did not. Exact-substring rescoring, added
later, does attack exactly this gap and is the largest single reason our hit rate
moved from 0.950 to 0.970.

Measured at the 0.80382 configuration; the buckets are a property of the sessions
rather than of the agent, but the HR and rank-1 columns predate exact rescoring.

## Oracle ceiling

To establish what "good" means on this benchmark, we built an agent that reads
the target product directly, receives all four constraints on turn 1, and
searches with the same retrieval pipeline. It is not a submission — it measures
what this pipeline achieves when handed perfect information.

| Agent  | HR@10 | MRR   | MTTC | Score   |
|--------|-------|-------|------|---------|
| Ours   | 0.970 | 0.694 | 2.53 | 0.86267 |
| Oracle | 0.970 | 0.870 | 1.68 | 0.93238 |

**We reach 92.5% of that ceiling.**

Hit rate is identical. Our agent finds every target the oracle finds, despite
having to elicit constraints over several turns while the oracle receives all
four immediately.

Both agents miss the same 6 sessions. Those appear to be unsolvable in principle:
every phrase the customer can disclose matches thousands of catalog products, so
the transcript never identifies one item.

**We state that more carefully than we did before.** An earlier version of this
document made the same claim about a different set of ten sessions, and exact
rescoring subsequently recovered four of them. The oracle shares our ranking
stage, so any weakness in that stage lowers both agents together and reads as a
property of the benchmark rather than of our code. This is a real limitation of
the method: an oracle built from your own pipeline measures the pipeline, not the
task. We report the remaining six as unsolved rather than as proven unsolvable.

The remaining gap is MRR and MTTC, both of which measure the cost of *eliciting*
constraints rather than being handed them. The evaluator discloses at most two
constraints per turn, so an agent that must ask cannot match one that already
knows. Per-scenario the oracle caps at 0.938 on `buying` and 0.967 on
`intent_override` — the latter ignores hits before the override turn fires, a
ceiling no agent can cross.

## Generalization to unseen targets

Every figure above comes from the same 200 public sessions we developed against.
The private evaluation set uses disjoint users and disjoint targets, so the
obvious question is whether our score reflects a general capability or a fit to
those 200.

We build 200 synthetic sessions from catalog products that never appear as public
targets, using the evaluator's own generation path (`eval/generalization.py`).
The scenario mix matches the official 40/40/15/5 split, and the sample is
stratified by constraint entropy to match the public distribution — without that
control, a score gap could not be attributed, since a random catalog draw has
different discriminability from the curated official targets. The official
evaluator is used unmodified.

A single draw is one sample of a noisy process, so we repeat it across three
seeds (`eval/generalization_seeds.py`):

| Set                | HR@10 | MRR   | MTTC | Score   |
|--------------------|-------|-------|------|---------|
| Public             | 0.970 | 0.694 | 2.53 | 0.86267 |
| Synthetic, draw 1  | 0.890 | 0.637 | 3.25 | 0.79119 |
| Synthetic, draw 2  | 0.960 | 0.665 | 2.58 | 0.84768 |
| Synthetic, draw 3  | 0.915 | 0.648 | 2.96 | 0.81249 |

Synthetic mean 0.81712, standard deviation 0.02853, range 0.79119 - 0.84768. The
gap from public is 0.046, and public sits above the whole range rather than
inside it — the gap is not an artifact of which products a single draw happened
to select. With three draws we report the spread rather than a significance
claim.

**The decomposition is more informative than the aggregate.** Hit rate varies
widely across draws (0.890 - 0.960) while MRR is uniformly worse (0.637 - 0.665
against 0.694) with no overlap. Whether we *find* an uncurated target depends on
how the draw fell; how well we *rank* it does not. That is the signature of
sparse `features` and `details` fields, which give the ranker less to separate
the target from its distractors.

To determine whether the difference reflects fitted components or a harder
population, we ran the per-component ablation on both sets
(`eval/generalization_diag.py`, first draw, measured at the 0.83995
configuration before exact rescoring was added):

| Component removed  | Public delta | Synthetic delta |
|--------------------|--------------|-----------------|
| Category filter    | -0.13829     | -0.10095        |
| Clause duplication | -0.01232     | -0.01359        |
| Phrase adjacency   | -0.00616     | -0.01193        |
| Field reweighting  | -0.00934     | -0.00374        |

**No component is fitted.** Every one keeps its sign on both sets. Clause
duplication transfers within 0.0013, and phrase adjacency is worth roughly
*twice* as much on unseen targets as on the set we tuned against — the opposite
of an overfitting signature. These are the two components whose mechanisms we
isolated experimentally, which is the pattern worth noting: the components we
understand are the components that generalise.

Field reweighting is the weakest transfer, worth 0.009 on the public set and
0.004 on synthetic. It is also the most heavily tuned component (a six-point
weight sweep) and the least mechanistically explained, so it is the plausible
locus of whatever fitting exists. It remains positive on both sets, so there is
nothing to correct, but we treat its public-set contribution as an overestimate.

The category filter is worth less on synthetic targets. That points at population
rather than fitting: products drawn at random from the catalog sit in broader and
messier category paths, so the hard gate removes less of the pool.

**We read the gap as population difficulty rather than overfitting.** The
official targets come from a pipeline requiring usable pre-target purchase
history, so they are items customers actually bought — mainstream products with
well-populated `features` and `details`. Our synthetic pool draws from the whole
catalog, including obscure records with sparse fields. The private 800 are
generated by the same curated pipeline as the public 200, so they should resemble
the public population, not ours.

**All figures in this section predate exact-substring rescoring** and were
measured at the 0.83995 configuration. The rescorer should transfer at least as
well as the components measured here, since it depends only on the verbatim
property that defines the benchmark rather than on any tuned parameter, but we
have not re-run the sweep and do not claim it.

## Template drift robustness

Our extraction originally keyed on three exact lead-in strings emitted by the
simulator. To test whether that would survive a rewording in the private
evaluation set, we built a separate harness (`eval/drift_test.py`) that rephrases
every message template the simulator produces. The official evaluator is not
modified; drift figures come from this separate harness and are reported as such.

| Extraction      | Official templates | Rephrased templates |
|-----------------|--------------------|---------------------|
| Fixed-template  | 0.80695            | 0.66007             |
| Structure-based | 0.83995            | 0.83995             |

Structure-based extraction keys on the colon delimiter that separates lead-in
from constraint, plus intent markers (`ignore`, `actually`, `no strong feelings`)
rather than exact strings. It is byte-identical under the rewrite.

The fixed-template row was measured at our 0.80695 configuration, before later
retrieval work; it is retained to show the failure mode we were guarding against
— a 0.147 collapse when the wrapper wording changes.

The invariance is free. Until turn-1 tail capture was added, the fixed-template
extractor scored 0.003 higher on the official templates, and we treated that as
the price of robustness. It read the colon-free turn-1 constraint that our
structural path was discarding. With that gap closed, the two extractors score
identically on the official templates while only the structural one survives a
rewording. There is no longer a trade to justify.

Measured at the 0.83995 configuration.

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
than on general retrieval capability. Exact-substring rescoring, added after
these figures were taken, depends on the same property and would be expected to
contribute nothing under paraphrase — it degrades to a no-op rather than to a
penalty, since ties break on the original retrieval order.

**Semantic reranking does not recover the loss.** We expected the embedding layer
to earn its place once the lexical signal degraded. It did not: 0.628 versus
exact match's 0.632, marginally better on recall and worse on rank quality — the
same pattern as the verbatim condition. The limitation is not that queries are
unnaturally phrased but that the *documents* are. Product records are marketing
copy and specification fields, which embed poorly regardless of how the query is
worded. Closing this gap would require purpose-built product representations
rather than embedding raw catalog text, which we did not attempt.

Measured at the 0.80382 configuration.

## Latency and resource profile

The submission rules ask for latency to be disclosed. Measured with
`eval/perf.py` on a Windows ARM64 laptop (Snapdragon X), Python 3.14, across the
500 turns of the 200 public sessions.

| Measure                              | Value    |
|--------------------------------------|----------|
| Median per-turn latency              | 30.6 ms  |
| Mean per-turn latency                | 44.9 ms  |
| p95 per-turn latency                 | 130.4 ms |
| p99 per-turn latency                 | 247.5 ms |
| Worst single turn                    | 374.7 ms |
| Full 200-session run, after startup  | 23.1 s   |
| FTS5 index build (once per process)  | 69.2 s   |
| Peak RSS                             | 442.1 MB |
| ...baseline before agent constructed | 236.6 MB |
| ...marginal cost of the agent        | ~205 MB  |
| LLM tokens consumed                  | 0        |

Unlike every other figure in this document, these are wall-clock measurements and
vary between runs. The scores do not: the agent is deterministic, so a change in
score is always a real effect of a code change.

**Exact-substring rescoring roughly tripled per-turn latency.** Before that
component the median was 10.9 ms and the index build 32 s. The cost is a second
pass over the catalog at construction time, caching the lowercased
features+details text of all 50,000 products, plus a substring scan over 20
candidates per turn. Python-level allocation rose from 10.4 MB to 52.9 MB, which
is that cache. We consider 31 ms per turn an acceptable price for +0.023, but it
is a real cost and it is the reason the component is not free.

Two figures deserve qualification rather than a favourable reading.

**The peak RSS is not all ours.** 236.6 MB is already resident before the agent
is constructed — the Python interpreter plus the evaluator's own in-memory
catalog. The agent's marginal footprint is roughly 205 MB: the SQLite FTS5 index
over 50,000 products, plus the 53 MB substring cache. Note that the index itself
lives in SQLite's C layer, which `tracemalloc` cannot see, so reporting the
Python figure alone would understate the real cost substantially.

**The 69 s startup is a one-off.** It is index construction plus cache building,
paid once per process rather than per session or per turn. Amortised over 200
sessions it is 346 ms each; over the 800 private sessions it would be 86 ms each.
If startup cost mattered more than it does here, the index could be built once
and attached from disk.

The agent imports nothing beyond the Python standard library, downloads no model
weights, and makes no network calls. `psutil` is used by this profiling script
only and is not a dependency of the submitted agent.

Token usage is zero, and that is a measured fact rather than an omission: no code
path in the submitted agent calls a model.
