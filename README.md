# TechJam 2026 — Track 4: Conversational Shopping Agent

A conversational product-search agent for the TechJam Track 4 challenge. Given a
customer's messages across at most 10 turns, it must surface the customer's
hidden target product in a top-10 recommendation list, as quickly and as highly
ranked as possible.

**Score: 0.86267** on the 200 public development sessions, against the
organizer's BM25 baseline of **0.10671** — an 8.1× improvement. Hit rate 0.970.

The agent uses **no machine-learning model and requires no network access.** It
runs on the Python standard library alone. Median latency is ~11 ms per turn and
it consumes zero LLM tokens.

We also measured a ceiling. An oracle agent handed all four constraints on turn 1
scores 0.93238 on the same sessions, so **we reach 92.5% of what this pipeline
achieves with perfect information**, with identical hit rate. Details in
`RESULTS.md`.

---

## How it works

Four stages per turn.

**1. Constraint extraction.** The simulated customer discloses constraints in the
form `<lead-in>: <phrase>[; <phrase>]`. We extract the body after the colon
rather than matching fixed lead-in strings, so the extractor survives a rewording
of the templates — verified byte-identical under a full template rewrite, where a
fixed-string extractor collapses by 0.147. Intent markers (`ignore`, `actually`,
`no strong feelings`) are detected separately to handle mid-session preference
changes and refusals. A constraint trailing the turn-1 category sentence with no
colon at all is also captured.

**2. Category gating.** Turn 1 always names the product category
(`Women Bodysuits`, `Accessories Belts`). We AND this against the catalog's
`categories` column before any phrase scoring, narrowing the pool from 50,000
products to a few hundred. This is the single largest component in the system,
worth **+0.138**. It is used as a filter only — letting category terms also drive
the ranking costs 0.013, since every surviving candidate matches the category
equally.

**3. Phrase retrieval.** Accumulated constraint phrases are compiled into an
SQLite FTS5 query. Each phrase is emitted twice — once as a conjunctive token
group and once as a contiguous phrase match, since the phrase was copied
contiguously out of the target's record and contiguity is itself evidence. Groups
are OR'd and BM25 ranks the result, with column weights tuned heavily toward
`features` and `details`, where the customer's disclosed phrases originate.

**4. Exact-substring rescoring.** BM25 scores token overlap, which is a proxy for
what we actually care about: whether a disclosed phrase was copied
character-for-character out of one particular product's record. We reorder the
top 20 candidates by how many constraints appear as literal substrings of their
`features` and `details` text, keeping FTS order as the tiebreak. Worth
**+0.023** — the second largest component — improving all three metrics at once.
It degrades safely: a pool with no substring hits comes back in exactly the order
retrieval produced.

Recommendations are returned on **every** turn. The evaluator checks the
recommendation list before it reads `ask_attribute`, so asking a question costs
nothing — there is never a reason to withhold a guess.

### The central finding

The simulated customer does not paraphrase. The evaluator builds it by lifting
four strings **verbatim** out of the target product's own `features` and
`details` fields — hence utterances like `Material:alloy` and
`Triple Moon Pentagram Symbol`, which no human would type.

An exact lexical match is therefore not an approximation of user intent; it is
direct evidence of which product a string was copied from. This is why semantic
reranking, IDF filtering, and phrase-removal heuristics all measurably *hurt* —
each dilutes or discards information that exact matching was using correctly, and
why the exact-substring rescorer is worth more than any of them. All are
documented with their numbers in `RESULTS.md`, including one result that
contradicts the organizers' own workshop guidance.

---

## Setup

Requires Python 3.10+. No third-party packages are needed to run the agent.

```
git clone https://github.com/Swetha-aaa/techjam-track4.git
cd techjam-track4
```

Download the participant kit from the [organizer's release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit)
and copy `catalog.jsonl` and `public_set.jsonl` into `data/`. These files are
gitignored (large, and organizer-distributed).

Verify:

```
python -m evaluator.local_evaluator
```

Expected: `recommended_technical_score: 0.86267`. See `SETUP.md` for details.

---

## Reproducing our results

```
python -m evaluator.local_evaluator     # score the current agent
python -m eval.ablation                 # full sweep; regenerates RESULTS.md
python -m eval.oracle                   # ceiling measurement
python -m eval.entropy                  # constraint rarity vs. success rate
python -m eval.generalization_seeds     # synthetic sets across three seeds
python -m eval.generalization_diag      # overfitting vs. population difficulty
python -m eval.drift_test               # template-rewording robustness
python -m eval.paraphrase               # natural-language rewrite harness
python -m eval.perf                     # latency and memory
python -m eval.rank_dist                # rank distribution and miss breakdown
python -m eval.transcript --miss        # inspect the first session we fail
```

Everything is **deterministic** — no model sampling, and the evaluator seeds its
RNG per session, so repeated runs produce byte-identical results. Any change in
score is a real effect of a code change, never variance. (Latency figures from
`eval.perf` are wall-clock and vary a little between runs; every other number
here does not.)

The ablation sweep loads a sentence-transformers model for the two rejected
rerank configurations. To run it without that dependency, remove those entries
from `CONFIGS` in `eval/ablation.py`.

---

## What we measured beyond the score

`RESULTS.md` contains every figure, generated rather than transcribed. The
headline results:

| Question | Answer |
|---|---|
| How good is 0.86267? | 92.5% of a measured oracle ceiling of 0.93238 |
| Is recall still improvable? | Not by this pipeline — our hit rate equals the oracle's, 0.970 |
| Why do we miss 6 sessions? | Every disclosed phrase matches thousands of products; the oracle misses the same six |
| Are we overfitted to the public 200? | No — every component keeps its sign on synthetic sets built from unseen targets |
| What do we score on unseen targets? | 0.806 mean across three stratified draws (range 0.793–0.822) |
| What if the templates are reworded? | Byte-identical |
| What if the customer paraphrases? | 0.632. This is the honest limit of the approach |

---

## Repository layout

```
starter/agent.py              the submitted agent
starter/agent_baseline.py     organizer's BM25 baseline, preserved for comparison
src/rerank.py                 semantic reranker (tested, disabled — see RESULTS.md)
src/build_embeddings.py       one-time embedding cache builder for the above
eval/ablation.py              config sweep; regenerates RESULTS.md
eval/oracle.py                ceiling measurement
eval/entropy.py               constraint-rarity vs. success-rate analysis
eval/generalization.py        synthetic session builder from unseen targets
eval/generalization_seeds.py  multi-seed range
eval/generalization_diag.py   per-component ablation on both sets
eval/drift_test.py            separate harness that rephrases simulator templates
eval/paraphrase.py            separate harness that rewrites constraints naturally
eval/check_rewrite.py         coverage check for the paraphrase rewrite rules
eval/rank_dist.py             rank distribution of hits, and miss breakdown by scenario
eval/transcript.py            single-session inspector
eval/phrase_counts.py         phrases held per turn
eval/dup_probe.py             isolates the clause-duplication mechanism
eval/dup_diff.py              which sessions duplication changes
eval/perf.py                  latency and memory profile
evaluator/                    organizer's evaluator — unmodified
docs/                         organizer's specification and rules
RESULTS.md                    all measurements, including rejected components
```

`evaluator/` and `data/public_set.jsonl` are official artifacts and are never
edited. The drift and paraphrase harnesses operate on **copies** of the
evaluator's message producers, and their figures are reported separately from
official scores.

---

## Configuration

All behaviour is controlled by `DEFAULT_CONFIG` in `starter/agent.py`, and any
key can be overridden per instance:

```
Agent(config={"use_category_filter": False})
```

`eval/ablation.py` uses this to generate every ablation row automatically, so no
measurement in `RESULTS.md` is transcribed by hand.

---

## Limitations

**Sensitive to verbatim disclosure.** The agent is built for a simulator that
quotes catalog text verbatim. Our score falls from 0.804 to 0.632 when the
customer paraphrases instead — a 21% relative drop. We tested whether semantic
reranking recovers this and it does not: the constraint is that product records
are marketing copy and specification fields, which embed poorly regardless of
query phrasing. Closing this gap would need purpose-built product
representations, which we did not attempt.

**Our first ceiling estimate was too confident.** Before exact-substring
rescoring, our hit rate matched the oracle's at 0.950 and we concluded that
recall was solved and the remaining ten misses were unsolvable in principle.
Rescoring then recovered four of them. The oracle shares our ranking stage, so a
weakness there lowers both agents together and reads as a property of the
benchmark rather than of our code. An oracle built from your own pipeline
measures the pipeline, not the task. We now report the six remaining misses as
unsolved rather than as proven unsolvable.

**Extraction assumes a colon-delimited disclosure format.** If the private
evaluator discloses constraints in a structurally different way — no delimiter,
or embedded in free prose — extraction degrades to treating the whole message as
a query. That path still functions but scores lower.

**Tuned against 200 sessions.** The private set uses different users and
different targets. We built synthetic sessions from targets absent from the
public set to test this, stratified by constraint entropy so the comparison is
controlled, and score 0.806 on average across three draws. Every component keeps
its sign on both sets, so no component is fitted, but we cannot verify the
private set directly. Those figures predate exact rescoring.

---

## Given more time

Each of these follows from a measurement above rather than from speculation.

**Purpose-built product representations.** Our paraphrase experiment localizes
the real-world weakness precisely: the failure is on the document side, not the
query side. We would normalize catalog records into canonical attribute
statements at index time — so that `Material:alloy` and "it's made of alloy" meet
in the same space — rather than embedding raw marketing copy. This is the one
change our data says would move the number that matters outside this benchmark.

**Rarity-weighted rescoring.** The rescorer currently counts each substring hit
equally. A product containing `Triple Moon Pentagram Symbol` (16 catalog matches)
is far stronger evidence than one containing `Imported` (15,300). Weighting hits
by inverse document frequency should attack the 500–5,000 entropy bucket, where
hit rate is high but rank-1 rate is not.

**Gate outside the match expression.** The category is a hard filter, but it
currently sits inside the scored FTS expression, which is the only reason clause
duplication earns anything. Filtering via a precomputed rowid set instead would
make the category contribute exactly zero to ranking by construction, and remove
the need for duplication entirely — a simpler system with the same behaviour.

**Question selection by information gain.** Our entire remaining gap to the
oracle is MRR and MTTC — the cost of eliciting constraints across turns. We
currently rotate through attributes in a fixed order. Choosing the question that
best partitions the current candidate set attacks that gap directly.

---

Organizer documentation for the challenge is preserved in `KIT_README.md` and
`docs/`.
