# TechJam 2026 — Track 4: Conversational Shopping Agent

A conversational product-search agent for the TechJam Track 4 challenge. Given a
customer's messages across at most 10 turns, it must surface the customer's
hidden target product in a top-10 recommendation list, as quickly and as highly
ranked as possible.

**Current score: 0.80382** on the 200 public development sessions, against the
organizer's BM25 baseline of **0.10671** — a 7.5× improvement.

The agent uses **no machine-learning model and requires no network access.** It
runs on the Python standard library alone.

---

## How it works

The system has three stages per turn.

**1. Constraint extraction.** The evaluator's simulated customer discloses
constraints in the form `<lead-in>: <phrase>[; <phrase>]`. We extract the body
after the colon rather than matching fixed lead-in strings, so the extractor
survives a rewording of the templates (see *Template drift robustness* in
`RESULTS.md`). Intent markers such as `ignore`, `actually` and `no strong
feelings` are detected separately to handle mid-session preference changes and
refusals.

**2. Category gating.** Turn 1 always names the product category
(`Women Bodysuits`, `Accessories Belts`). We AND this against the catalog's
`categories` column before any phrase scoring, narrowing the pool from 50,000
products to a few hundred. This is the single largest component in the system,
worth +0.118.

**3. Phrase retrieval.** Accumulated constraint phrases are compiled into an
SQLite FTS5 query — each phrase becomes a conjunctive token group, groups are
OR'd, and BM25 ranks the result. Column weights are tuned heavily toward
`features` and `details`, which is where the customer's disclosed phrases
originate.

Recommendations are returned on **every** turn. The evaluator checks the
recommendation list before it reads `ask_attribute`, so asking a question costs
nothing — there is never a reason to withhold a guess.

### The central finding

The simulated customer does not paraphrase. The evaluator builds it by lifting
four strings **verbatim** out of the target product's own `features` and
`details` fields — hence utterances like `Material:alloy` and `Triple Moon
Pentagram Symbol`, which no human would type.

An exact lexical match is therefore not an approximation of user intent; it is
direct evidence of which product a string was copied from. This is why semantic
reranking, IDF filtering, and phrase-removal heuristics all measurably *hurt* —
each dilutes or discards information that exact matching was using correctly.
All three are documented with their numbers in `RESULTS.md`.

---

## Setup

Requires Python 3.10+. No third-party packages are needed to run the agent.

```bash
git clone https://github.com/Swetha-aaa/techjam-track4.git
cd techjam-track4
```

Download the participant kit from the [organizer's release][kit] and copy
`catalog.jsonl` and `public_set.jsonl` into `data/`. These files are gitignored
(large, and organizer-distributed).

[kit]: https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit

Verify:

```bash
python -m evaluator.local_evaluator
```

Expected: `recommended_technical_score: 0.80382`. See `SETUP.md` for details.

---

## Reproducing our results

```bash
python -m evaluator.local_evaluator   # score the current agent
python -m eval.ablation               # full ablation sweep, regenerates RESULTS.md
python -m eval.entropy                # constraint entropy analysis
python -m eval.drift_test             # template-rewording robustness
python -m eval.rank_dist              # rank distribution of hits
```

Everything is **deterministic** — no model sampling, and the evaluator seeds its
RNG per session, so repeated runs produce byte-identical results. Any change in
score is a real effect of a code change, never variance.

The ablation sweep loads a sentence-transformers model for the two rejected
rerank configurations. To run it without that dependency, remove those entries
from `CONFIGS` in `eval/ablation.py`.

---

## Repository layout

```
starter/agent.py           the submitted agent
starter/agent_baseline.py  organizer's BM25 baseline, preserved for comparison
src/rerank.py              semantic reranker (tested, disabled — see RESULTS.md)
src/build_embeddings.py    one-time embedding cache builder for the above
eval/ablation.py           config sweep; regenerates RESULTS.md
eval/entropy.py            constraint-rarity vs. success-rate analysis
eval/drift_test.py         separate harness that rephrases simulator templates
eval/rank_dist.py          distribution of achieved ranks
evaluator/                 organizer's evaluator — unmodified
docs/                      organizer's specification and rules
RESULTS.md                 all measurements, including rejected components
```

`evaluator/` and `data/public_set.jsonl` are official artifacts and are never
edited. The drift test operates on a **copy** of the evaluator's message
producers and its figures are reported separately from official scores.

---

## Configuration

All behaviour is controlled by `DEFAULT_CONFIG` in `starter/agent.py`, and any
key can be overridden per instance:

```python
Agent(config={"use_category_filter": False})
```

`eval/ablation.py` uses this to generate every ablation row automatically, so no
measurement in `RESULTS.md` is transcribed by hand.

---

## Limitations

**Tuned against 200 sessions.** The private evaluation set uses different users
and different target products. Our drift test provides some assurance against
template variation, but we cannot verify generalisation to the private set
directly. We deliberately did not hardcode against any specific session.

**Extraction assumes a colon-delimited disclosure format.** If the private
evaluator discloses constraints in a structurally different way — no delimiter,
or embedded in free prose — extraction degrades to treating the whole message as
a query. That path still functions but scores lower.

**No free-text capability.** The agent is built for a simulator that quotes
catalog text verbatim. It would not handle a real shopper writing "something
warm for hiking" — that case genuinely needs the semantic layer we measured and
disabled here.

**A ceiling we did not reach.** 16 of our 17 remaining misses are sessions where
every disclosed phrase matches thousands of catalog products. No matching
strategy resolves these, because the transcript does not contain enough
information to identify one product. See the constraint entropy analysis in
`RESULTS.md`.

---


Organizer documentation for the challenge is preserved in `KIT_README.md` and
`docs/`.