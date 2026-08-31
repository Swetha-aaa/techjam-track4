## Inspiration

We started building what the problem statement described: an agent that infers shopper intent from conversation and narrows toward a product. Before tuning it, we did something we'd recommend to anyone entering a benchmarked competition — we read the evaluator.

The simulated customer does not paraphrase. It builds its messages by lifting four strings **verbatim** out of the target product's own `features` and `details` fields, so it says things like `Material:alloy` and `Stretchy fabric: 95% modal, 5% spandex`. No shopper types that.

That single observation inverts the problem. An exact lexical match is not an approximation of intent — it is direct evidence of which product a string was copied from. Every semantic method we tried afterwards lost, and this is why: embedding similarity converts near-certainty about identity into topical proximity, which carries strictly less information.

The rest of our work was spent finding out how far that insight goes, and where it stops.

## What it does

Given up to 10 conversational turns, the agent returns a ranked top-10 product list every turn while eliciting further constraints from the customer.

**Score: 0.86267** on the 200 public sessions, against an organizer baseline of **0.10671**. Hit rate 0.970, MRR 0.694, mean turns-to-conversion 2.53.

The number we care about more: we built an **oracle** — an agent that reads the target directly and receives all four constraints on turn 1 — to find out what this pipeline reaches with perfect information. It scores 0.93238. **We reach 92.5% of that, with an identical hit rate.** Our agent finds every target the oracle finds, despite having to elicit constraints across turns while the oracle is handed them.

Both agents miss the same 6 sessions, where every phrase the customer can disclose matches thousands of catalog products, so the transcript never identifies one item.

The agent uses **no ML model, makes no network calls, and imports nothing beyond the Python standard library.** Median latency ~31 ms per turn, zero LLM tokens.

**Code:** https://github.com/Swetha-aaa/techjam-track4
**Demo video:** PASTE YOUTUBE URL HERE

## How we built it

Four stages per turn.

**Structure-based extraction.** We key on the colon delimiter separating lead-in from constraint, plus intent markers (`ignore`, `actually`, `no strong feelings`), rather than on the simulator's exact template strings. This buys invariance to rewording — we built a harness that rephrases every message template and our score is byte-identical, while the fixed-string extractor we started with collapses by 0.147.

**Category gating.** Turn 1 names the category; ANDing it against the catalog narrows 50,000 products to a few hundred. Worth +0.138, the largest single component. Used as a filter only — we measured that letting category terms also drive ranking costs 0.013.

**Lexical retrieval.** Constraints compile into SQLite FTS5 queries. Each phrase is emitted both as a token conjunction and as a contiguous phrase match, because the phrase was copied contiguously out of the target's record and contiguity is itself evidence. BM25 column weights are tuned toward `features` and `details`, where the disclosed strings originate.

**Exact-substring rescoring.** BM25 measures token overlap, which is a proxy. The constraints were copied character-for-character out of one product's record, so we reorder the top 20 candidates by how many disclosed phrases appear as literal substrings of their `features` and `details`. Worth +0.023, the second largest component, improving hit rate, MRR and time-to-conversion simultaneously — the same insight that drives retrieval, applied one stage later to ranking.

Every behaviour lives in a config dict that can be overridden per instance, so `eval/ablation.py` generates every measurement automatically. No number in our results document is transcribed by hand.

**Development tools.** VS Code, PowerShell, Git. Windows ARM64 (Snapdragon X), Python 3.14. No cloud compute, no notebooks — everything runs on a laptop.

**APIs used.** None. The submitted agent makes no network calls of any kind.

**Libraries and frameworks.** The Python standard library only — `sqlite3` (FTS5 full-text index), `re`, `json`, `pathlib`. No third-party package is imported by the submitted agent, and no model weights are downloaded. `sentence-transformers` (MiniLM-L6-v2) and `psutil` appear in the repository for the semantic reranking experiments we tested and rejected, and for the performance profiler; neither is on any inference path.

**Datasets and assets.** The organizer-provided product catalog (50,000 items) and the 200 public development sessions, both from the TechJam participant kit. We additionally generated 200 synthetic evaluation sessions from catalog products that never appear as public targets, using the evaluator's own generation path, to test generalization. No external datasets, no manual labelling, no pre-trained weights.

## Challenges we ran into

**We were wrong twice about our own system, and caught it with diagnostics.**

Duplicating each retrieval clause was worth +0.013 and we could not explain why. Our first hypothesis was rarity weighting — clauses are ordered by rarity and the first N duplicated, so surely it boosts distinctive constraints. We wrote `eval/phrase_counts.py` to check, and no retrieval call holds more than six phrases before turn 6, while `eval/dup_diff.py` showed sessions changing at turns 1–3. At those turns *everything* is duplicated, so rarity ordering can't be the variable.

Second hypothesis: uniform amplification. Also wrong — BM25 sums per-clause contributions, so doubling a homogeneous query scales every score equally and leaves ranking untouched. We verified it directly with four query shapes, all returning byte-identical top-10 lists.

The actual mechanism is an asymmetry. The category clause appears once while content clauses appear twice, so duplication halves the category's relative contribution to the score. Three of four category-anchored queries reorder under duplication; no category-free query does. And the honest addendum: that is a property of *our query construction*, not of the benchmark. Filtering outside the match expression would make it a no-op by construction. The +0.013 recovers weight we spent, rather than being free gain.

**Two real bugs were invisible to every aggregate metric.** We built a single-session inspector that prints a conversation turn by turn with each constraint annotated by how many catalog products match it. It immediately showed that turn 1 extracted *nothing* on messages shaped `<category>. <constraint>` with no colon — on one session that discarded `Buckle closure`, matching 1,585 products, the most discriminative thing the customer ever said. It also showed we were ingesting the evaluator's own filler prompts as constraints. Every other figure we had was a mean over 200 sessions, and a mean cannot show you that turn 1 read nothing.

## Accomplishments that we're proud of

**We measured our ceiling instead of guessing at it.** "92.5% of a ceiling we built to measure" is a claim we can defend; a raw score is not.

**We rejected four components with data, including one the organizers recommended.** Workshop guidance says a strong agent replaces a superseded constraint on an override turn. We implemented two removal heuristics and both lost, because the evaluator never reveals *which* constraint was superseded — so any removal sometimes deletes a still-valid phrase, and losing a valid constraint costs more than keeping a stale one. IDF filtering, semantic reranking, and category-as-ranking-signal were rejected the same way, each with its numbers published.

**We tested generalization with a control.** Anyone can report a public-set score. We built 200 synthetic sessions from targets absent from the public set, stratified by constraint entropy so the comparison isolates fitting from population difficulty, and repeated it across three seeds: 0.806 mean, range 0.793–0.822. Every component keeps its sign on both sets. Phrase adjacency is worth roughly *twice* as much on unseen targets — the opposite of an overfitting signature.

**It holds up outside the benchmark harness.** Resource usage is proportionate and the architecture depends on nothing that might be absent at scoring time or in production:

| | |
|---|---|
| Dependencies | None beyond the Python standard library |
| Model weights | None |
| Network | Not required |
| Per-turn latency | ~31 ms median |
| Memory | ~150 MB, essentially all the FTS5 index |
| LLM tokens | 0 |
| Determinism | Same input, same output, to five decimals |

The competition rules warn that network access may be disabled during scoring. An agent built on a hosted model API may not run at all under that condition; ours is unaffected. The same property is what makes it deployable — no per-query inference cost, no rate limit, no vendor dependency, and a failure mode that degrades to a looser query rather than to an outage.

## What we learned

**We over-claimed a ceiling, and our own measurement caught it.** For most of the competition our hit rate was 0.950 — exactly the oracle's — and we concluded that recall was solved to the benchmark's limit and the ten remaining misses were unsolvable in principle. Then we added exact-substring rescoring and it recovered four of them. The oracle shares our ranking stage, so a weakness in that stage lowered both agents together and looked like a property of the task. An oracle built from your own pipeline measures the pipeline, not the problem. We now report the six remaining misses as unsolved rather than proven unsolvable, and we would treat any self-built ceiling with more suspicion next time.

**Our result depends on verbatim disclosure, and we measured exactly how much.** We built a harness that rewrites each disclosed constraint into natural phrasing before it reaches the agent, covering 76% of constraint instances. Our score falls from 0.804 to 0.632 — a 21% relative drop. That is the honest measure of how much of this result belongs to the benchmark's design rather than to general retrieval capability.

**And the obvious fix does not work, for a reason worth knowing.** We expected semantic reranking to earn its place once the lexical signal degraded. It didn't: 0.628 against exact matching's 0.632. The limitation isn't that the *queries* are unnaturally phrased — it's that the *documents* are. Product records are marketing copy and spec fields, which embed poorly no matter how the query is worded.

That last point is the finding we'd carry into a real shopping system. The instinct when conversational search underperforms is to reach for a better embedding model on the query side. Our measurements say the constraint sits on the catalog side, and that closing it needs purpose-built product representations rather than embeddings over raw catalog text. We did not build those — but we know that's where the work is, and we know it because we measured rather than assumed.

## What's next for Reading the Evaluator: A Zero-Dependency Shopping Agent

**Purpose-built product representations.** The paraphrase experiment localizes the problem precisely: normalize catalog records into canonical attribute statements at index time, so that a natural utterance and a spec field meet in the same space. This is the one change our data says would move the real-world number.

**Rarity-weighted rescoring.** Our rescorer counts each substring hit equally, but a product containing `Triple Moon Pentagram Symbol` — 16 catalog matches — is far stronger evidence than one containing `Imported`, which matches 15,300. Weighting by inverse document frequency targets the entropy bucket where our hit rate is high but our rank-1 rate is not.

**Gate outside the match expression.** Makes the category a pure filter by construction and removes the need for clause duplication entirely.

**Elicitation policy.** Our remaining gap to the oracle is entirely MRR and MTTC — the cost of asking. We rotate attributes but do not choose them by expected information gain. Selecting the question that best partitions the current candidate set is the natural next step.

Full measurements, including every rejected component and every configuration we tested, are in [`RESULTS.md`](https://github.com/Swetha-aaa/techjam-track4/blob/main/RESULTS.md) — generated by our ablation script rather than transcribed by hand.