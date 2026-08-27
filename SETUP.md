# Setup

## Prerequisites
- Python 3.10+
- Git

## Steps

1. Clone this repo
2. Download the participant kit release:
   https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit
3. Copy `catalog.jsonl` and `public_set.jsonl` into `data/`
4. Verify the setup by running: `python -m evaluator.local_evaluator`

Expected output: `recommended_technical_score: 0.10671`

If your score differs, your catalog download is incomplete — re-download before doing any work.

## Notes

- `data/*.jsonl` is gitignored (large files, organizer-distributed data)
- Never commit API keys — use `.env`, which is gitignored
- Don't edit `evaluator/` or `data/public_set.jsonl`


## Do NOT edit

These are official artifacts. Editing them invalidates every number we report,
and the organizers run their own copies anyway.

- `evaluator/local_evaluator.py`, `evaluator/__init__.py` — the official scorer
- `data/public_set.jsonl` — official session labels
- `data/catalog.jsonl` — frozen catalog, read-only per competition rules
- `docs/` — organizer reference material
- `starter/agent_baseline.py` — preserved BM25 baseline, our ablation zero row
- `DATA_ATTRIBUTION.md` — licensing requirement

Exception: for robustness testing, COPY the evaluator to `eval/drift_test.py`
and modify the copy. Never the original.

## Ours to edit

- `src/retrieval.py` — [owner]
- `src/rerank.py` — [owner]
- `src/protocol.py` — [owner]
- `src/config.py` — [owner]
- `eval/ablation.py` — [owner]

`starter/agent.py` is our submission — the evaluator hardcodes this path.
Keep it a thin wrapper over `src/`. Coordinate before editing it.

## Workflow

```
git pull                          # before starting work
git checkout -b your-feature      # branch per feature
# work, then:
git add . && git commit -m "..."
git push -u origin your-feature   # open a PR into main
```