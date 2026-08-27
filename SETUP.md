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