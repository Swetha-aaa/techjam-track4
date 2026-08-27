# Results (200 public dev sessions)

| Config                    | HR@10 | MRR   | MTTC | Score   |
|---------------------------|-------|-------|------|---------|
| BM25 baseline (organizer) | 0.125 | 0.068 | 9.81 | 0.10671 |
| Ours                      | 0.730 | 0.547 | 4.87 | 0.65161 |

## Per-scenario (ours)

| Scenario        | n  | HR@10 | MRR   | MTTC |
|-----------------|----|-------|-------|------|
| buying          | 80 | 0.637 | 0.463 | 5.39 |
| browsing        | 80 | 0.800 | 0.582 | 4.20 |
| intent_override | 30 | 0.733 | 0.642 | 5.60 |
| boundary        | 10 | 0.900 | 0.649 | 3.90 |
