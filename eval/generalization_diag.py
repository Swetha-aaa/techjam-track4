"""Is the synthetic-set gap overfitting, or is the population simply harder?

eval/generalization.py scores 0.047 lower on synthetic targets than on the public
200. Two explanations, with opposite implications:

  (a) Our components are fitted to the public set. The private 800 would show the
      same drop, and we should find and fix the offending component.

  (b) The synthetic population is harder in a way entropy stratification does not
      capture. The official targets were drawn from a pipeline requiring usable
      pre-target purchase history, so they are products people actually bought —
      mainstream, well-described items. Random catalog products include obscure
      ones with sparse records. If this is the cause it does not threaten the
      private set, which comes from the same curated pool as the public one.

Two controls:

  ORACLE. The oracle receives all four constraints on turn 1 and uses the same
  retrieval pipeline. It has nothing fitted to the public set — no extraction, no
  elicitation, no turn strategy. If the oracle drops by a similar margin, the
  population is harder. If the oracle holds while we fall, the gap is ours.

  PER-COMPONENT ABLATION. If a component that helps on the public set hurts on
  synthetic targets, that component is fitted. If every component keeps the same
  sign and rough magnitude, they generalise and the gap is population.

Run: python -m eval.generalization_diag
"""
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent
from eval.oracle import OracleAgent, CHEAT   # reuses the public-set cheat sheet

PUBLIC = load_jsonl("data/public_set.jsonl")
SYNTHETIC = load_jsonl("data/synthetic_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")

CONFIGS = {
    "full":                {},
    "- category filter":   {"use_category_filter": False},
    "- clause duplication": {"clause_duplication": 0},
    "- phrase adjacency":  {"phrase_adjacency": False},
    "- field reweighting": {"rank": "bm25(products,0.0,6.0,4.0,2.5,2.5,1.5,1.0)"},
}


def score(agent, sessions):
    return evaluate(agent, sessions, IDS, CATS, PRODS)


def main():
    print("PART 1 — per-component ablation on both sets\n")
    rows = []
    for name, override in CONFIGS.items():
        print(f"  running {name}...")
        p = score(Agent(config=override), PUBLIC)["recommended_technical_score"]
        s = score(Agent(config=override), SYNTHETIC)["recommended_technical_score"]
        rows.append((name, p, s))

    base_p, base_s = rows[0][1], rows[0][2]
    print()
    print(f"{'config':>22}  {'public':>8}  {'synth':>8}  "
          f"{'pub delta':>10}  {'syn delta':>10}")
    for name, p, s in rows:
        dp = "" if name == "full" else f"{p - base_p:+.5f}"
        ds = "" if name == "full" else f"{s - base_s:+.5f}"
        print(f"{name:>22}  {p:>8.5f}  {s:>8.5f}  {dp:>10}  {ds:>10}")

    print("\nIf every component keeps the same sign and rough magnitude on both "
          "sets,\nthe components generalise and the gap is population difficulty.")

    print("\n" + "=" * 70)
    print("PART 2 — oracle control on the public set\n")
    print("  running oracle...")
    orc = score(OracleAgent(), PUBLIC)
    ours = score(Agent(), PUBLIC)
    print(f"  public   ours {ours['recommended_technical_score']:.5f}   "
          f"oracle {orc['recommended_technical_score']:.5f}   "
          f"ratio {ours['recommended_technical_score'] / orc['recommended_technical_score']:.1%}")
    print("\nNote: the oracle's cheat sheet is built for the public sessions only,"
          "\nso a synthetic oracle needs eval/oracle.py generalised to take a"
          "\nsession file. Part 1 is the decisive test on its own.")


if __name__ == "__main__":
    main()