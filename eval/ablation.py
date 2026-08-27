"""Run configs through the evaluator and regenerate RESULTS.md."""
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent
from starter.agent_baseline import Agent as BaselineAgent

SESSIONS = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")

SCENARIOS = ["buying", "browsing", "intent_override", "boundary"]


def run(agent):
    return evaluate(agent, SESSIONS, IDS, CATS, PRODS)


def fmt_table(headers, rows):
    """Build a markdown table with columns padded to equal width."""
    cols = [headers] + rows
    widths = [max(len(str(r[i])) for r in cols) for i in range(len(headers))]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return out


def main():
    results = {}
    print("running baseline...")
    results["BM25 baseline (organizer)"] = run(BaselineAgent())
    print("running ours...")
    results["Ours"] = run(Agent())

    lines = ["# Results (200 public dev sessions)", ""]

    main_rows = [[name, f"{r['hit_rate_at_10']:.3f}", f"{r['mrr']:.3f}",
                  f"{r['mttc']:.2f}", f"{r['recommended_technical_score']:.5f}"]
                 for name, r in results.items()]
    lines += fmt_table(["Config", "HR@10", "MRR", "MTTC", "Score"], main_rows)

    lines += ["", "## Per-scenario (ours)", ""]
    sm = results["Ours"]["scenario_metrics"]
    scen_rows = [[s, str(sm[s]["sample_count"]), f"{sm[s]['hit_rate_at_10']:.3f}",
                  f"{sm[s]['mrr']:.3f}", f"{sm[s]['mttc']:.2f}"] for s in SCENARIOS]
    lines += fmt_table(["Scenario", "n", "HR@10", "MRR", "MTTC"], scen_rows)

    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for name, r in results.items():
        print(f"  {name:32} {r['recommended_technical_score']:.5f}")
    print("\nwrote RESULTS.md")

if __name__ == "__main__":
    main()