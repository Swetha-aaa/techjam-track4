"""Print one session turn by turn: what the customer said, what we extracted,
what query we built, what came back, and where the target actually ranked.

Every other figure in this repository is an aggregate. This is the tool for
looking at a single session — for demo walkthroughs, for error analysis, and for
answering "show me one you lose and explain why".

    python -m eval.transcript                  # first session
    python -m eval.transcript public_0010      # by sample_id
    python -m eval.transcript --miss           # first session we fail
    python -m eval.transcript --miss 3         # fourth session we fail
    python -m eval.transcript --slow           # a session that takes many turns

The evaluator is driven directly rather than reimplemented, so the transcript is
exactly the conversation that produced our score.
"""
import collections
import sys

from evaluator import local_evaluator as ev
from evaluator.local_evaluator import (evaluate, load_jsonl, catalog_index,
                                       materialize_hidden_fields, coarse_category)
from starter.agent import Agent

SESSIONS = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")

W = 78


def rule(char="-"):
    print(char * W)


def wrap(text, indent=0, width=W):
    """Simple word wrap so long marketing constraints stay readable."""
    pad = " " * indent
    line, out = pad, []
    for word in str(text).split():
        if len(line) + len(word) + 1 > width:
            out.append(line.rstrip())
            line = pad + word + " "
        else:
            line += word + " "
    if line.strip():
        out.append(line.rstrip())
    return "\n".join(out)


def pick_session(argv):
    """Resolve the command-line selector to one sample record."""
    if argv and not argv[0].startswith("--"):
        for s in SESSIONS:
            if s["sample_id"] == argv[0]:
                return s
        sys.exit(f"no session with sample_id {argv[0]!r}")

    if not argv:
        return SESSIONS[0]

    print("scoring the public set to locate the session...\n")
    result = evaluate(Agent(), SESSIONS, IDS, CATS, PRODS)
    rows = list(zip(SESSIONS, result["sessions"]))
    nth = int(argv[1]) if len(argv) > 1 else 0

    if argv[0] == "--miss":
        pool = [s for s, r in rows if not r.get("best_rank")]
        if not pool:
            sys.exit("no missed sessions")
    elif argv[0] == "--slow":
        pool = [s for s, r in
                sorted(rows, key=lambda x: -(x[1].get("first_hit_turn") or 11))]
    else:
        sys.exit(f"unknown option {argv[0]!r}")
    return pool[min(nth, len(pool) - 1)]


def main():
    sample = pick_session(sys.argv[1:])
    target = sample["ground_truth"]["parent_asin"]
    product = PRODS[target]
    card, behavior = materialize_hidden_fields(sample, PRODS)
    constraints = card["hard_constraints"] + card["soft_preferences"]

    agent = Agent()

    rule("=")
    print(f"SESSION {sample['sample_id']}   scenario: {sample['scenario_type']}"
          f"   difficulty: {sample.get('difficulty_bucket', '?')}")
    rule("=")
    print("\nHIDDEN TARGET (the agent never sees this)\n")
    print(f"  asin      {target}")
    print(wrap(f"title     {product.get('title', '')}", 0))
    print(f"  category  {coarse_category(CATS.get(target, []))}")
    print("\n  the four constraints the simulator will disclose, each lifted")
    print("  verbatim from this product's own features/details:\n")
    for c in constraints:
        rarity = agent._phrase_rarity(c)
        print(wrap(f"    [{rarity:>6} products match] {c}", 0))
    if behavior.get("override"):
        o = behavior["override"]
        print(f"\n  override fires on turn {o['turn']}: "
              f"{o['old_value']!r} -> {o['new_value']!r}")

    # drive the evaluator's own session loop, one turn at a time
    disclosed = set()
    boundary_used = False
    session_id = "transcript"
    agent.reset(session_id, sample["user_profile"])
    effective = {**sample, "intent_card": card, "behavior": behavior}
    category = coarse_category(CATS.get(target, []))

    message = ev.initial_message(effective, category, disclosed)
    override_applied = sample["scenario_type"] != "intent_override"

    print()
    rule("=")
    print("TRANSCRIPT")
    rule("=")

    for turn in range(1, 11):
        override = behavior.get("override")
        if override and turn == override["turn"]:
            message = override["message"]
            override_applied = True

        print(f"\nTURN {turn}")
        rule()
        print(wrap(f"customer   {message}", 0))

        before = list(agent.s[session_id]["phrases"])
        response = agent.respond(session_id, message, turn, 10)
        st = agent.s[session_id]

        # Compare by multiset, not by slice: the override branch PREPENDS, so
        # slicing from the end reports the wrong phrase as newly extracted.
        gained = list((collections.Counter(st["phrases"])
                       - collections.Counter(before)).elements())

        if gained:
            print("\n  extracted this turn:")
            for g in gained:
                print(wrap(f"    + {g}", 0))
        else:
            print("\n  extracted this turn: (nothing)")

        print(f"\n  state      category={st['category']!r}  "
              f"phrases={len(st['phrases'])}  refusals={st['refusals']}")
        print(f"  asking     {response['ask_attribute']!r}")

        recs = [r["parent_asin"] for r in response["recommendations"]]
        rank = recs.index(target) + 1 if target in recs else None
        print(f"\n  returned {len(recs)} products"
              + (f", TARGET AT RANK {rank}" if rank else ", target not present"))
        for i, asin in enumerate(recs[:5], 1):
            mark = "  <-- TARGET" if asin == target else ""
            title = PRODS.get(asin, {}).get("title", "")[:52]
            print(f"    {i:>2}. {asin}  {title}{mark}")
        if len(recs) > 5:
            print(f"    ... {len(recs) - 5} more")

        if rank and override_applied:
            print()
            rule("=")
            print(f"HIT on turn {turn} at rank {rank}   "
                  f"reciprocal rank {1 / rank:.3f}")
            rule("=")
            return
        if rank and not override_applied:
            print("\n  (hit ignored — the override has not fired yet)")

        message, boundary_used = ev.customer_reply(
            effective, response.get("ask_attribute"), disclosed, boundary_used)

    print()
    rule("=")
    print("MISS — ten turns elapsed without the target entering the top 10")
    rule("=")
    print("\nwhy: compare the rarity figures above. When every constraint the")
    print("customer can disclose matches thousands of products, the transcript")
    print("does not contain enough information to identify one item. Our oracle,")
    print("given all four constraints on turn 1, misses these sessions too.")


if __name__ == "__main__":
    main()