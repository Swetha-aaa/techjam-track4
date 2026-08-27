"""Robustness check: does our extraction survive a change in the simulator's
message templates?

Copies the official evaluator's session loop but rephrases the lead-ins the
simulated customer uses. The official evaluator is NOT modified — this is a
separate harness and its numbers are reported as such.
"""
import re

from evaluator import local_evaluator as ev
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent

SESSIONS = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")

# Alternative phrasings an organizer might plausibly use instead.
REWRITES = [
    (r"A key requirement is:", "One thing I care about:"),
    (r"For that, what matters is:", "Here's what's important to me:"),
    (r"What I need is:", "Scratch that — I actually want:"),
    (r"I'm looking for", "I need"),
    (r"I don't have an additional preference for", "No strong feelings about"),
    (r"I don't have a preference for", "No strong feelings about"),
]


def rewrite(msg):
    for pat, repl in REWRITES:
        msg = re.sub(pat, repl, msg, flags=re.I)
    return msg


def patched_evaluate(agent):
    """Wrap the evaluator's message producers so every utterance is rephrased."""
    orig_initial = ev.initial_message
    orig_reply = ev.customer_reply

    def initial_message(*a, **kw):
        return rewrite(orig_initial(*a, **kw))

    def customer_reply(*a, **kw):
        out = orig_reply(*a, **kw)
        if isinstance(out, tuple):
            return (rewrite(out[0]),) + out[1:]
        return rewrite(out)

    ev.initial_message = initial_message
    ev.customer_reply = customer_reply
    try:
        return evaluate(agent, SESSIONS, IDS, CATS, PRODS)
    finally:
        ev.initial_message = orig_initial
        ev.customer_reply = orig_reply


def main():
    print("official templates...")
    base = evaluate(Agent(), SESSIONS, IDS, CATS, PRODS)
    print("rephrased templates...")
    drift = patched_evaluate(Agent())

    print()
    for name, r in (("official", base), ("rephrased", drift)):
        print(f"  {name:12} HR {r['hit_rate_at_10']:.3f}  "
              f"MRR {r['mrr']:.3f}  score {r['recommended_technical_score']:.5f}")


if __name__ == "__main__":
    main()