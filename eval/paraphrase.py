"""Paraphrase harness: does the semantic layer earn its place when the customer
stops quoting catalog text verbatim?

The official simulator lifts constraint strings straight out of the target
product's `features` and `details`. A real shopper would not. This harness
rewrites each disclosed constraint into natural phrasing before it reaches the
agent, then compares exact-match retrieval against exact-match + semantic
reranking under both conditions.

The official evaluator is NOT modified. Figures from this harness are reported
separately from official scores.
"""
import re

from evaluator import local_evaluator as ev
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent

SESSIONS = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")

# --- paraphrase rules --------------------------------------------------------
# Hand-written natural phrasings for the most frequent constraints. Covers the
# common head of the distribution; the regex families below cover structured
# variants. Anything unmatched is left as-is and counted, so coverage is known.
PARAPHRASE = {
    "Imported": "not made locally, it's shipped in from overseas",
    "cotton": "made of cotton",
    "polyester": "a polyester material",
    "leather": "real leather",
    "nylon": "nylon material",
    "rayon": "rayon fabric",
    "wool": "woollen",
    "silk": "silky material",
    "mesh": "breathable mesh",
    "spandex": "a bit of stretch to it",
    "fabric": "the material matters to me",
    "Rubber sole": "the sole should be rubber",
    "Leather sole": "leather on the sole",
    "Synthetic sole": "a synthetic sole",
    "Manmade sole": "man-made sole",
    "Hand Wash Only": "I'd need to wash it by hand",
    "Machine Wash": "something I can throw in the washing machine",
    "Made in the USA": "made in America",
    "Made in the USA or Imported": "either American-made or brought in from abroad",
    "cotton blend": "a cotton mix",
    "Faux Fur": "fake fur",
    "Satin": "satiny finish",
    "Elastic": "stretchy elastic",
    "Textile": "a textile upper",
    "PU": "faux leather",
    "PU Leather": "synthetic leather",
    "100% Leather": "entirely leather",
    "100% Cotton": "pure cotton, nothing mixed in",
    "100% Polyester": "all polyester",
    "100% Nylon": "completely nylon",
    "100% Rayon": "all rayon",
    "100% Canvas": "canvas throughout",
    "100% Mesh": "mesh all over",
    "100% Synthetic": "fully synthetic",
    "Department: womens": "it's a women's item",
    "Department: mens": "it's a men's item",
}

# Structured families, applied when no exact entry matches.
FAMILY = [
    # "95% Polyester, 5% Spandex" -> "mostly polyester with a little spandex"
    (re.compile(r"^(\d+)%\s*([A-Za-z]+)[,;]\s*(\d+)%\s*([A-Za-z]+)\.?$"),
     lambda m: f"mostly {m.group(2).lower()} with a bit of {m.group(4).lower()} in it"),
    # "100% Cotton" style singletons not in the dict
    (re.compile(r"^(\d+)%\s*([A-Za-z ]+)\.?$"),
     lambda m: (f"entirely {m.group(2).lower()}" if m.group(1) == "100"
                else f"mostly {m.group(2).lower()}")),
    # "color: black" -> "in black"
    (re.compile(r"^colou?r:\s*(.+)$", re.I),
     lambda m: f"in {m.group(1).strip()}"),
    # "Pull On closure" -> "something you pull on rather than fasten"
    (re.compile(r"^(.+?)\s+closure$", re.I),
     lambda m: f"it fastens with {m.group(1).lower()}"),
    # "Ethylene Vinyl Acetate sole" -> "the sole is ..."
    (re.compile(r"^(.+?)\s+sole$", re.I),
     lambda m: f"the sole is {m.group(1).lower()}"),
    # "Stretchy fabric: 95% modal, 5% spandex" -> strip the label, paraphrase rest
    (re.compile(r"^([A-Za-z][A-Za-z ]{0,20}):\s*(\d+)%\s*([A-Za-z]+)"
                r"(?:[,;]\s*(\d+)%\s*([A-Za-z]+))?"),
     lambda m: (f"mostly {m.group(3).lower()} with a bit of {m.group(5).lower()}"
                if m.group(5) else f"mostly {m.group(3).lower()}")),
    # "Material:alloy" -> "made of alloy"
    (re.compile(r"^Material:\s*(.+)$", re.I),
     lambda m: f"made of {m.group(1).strip().lower()}"),
]

STATS = {"seen": 0, "dict": 0, "family": 0, "unmatched": 0}


def paraphrase_one(phrase):
    p = phrase.strip().rstrip(".")
    STATS["seen"] += 1
    if p in PARAPHRASE:
        STATS["dict"] += 1
        return PARAPHRASE[p]
    for pat, fn in FAMILY:
        m = pat.match(p)
        if m:
            STATS["family"] += 1
            return fn(m)
    STATS["unmatched"] += 1
    return phrase


def paraphrase_message(msg):
    """Rewrite only the constraint body (after the first colon)."""
    m = re.match(r"^([^:]*:\s*)(.+?)(\.?)$", msg)
    if not m:
        return msg
    head, body, tail = m.groups()
    parts = [paraphrase_one(p) for p in body.split(";") if p.strip()]
    return head + "; ".join(parts) + tail
# -----------------------------------------------------------------------------


def patched_evaluate(agent):
    orig_initial, orig_reply = ev.initial_message, ev.customer_reply

    def initial_message(*a, **kw):
        return paraphrase_message(orig_initial(*a, **kw))

    def customer_reply(*a, **kw):
        out = orig_reply(*a, **kw)
        if isinstance(out, tuple):
            return (paraphrase_message(out[0]),) + out[1:]
        return paraphrase_message(out)

    ev.initial_message, ev.customer_reply = initial_message, customer_reply
    try:
        return evaluate(agent, SESSIONS, IDS, CATS, PRODS)
    finally:
        ev.initial_message, ev.customer_reply = orig_initial, orig_reply


EXACT = {}
SEMANTIC = {"use_rerank": True, "fts_weight": 2.0}


def main():
    out = {}
    print("verbatim / exact-match...")
    out[("verbatim", "exact")] = evaluate(Agent(config=EXACT), SESSIONS, IDS, CATS, PRODS)
    print("verbatim / + semantic...")
    out[("verbatim", "semantic")] = evaluate(Agent(config=SEMANTIC), SESSIONS, IDS, CATS, PRODS)

    STATS.update({"seen": 0, "dict": 0, "family": 0, "unmatched": 0})
    print("paraphrased / exact-match...")
    out[("paraphrased", "exact")] = patched_evaluate(Agent(config=EXACT))
    cov = STATS.copy()
    print("paraphrased / + semantic...")
    out[("paraphrased", "semantic")] = patched_evaluate(Agent(config=SEMANTIC))

    print()
    print(f"{'condition':14} {'retrieval':12} {'HR':>6} {'MRR':>6} {'MTTC':>6} {'score':>8}")
    for (cond, ret), r in out.items():
        print(f"{cond:14} {ret:12} {r['hit_rate_at_10']:6.3f} {r['mrr']:6.3f} "
              f"{r['mttc']:6.2f} {r['recommended_technical_score']:8.5f}")

    tot = cov["seen"] or 1
    print(f"\nparaphrase coverage: {(cov['dict'] + cov['family']) / tot:.1%} "
          f"({cov['dict']} dict, {cov['family']} family, {cov['unmatched']} unchanged)")


if __name__ == "__main__":
    main()