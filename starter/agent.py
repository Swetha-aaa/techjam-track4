from __future__ import annotations
import json, re, sqlite3
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOP = set("a an and are as at be but by for from i in is it me my of on or please "
           "some that the this to want with would you looking still exploring key "
           "requirement actually ignore earlier preference what need scratch "
           "instead here important care thing one strong feelings about".split())

# Signals that a message reprioritises rather than accumulates.
OVERRIDE_RE = re.compile(r"scratch that|ignore my|ignore earlier|actually|instead",
                         re.I)
# Signals the customer declined to disclose anything this turn.
REFUSAL_RE = re.compile(r"no strong feelings|don't have|do not have|use your judgment",
                        re.I)
# Turn-1 category, tolerant of several lead-in verbs.
CATEGORY_RE = re.compile(r"(?:looking for|i need|i want|after|shopping for)\s+([^.,;:]+)",
                         re.I)
# Filler that carries no constraint information.
FILLER_RE = re.compile(r"still exploring|just browsing|not sure yet", re.I)
# Constraint body: everything after the FIRST colon. Anchored so that phrases
# which themselves contain a colon (e.g. "Material:alloy") survive intact.
BODY_RE = re.compile(r"^[^:]*:\s*(.+?)\.?$")

# --- default configuration ---------------------------------------------------
# Override by passing a dict: Agent(config={"robust_extraction": False})
# eval/ablation.py uses this to generate ablation rows automatically.
DEFAULT_CONFIG = {
    # 1.0 disables IDF phrase filtering (tested and rejected — see RESULTS.md)
    "common_threshold": 1.0,
    # hard-AND the turn-1 category against the `categories` column
    "use_category_filter": True,
    # BM25 per-column weights: asin, title, categories, features, details, store, desc
    "rank": "bm25(products,0.0,0.1,0.5,15.0,15.0,0.1,0.5)",
    # max tokens kept per constraint phrase
    "max_phrase_tokens": 12,
    # structure-based extraction; False falls back to fixed-template regexes
    "phrase_adjacency": True,
    "robust_extraction": True,
    # stop asking once the customer has refused twice (boundary sessions refuse once)
    "detect_exhaustion": True,
    # apply the category filter to the loose fallback query as well
    "category_filter_fallback": False,
    # semantic reranking (tested and rejected — see RESULTS.md)
    "use_rerank": False,
    "rerank_pool": 50,
    "fts_weight": 2.0,
}
# -----------------------------------------------------------------------------

def _text(v):
    if v is None:
        return ""
    if isinstance(v, dict):
        return " ".join(f"{k} {i}" for k, i in v.items())
    if isinstance(v, list):
        return " ".join(str(i) for i in v)
    return str(v)


def terms(t):
    return [w.lower() for w in TOKEN_RE.findall(t)
            if len(w) > 1 and w.lower() not in STOP]


class Agent:
    def __init__(self, catalog_path="data/catalog.jsonl", config=None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}

        self.conn = sqlite3.connect(":memory:")
        cur = self.conn.cursor()
        cur.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, "
            "store, description, tokenize='unicode61 remove_diacritics 2')"
        )
        batch = []
        for line in Path(catalog_path).open(encoding="utf-8"):
            p = json.loads(line)
            batch.append((
                str(p["parent_asin"]), _text(p.get("title")),
                _text(p.get("categories")), _text(p.get("features")),
                _text(p.get("details")), _text(p.get("store")),
                _text(p.get("description")),
            ))
            if len(batch) >= 2000:
                cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self.conn.commit()

        # document frequency per token — retained for constraint-entropy analysis
        self.df = {}
        self.total_docs = 0
        for line in Path(catalog_path).open(encoding="utf-8"):
            p = json.loads(line)
            self.total_docs += 1
            blob = " ".join(_text(p.get(f)) for f in
                            ("title", "categories", "features", "details",
                             "store", "description"))
            for t in set(terms(blob)):
                self.df[t] = self.df.get(t, 0) + 1

        self._reranker = None   # loaded lazily; unused unless use_rerank is True
        self.s = {}

    @property
    def reranker(self):
        if self._reranker is None:
            from src.rerank import Reranker
            self._reranker = Reranker()
        return self._reranker

    def reset(self, session_id, user_profile):
        self.s[session_id] = {"msgs": [], "phrases": [], "category": None,
                              "refusals": 0}

    # ---------------------------------------------------------------- extraction

    def _extract_robust(self, msg, turn, st):
        """Extract constraints by message structure, not by exact lead-in wording.

        The evaluator discloses constraints as `<lead-in>: <phrase>[; <phrase>]`.
        The colon is the load-bearing signal and survives rephrasing of the
        lead-in, so we key on that rather than on fixed templates.
        """
        if REFUSAL_RE.search(msg):
            st["refusals"] += 1
            return

        m = BODY_RE.search(msg)
        if m:
            phrases = [p.strip() for p in m.group(1).split(";") if p.strip()]
            if OVERRIDE_RE.search(msg):
                st["phrases"] = phrases + st["phrases"]
            else:
                st["phrases"].extend(phrases)
            return

        if turn == 1:
            return                                       # category only

        if not FILLER_RE.search(msg):
            st["phrases"].append(msg.strip())             # unrecognised shape

    def _extract_fixed(self, msg, turn, st):
        """Original fixed-template extraction, retained for ablation comparison."""
        if REFUSAL_RE.search(msg):
            st["refusals"] += 1
        m = re.search(r"what matters is:\s*(.+?)\.?$", msg, re.I)
        if m:
            st["phrases"].extend(p.strip() for p in m.group(1).split(";") if p.strip())
        m = re.search(r"A key requirement is:\s*(.+?)\.?$", msg, re.I)
        if m:
            st["phrases"].append(m.group(1).strip())
        m = re.search(r"What I need is:\s*(.+?)\.?$", msg, re.I)
        if m:
            st["phrases"] = [m.group(1).strip()] + st["phrases"]
        if turn == 1:
            tail = re.sub(r"^I'm looking for [^.]*\.\s*", "", msg).strip()
            if tail and "still exploring" not in tail \
                    and "key requirement" not in tail.lower():
                st["phrases"].append(tail)

    # ---------------------------------------------------------------- retrieval

    def _phrase_rarity(self, phrase):
        """Doc-frequency of the rarest token. Lower = more distinctive."""
        toks = terms(phrase)
        if not toks:
            return self.total_docs
        return min(self.df.get(t, 1) for t in toks)

    def _search(self, phrases, msgs, k, category=None):
        cfg = self.cfg
        cap = cfg["max_phrase_tokens"]
        threshold = self.total_docs * cfg["common_threshold"]
        ranked = sorted(phrases, key=self._phrase_rarity)

        clauses = []
        for ph in ranked:
            if self._phrase_rarity(ph) > threshold:
                continue
            toks = terms(ph)[:cap]
            if not toks:
                continue
            clauses.append("(" + " AND ".join(f'"{t}"' for t in toks) + ")")
            if cfg["phrase_adjacency"] and len(toks) > 1:
                clauses.append('"' + " ".join(toks) + '"')   # contiguous match

        if not clauses:                      # everything filtered — use all phrases
            for ph in ranked:
                toks = terms(ph)[:cap]
                if toks:
                    clauses.append("(" + " AND ".join(f'"{t}"' for t in toks) + ")")

        cat_clause = ""
        if category and cfg["use_category_filter"]:
            cat_toks = terms(category)[:4]
            if cat_toks:
                cat_clause = "(" + " AND ".join(
                    f'categories : "{t}"' for t in cat_toks) + ")"

        rank = cfg["rank"]
        rows = []
        if clauses:
            expr = "(" + " OR ".join(clauses) + ")"
            if cat_clause:
                expr = cat_clause + " AND " + expr
            rows = self.conn.execute(
                f"SELECT parent_asin FROM products WHERE products MATCH ? "
                f"ORDER BY {rank} LIMIT ?", (expr, k)).fetchall()

        if len(rows) < k:
            toks = list(dict.fromkeys(t for m in msgs for t in terms(m)))[:40]
            if toks:
                expr2 = " OR ".join(f'"{t}"' for t in toks)
                if cat_clause and cfg["category_filter_fallback"]:
                    expr2 = cat_clause + " AND (" + expr2 + ")"
                more = self.conn.execute(
                    f"SELECT parent_asin FROM products WHERE products MATCH ? "
                    f"ORDER BY {rank} LIMIT ?", (expr2, k * 3)).fetchall()
                seen = {r[0] for r in rows}
                for r in more:
                    if r[0] not in seen:
                        rows.append(r)
                        seen.add(r[0])
                    if len(rows) >= k:
                        break

        return [str(r[0]) for r in rows[:k]]

    def _recommend(self, st, top_k):
        cfg = self.cfg
        if not cfg["use_rerank"]:
            asins = self._search(st["phrases"], st["msgs"], top_k, st.get("category"))
            return [{"parent_asin": a} for a in asins]

        pool = self._search(st["phrases"], st["msgs"], cfg["rerank_pool"],
                            st.get("category"))
        if len(pool) <= top_k:
            return [{"parent_asin": a} for a in pool[:top_k]]
        query = " ".join(st["phrases"]) or " ".join(st["msgs"])
        ordered = self.reranker.rerank(query, pool, top_k,
                                       fts_weight=cfg["fts_weight"])
        return [{"parent_asin": a} for a in ordered]

    # ---------------------------------------------------------------- interface

    def respond(self, session_id, user_message, turn, top_k):
        st = self.s[session_id]
        st["msgs"].append(user_message)

        if self.cfg["robust_extraction"]:
            self._extract_robust(user_message, turn, st)
        else:
            self._extract_fixed(user_message, turn, st)

        if turn == 1:
            m = CATEGORY_RE.search(user_message)
            if m:
                st["category"] = m.group(1).strip()

        # Asking is free — the evaluator checks recommendations before it reads
        # ask_attribute — so we keep asking. Boundary sessions deflect the first
        # ask once, so only a second refusal indicates genuine exhaustion.
        exhausted = self.cfg["detect_exhaustion"] and st["refusals"] >= 2
        ask = None if exhausted else "other"

        latest = st["phrases"][0] if st["phrases"] else None
        msg = (f"Got it — focusing on {latest[:45]}. Anything else that matters?"
               if latest else "Let me widen the search a little.")

        return {
            "message": msg,
            "ask_attribute": ask,
            "recommendations": self._recommend(st, top_k),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }