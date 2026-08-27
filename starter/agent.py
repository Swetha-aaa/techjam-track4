from __future__ import annotations
import json, re, sqlite3
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOP = set("a an and are as at be but by for from i in is it me my of on or please "
           "some that the this to want with would you looking still exploring key "
           "requirement actually ignore earlier preference what need".split())

# --- default configuration ---------------------------------------------------
# Override any of these by passing a dict: Agent(config={"use_category_filter": False})
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

        self.s = {}

    def reset(self, session_id, user_profile):
        self.s[session_id] = {"msgs": [], "phrases": [], "asked": [],
                              "category": None}

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
            if toks:
                clauses.append("(" + " AND ".join(f'"{t}"' for t in toks) + ")")

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

        return [{"parent_asin": str(r[0])} for r in rows[:k]]

    def respond(self, session_id, user_message, turn, top_k):
        st = self.s[session_id]
        st["msgs"].append(user_message)

        m = re.search(r"what matters is:\s*(.+?)\.?$", user_message, re.I)
        if m:
            st["phrases"].extend(p.strip() for p in m.group(1).split(";") if p.strip())
        m = re.search(r"A key requirement is:\s*(.+?)\.?$", user_message, re.I)
        if m:
            st["phrases"].append(m.group(1).strip())
        m = re.search(r"What I need is:\s*(.+?)\.?$", user_message, re.I)
        if m:
            st["phrases"] = [m.group(1).strip()] + st["phrases"]

        if turn == 1:
            m = re.match(r"I'm looking for ([^.]+)\.", user_message, re.I)
            if m:
                st["category"] = m.group(1).strip()
            tail = re.sub(r"^I'm looking for [^.]*\.\s*", "", user_message).strip()
            if tail and "still exploring" not in tail \
                    and "key requirement" not in tail.lower():
                st["phrases"].append(tail)

        return {
            "message": "Anything else that matters to you?",
            "ask_attribute": "other",
            "recommendations": self._search(st["phrases"], st["msgs"],
                                            top_k, st.get("category")),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }