from __future__ import annotations
import json, re, sqlite3
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOP = set("a an and are as at be but by for from i in is it me my of on or please "
           "some that the this to want with would you looking still exploring key "
           "requirement actually ignore earlier preference what need".split())


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
    def __init__(self, catalog_path="data/catalog.jsonl"):
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
        self.s = {}

    def reset(self, session_id, user_profile):
        self.s[session_id] = {"msgs": [], "phrases": [], "asked": []}

    def _search(self, phrases, msgs, k):
        clauses = []
        for ph in phrases:
            toks = terms(ph)[:12]
            if toks:
                clauses.append("(" + " AND ".join(f'"{t}"' for t in toks) + ")")
        rows = []
        rank = "bm25(products,0.0,6.0,4.0,2.5,2.5,1.5,1.0)"
        if clauses:
            expr = " OR ".join(clauses)
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
            tail = re.sub(r"^I'm looking for [^.]*\.\s*", "", user_message).strip()
            if tail and "still exploring" not in tail \
                    and "key requirement" not in tail.lower():
                st["phrases"].append(tail)

        return {
            "message": "Anything else that matters to you?",
            "ask_attribute": "other",
            "recommendations": self._search(st["phrases"], st["msgs"], top_k),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }