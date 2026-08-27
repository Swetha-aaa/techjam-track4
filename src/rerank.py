"""Semantic reranking over FTS candidates using cached product embeddings."""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECS_PATH = "data/embeddings.npy"
IDS_PATH = "data/embedding_ids.json"


class Reranker:
    def __init__(self, model_name=MODEL_NAME,
                 vecs_path=VECS_PATH, ids_path=IDS_PATH):
        self.model = SentenceTransformer(model_name)
        self.vecs = np.load(vecs_path)
        ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
        self.row_of = {asin: i for i, asin in enumerate(ids)}

    def score(self, query_text, asins):
        """Cosine similarity between the query and each candidate.

        Embeddings are pre-normalised, so a dot product is the cosine.
        Unknown ASINs score 0.
        """
        q = self.model.encode([query_text], normalize_embeddings=True)[0]
        rows = [self.row_of.get(a) for a in asins]
        out = np.zeros(len(asins), dtype=np.float32)
        known = [(i, r) for i, r in enumerate(rows) if r is not None]
        if known:
            idx, r = zip(*known)
            out[list(idx)] = self.vecs[list(r)] @ q
        return out

    def rerank(self, query_text, asins, k, fts_weight=0.0):
        """Reorder candidates by semantic score.

        fts_weight blends in the original FTS ordering: 0.0 ignores it,
        higher values preserve more of the lexical ranking.
        """
        if not asins:
            return []
        sem = self.score(query_text, asins)
        n = len(asins)
        # original position as a 1..0 signal
        pos = np.linspace(1.0, 0.0, n, dtype=np.float32)
        combined = sem + fts_weight * pos
        order = np.argsort(-combined)
        return [asins[i] for i in order[:k]]