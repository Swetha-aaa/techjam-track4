"""One-time: encode every catalog product to a vector, cache to disk.

Run:  python -m src.build_embeddings
Takes ~2-3 minutes. Outputs are gitignored and regenerable.
"""
import json, time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CATALOG = "data/catalog.jsonl"
OUT_VECS = "data/embeddings.npy"
OUT_IDS = "data/embedding_ids.json"
BATCH = 256


def _text(v):
    if v is None:
        return ""
    if isinstance(v, dict):
        return " ".join(f"{k} {i}" for k, i in v.items())
    if isinstance(v, list):
        return " ".join(str(i) for i in v)
    return str(v)


def product_text(p):
    """Text representation used for semantic matching.

    Weighted toward features/details, since that is where the simulator draws
    its constraint phrases from.
    """
    parts = [
        _text(p.get("title")),
        _text(p.get("features")),
        _text(p.get("details")),
        _text(p.get("description"))[:500],
    ]
    return " ".join(x for x in parts if x)[:2000]


def main():
    print(f"loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    ids, texts = [], []
    for line in Path(CATALOG).open(encoding="utf-8"):
        p = json.loads(line)
        ids.append(str(p["parent_asin"]))
        texts.append(product_text(p))
    print(f"{len(ids)} products to encode")

    t0 = time.time()
    vecs = model.encode(texts, batch_size=BATCH, show_progress_bar=True,
                        normalize_embeddings=True)
    print(f"encoded in {time.time() - t0:.1f}s")

    np.save(OUT_VECS, vecs.astype(np.float32))
    with open(OUT_IDS, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    mb = Path(OUT_VECS).stat().st_size / 1e6
    print(f"wrote {OUT_VECS} ({vecs.shape}, {mb:.0f} MB) and {OUT_IDS}")


if __name__ == "__main__":
    main()