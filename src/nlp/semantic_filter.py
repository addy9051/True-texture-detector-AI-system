"""Isolate review sentences that talk about fabric feel, texture, or thermal comfort.

Embedding-based aspect filtering (deliberate upgrade over LDA topic modeling):
each review is split into sentences, embedded with a local MiniLM model, and
scored against anchor phrases describing the tactile/thermal experience of a
garment. Runs entirely on CPU, no cloud cost.

The 0.50 default threshold was calibrated 2026-07-03 on 48,040 sentences from
17,754 real Amazon Fashion reviews (scripts/calibrate_threshold.py): precision
jumps from ~50% in the 0.45-0.50 band (generic "very comfortable" chatter) to
~90% at 0.50-0.55 (actual fabric-feel statements). Above 0.50 the filter keeps
~3.6% of all sentences. Recalibrate if the anchor set or model changes.

For Indian review data (Phase 1b) — reviews mix English, Hinglish, and regional
languages — pass `model_name=MULTILINGUAL_MODEL_NAME` when constructing
TextureFilter. The English anchors below still work: the multilingual model maps
"kapda bilkul plastic jaisa hai" near "it feels like plastic". Recalibrate the
threshold when switching models; cross-lingual similarities run lower.
"""

import re
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TEXTURE_ANCHORS = [
    "how the fabric feels against the skin",
    "the material feels cheap and scratchy",
    "the fabric is soft, breathable and comfortable",
    "it feels like plastic or fake synthetic material",
    "this shirt made me sweat, not breathable at all",
    "the fabric is thick, heavy and stiff",
    "the material is thin and see-through",
    "the texture looks different than in the product photos",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_HTML_TAG = re.compile(r"<br\s*/?>|</?\w+[^>]*>")


@dataclass
class TextureSentence:
    parent_asin: str
    sentence: str
    similarity: float
    rating: Optional[float]

    def to_dict(self):
        return asdict(self)


def split_sentences(text: str, min_len: int = 15, max_len: int = 400) -> list[str]:
    text = _HTML_TAG.sub(" ", text or "")
    return [s.strip() for s in _SENT_SPLIT.split(text)
            if min_len <= len(s.strip()) <= max_len]


class TextureFilter:
    def __init__(self, threshold: float = 0.50, model_name: str = MODEL_NAME):
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)
        self.anchor_emb = self.model.encode(TEXTURE_ANCHORS, normalize_embeddings=True)

    def filter_reviews(self, reviews: Iterable[dict]) -> list[TextureSentence]:
        """Return every sentence across `reviews` that is about fabric feel."""
        rows = [(r, s) for r in reviews for s in split_sentences(r.get("text", ""))]
        if not rows:
            return []
        emb = self.model.encode([s for _, s in rows], normalize_embeddings=True,
                                batch_size=64, show_progress_bar=True)
        best = util.cos_sim(emb, self.anchor_emb).max(dim=1).values
        return [
            TextureSentence(r.get("parent_asin", ""), s, round(float(score), 3), r.get("rating"))
            for (r, s), score in zip(rows, best)
            if float(score) >= self.threshold
        ]
