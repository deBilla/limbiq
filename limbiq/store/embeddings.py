"""
Embedding-based semantic search.

Two modes:
1. sentence-transformers (default) -- good quality, runs locally
2. Fallback: keyword-based TF-IDF -- no ML dependency, worse quality
"""

import math
from collections import Counter


class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", custom_fn=None):
        if custom_fn:
            self.embed = custom_fn
            self._mode = "custom"
        else:
            try:
                from sentence_transformers import SentenceTransformer
                import platform

                # Force CPU on Apple Silicon — MPS (Metal) conflicts with MLX
                # which owns the GPU for LLM inference.
                device = "cpu" if platform.machine() == "arm64" else None
                self._model = SentenceTransformer(model_name, device=device)
                self.embed = self._transformer_embed
                self._mode = "transformer"
            except ImportError:
                self.embed = self._tfidf_embed
                self._mode = "tfidf"
                self._vocab: dict[str, int] = {}
                self._vocab_size = 512

    def _transformer_embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def _tfidf_embed(self, text: str) -> list[float]:
        """Simple bag-of-words fallback when sentence-transformers isn't installed."""
        tokens = text.lower().split()
        counts = Counter(tokens)

        for token in tokens:
            if token not in self._vocab and len(self._vocab) < self._vocab_size:
                self._vocab[token] = len(self._vocab)

        vector = [0.0] * self._vocab_size
        for token, count in counts.items():
            if token in self._vocab:
                vector[self._vocab[token]] = float(count)

        # Normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def similarity(self, embedding_a: list[float], embedding_b: list[float]) -> float:
        """Cosine similarity between two embeddings."""
        if len(embedding_a) != len(embedding_b):
            return 0.0

        dot = sum(a * b for a, b in zip(embedding_a, embedding_b))
        norm_a = math.sqrt(sum(a * a for a in embedding_a))
        norm_b = math.sqrt(sum(b * b for b in embedding_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)
