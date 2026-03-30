"""
Embedding-based semantic search.

Two modes:
1. sentence-transformers (default) -- good quality, runs locally
2. Fallback: keyword-based TF-IDF -- no ML dependency, worse quality
"""

import math
import threading
from collections import Counter


class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", custom_fn=None):
        self._cache = {}
        self._cache_max = 500
        self._lock = threading.Lock()
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
                self.embed = self._cached_transformer_embed
                self._mode = "transformer"
            except ImportError:
                self.embed = self._cached_tfidf_embed
                self._mode = "tfidf"
                self._vocab: dict[str, int] = {}
                self._vocab_size = 512

    def _cached_transformer_embed(self, text: str) -> list[float]:
        """Transformer embedding with caching (thread-safe)."""
        with self._lock:
            if text in self._cache:
                return self._cache[text]

            result = self._model.encode(text).tolist()

            # Cache result
            if len(self._cache) >= self._cache_max:
                # Remove oldest entry
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[text] = result
            return result

    def _cached_tfidf_embed(self, text: str) -> list[float]:
        """TF-IDF embedding with caching (thread-safe)."""
        with self._lock:
            if text in self._cache:
                return self._cache[text]

            result = self._tfidf_embed_impl(text)

            # Cache result
            if len(self._cache) >= self._cache_max:
                # Remove oldest entry
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[text] = result
            return result

    def _tfidf_embed_impl(self, text: str) -> list[float]:
        """Actual TF-IDF implementation."""
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

    def embed_tokens(self, text: str) -> list[list[float]]:
        """Per-token embeddings for whitespace-split tokens.

        Each token is embedded via the same engine (with caching).
        Used by the contextual relation classifier for self-attention
        over sentence tokens.
        """
        tokens = text.split()
        return [self.embed(token) for token in tokens]

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
