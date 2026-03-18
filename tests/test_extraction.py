"""Tests for steering vector extraction.

Tests that don't require MLX use mocks.
Tests that require MLX are skipped if MLX is not available.
"""

import json
import os
import tempfile
import shutil

import numpy as np
import pytest

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class TestSteeringVectorIO:
    """Test save/load of steering vectors (no MLX required)."""

    def test_save_and_load_vector(self, tmp_dir):
        """Vectors can be saved and loaded from disk."""
        from limbiq.steering.extractor import SteeringVectorExtractor

        # Create mock vector data
        vectors = {}
        for layer in [14, 15, 16]:
            vectors[layer] = np.random.randn(128).astype(np.float32)

        meta = {
            "vectors": vectors,
            "target_layers": [14, 15, 16],
            "positive_prompt": "Be concise.",
            "negative_prompt": "Be verbose.",
            "method": "mean_diff",
            "hidden_dim": 128,
            "num_layers": 32,
        }

        # Save
        vector_dir = os.path.join(tmp_dir, "vectors")
        # Manual save since we can't instantiate without MLX
        os.makedirs(vector_dir, exist_ok=True)
        save_meta = {k: v for k, v in meta.items() if k != "vectors"}
        with open(os.path.join(vector_dir, "test_meta.json"), "w") as f:
            json.dump(save_meta, f)
        for layer_idx, vec in vectors.items():
            np.save(os.path.join(vector_dir, f"test_layer{layer_idx}.npy"), vec)

        # Load
        loaded = SteeringVectorExtractor.load_vector("test", vector_dir)

        assert loaded["target_layers"] == [14, 15, 16]
        assert loaded["hidden_dim"] == 128
        assert loaded["method"] == "mean_diff"
        assert len(loaded["vectors"]) == 3

        for layer in [14, 15, 16]:
            loaded_vec = np.array(loaded["vectors"][layer])
            np.testing.assert_allclose(loaded_vec, vectors[layer], rtol=1e-5)

    def test_load_nonexistent_raises(self, tmp_dir):
        from limbiq.steering.extractor import SteeringVectorExtractor

        with pytest.raises(FileNotFoundError):
            SteeringVectorExtractor.load_vector("nonexistent", tmp_dir)


class TestContrastivePairs:
    """Test the predefined contrastive pair library."""

    def test_all_pairs_have_positive_and_negative(self):
        from limbiq.steering.library import CONTRASTIVE_PAIRS

        for name, pair in CONTRASTIVE_PAIRS.items():
            assert "positive" in pair, f"{name} missing positive"
            assert "negative" in pair, f"{name} missing negative"
            assert len(pair["positive"]) > 10, f"{name} positive too short"
            assert len(pair["negative"]) > 10, f"{name} negative too short"

    def test_memory_attention_pair_exists(self):
        from limbiq.steering.library import CONTRASTIVE_PAIRS

        assert "memory_attention" in CONTRASTIVE_PAIRS
        pair = CONTRASTIVE_PAIRS["memory_attention"]
        assert "context" in pair["positive"].lower()

    def test_eight_dimensions_available(self):
        from limbiq.steering.library import CONTRASTIVE_PAIRS

        assert len(CONTRASTIVE_PAIRS) == 8


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")
class TestMLXExtraction:
    """Integration tests requiring MLX and a model. Skipped in CI."""

    def test_extractor_initialization(self):
        """Test that extractor can be instantiated (requires model download)."""
        # This test would require an actual model path
        # Placeholder for when running locally with MLX
        pass
