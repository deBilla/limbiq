"""
Steering vector extraction via contrastive activation addition (CAA).

Given a pair of contrasting prompts, computes the direction in
activation space that separates them. This direction vector can
then be added to hidden states at inference time to steer behavior.

Requires: pip install mlx mlx-lm
"""

import json
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def _require_mlx():
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX is required for steering vector extraction. "
            "Install with: pip install limbiq[steering-mlx]"
        )


class SteeringVectorExtractor:
    """
    Extracts steering vectors from contrasting prompt pairs using MLX.

    llama-cpp-python only exposes final-layer embeddings, not per-layer
    hidden states. MLX provides full access to intermediate activations,
    making it the only viable backend for proper activation steering.
    """

    def __init__(self, model_path: str):
        _require_mlx()
        self.model, self.tokenizer = mlx_load(model_path)
        self.num_layers = len(self.model.model.layers)
        self._hidden_dim = None

    def extract(
        self,
        positive_prompt: str,
        negative_prompt: str,
        target_layers: list[int] = None,
        method: str = "mean_diff",
    ) -> dict:
        """
        Extract a steering vector from contrasting prompts.

        Args:
            positive_prompt: Prompt representing the desired behavior
            negative_prompt: Prompt representing the opposite behavior
            target_layers: Which layers to extract from (default: middle layers)
            method: "mean_diff" or "last_token_diff"

        Returns:
            dict with vectors per layer, metadata
        """
        _require_mlx()

        if target_layers is None:
            mid = self.num_layers // 2
            target_layers = list(range(mid - 2, mid + 3))

        pos_activations = self._get_activations(positive_prompt, target_layers)
        neg_activations = self._get_activations(negative_prompt, target_layers)

        vectors = {}
        for layer_idx in target_layers:
            pos = pos_activations[layer_idx]
            neg = neg_activations[layer_idx]

            if method == "mean_diff":
                vec = mx.mean(pos, axis=0) - mx.mean(neg, axis=0)
            elif method == "last_token_diff":
                vec = pos[-1] - neg[-1]
            else:
                raise ValueError(f"Unknown method: {method}")

            norm = mx.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors[layer_idx] = vec

        hidden_dim = int(vectors[target_layers[0]].shape[0])
        self._hidden_dim = hidden_dim

        return {
            "vectors": vectors,
            "target_layers": target_layers,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "method": method,
            "hidden_dim": hidden_dim,
            "num_layers": self.num_layers,
        }

    def _get_activations(self, prompt: str, target_layers: list[int]) -> dict:
        """Run a forward pass and capture hidden states at specified layers."""
        tokens = mx.array(self.tokenizer.encode(prompt))[None]

        activations = {}
        h = self.model.model.embed_tokens(tokens)

        # Build causal mask
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)

        # Create attention mask using the model's expected pattern
        mask = None
        seq_len = h.shape[1]
        if seq_len > 1:
            mask = mx.full((seq_len, seq_len), -mx.inf)
            mask = mx.triu(mask, k=1)
            mask = mask[None, None, :, :]  # (1, 1, seq_len, seq_len)

        for i, (layer, c) in enumerate(zip(self.model.model.layers, cache)):
            h = layer(h, mask, cache=c)
            if i in target_layers:
                activations[i] = h[0]  # Remove batch dim

        mx.eval(activations)
        return activations

    def save_vector(self, vector_data: dict, name: str, path: str = "./vectors"):
        """Save a steering vector to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)

        meta = {
            "name": name,
            "target_layers": vector_data["target_layers"],
            "positive_prompt": vector_data["positive_prompt"],
            "negative_prompt": vector_data["negative_prompt"],
            "method": vector_data["method"],
            "hidden_dim": vector_data["hidden_dim"],
            "num_layers": vector_data.get("num_layers"),
        }

        for layer_idx, vec in vector_data["vectors"].items():
            np_vec = np.array(vec)
            np.save(f"{path}/{name}_layer{layer_idx}.npy", np_vec)

        with open(f"{path}/{name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load_vector(name: str, path: str = "./vectors") -> dict:
        """Load a steering vector from disk."""
        with open(f"{path}/{name}_meta.json") as f:
            meta = json.load(f)

        vectors = {}
        for layer_idx in meta["target_layers"]:
            np_vec = np.load(f"{path}/{name}_layer{layer_idx}.npy")
            if MLX_AVAILABLE:
                vectors[layer_idx] = mx.array(np_vec)
            else:
                vectors[layer_idx] = np_vec

        meta["vectors"] = vectors
        return meta
