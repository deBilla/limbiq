"""
Steered inference using MLX.

Modifies the model's forward pass to inject steering vectors
at specified layers. The model generates text with its behavior
shifted by the active steering vectors.

The forward pass replicates mlx-lm's LlamaModel.__call__ exactly,
with one addition: after each target layer, the steering vector
is added to the hidden state.

Requires: pip install mlx mlx-lm
"""

try:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class SteeredModel:
    """
    Wraps an MLX model to support activation steering at inference time.

    Usage:
        model, tokenizer = mlx_lm.load("model_path")
        steered = SteeredModel(model, tokenizer)
        steered.add_steer("concise", vectors, alpha=1.0)
        text = steered.generate("Explain recursion.")
    """

    def __init__(self, model, tokenizer):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for steered inference.")
        self.model = model
        self.tokenizer = tokenizer
        self.active_steers: list[dict] = []

    def add_steer(self, name: str, vectors: dict, alpha: float = 1.0):
        """
        Add a steering vector to the active set.

        Args:
            name: Identifier for this steer
            vectors: Dict of {layer_idx: vector} from extraction
            alpha: Strength (0=none, 1=full, 2=strong, negative=reverse)
        """
        # Remove existing steer with same name
        self.active_steers = [s for s in self.active_steers if s["name"] != name]
        self.active_steers.append({
            "name": name,
            "vectors": vectors,
            "alpha": alpha,
        })

    def remove_steer(self, name: str):
        """Remove a steering vector by name."""
        self.active_steers = [s for s in self.active_steers if s["name"] != name]

    def clear_steers(self):
        """Remove all active steering vectors."""
        self.active_steers = []

    def get_active_steers(self) -> list[str]:
        """Return names of all active steers."""
        return [s["name"] for s in self.active_steers]

    def _steered_forward(self, tokens, cache):
        """
        Custom forward pass that injects steering vectors between layers.

        Replicates mlx-lm's LlamaModel.__call__ with steering additions.
        """
        h = self.model.model.embed_tokens(tokens)

        # Build causal mask
        seq_len = h.shape[1]
        mask = None
        if seq_len > 1:
            mask = mx.full((seq_len, seq_len), -mx.inf)
            mask = mx.triu(mask, k=1)

            # Account for cached sequence length
            if cache[0] is not None:
                cached_len = cache[0].offset
                if cached_len > 0:
                    pad = mx.zeros((seq_len, cached_len))
                    mask = mx.concatenate([pad, mask], axis=1)

            mask = mask[None, None, :, :]

        for i, (layer, c) in enumerate(zip(self.model.model.layers, cache)):
            # Determine correct mask for this layer
            layer_mask = mask
            h = layer(h, layer_mask, cache=c)

            # Inject steering vectors at this layer
            for steer in self.active_steers:
                if i in steer["vectors"]:
                    vec = steer["vectors"][i]
                    alpha = steer["alpha"]
                    # Add scaled vector to ALL token positions
                    # vec shape: (hidden_dim,) → broadcast to (1, seq_len, hidden_dim)
                    h = h + alpha * vec

        h = self.model.model.norm(h)
        return self.model.lm_head(h)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_tokens: list[int] = None,
    ) -> str:
        """
        Generate text with active steering vectors applied.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0=greedy)
            stop_tokens: Token IDs that stop generation
        """
        tokens = mx.array(self.tokenizer.encode(prompt))[None]
        cache = make_prompt_cache(self.model)

        if stop_tokens is None:
            stop_tokens = []
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                stop_tokens.append(self.tokenizer.eos_token_id)

        # Prefill: process entire prompt
        logits = self._steered_forward(tokens, cache)
        mx.eval(logits, *[c.state for c in cache])

        generated = []

        for _ in range(max_tokens):
            # Sample from last token's logits
            last_logits = logits[:, -1, :]

            if temperature > 0:
                probs = mx.softmax(last_logits / temperature, axis=-1)
                next_token = mx.random.categorical(probs)
            else:
                next_token = mx.argmax(last_logits, axis=-1)

            token_id = next_token.item()

            if token_id in stop_tokens:
                break

            generated.append(token_id)

            # Next step: feed single token
            next_input = next_token[:, None]
            logits = self._steered_forward(next_input, cache)
            mx.eval(logits, *[c.state for c in cache])

        return self.tokenizer.decode(generated)
