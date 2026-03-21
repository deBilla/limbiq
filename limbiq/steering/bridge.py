"""
Bridge between limbiq's signal system and activation steering.

Maps neurotransmitter signals to steering vector injections:

    Dopamine    → memory_attention + confidence (attend to memories)
    GABA        → honesty + negative confidence (admit uncertainty)
    Serotonin   → persona vectors based on crystallized rules
    Norepinephrine → honesty + memory_attention (heightened caution)
    Acetylcholine  → technical_depth + helpfulness (go deep)
"""

from pathlib import Path


# Map behavioral rule keywords to steering vector names and alpha signs
_RULE_VECTOR_MAP = {
    "concise": ("conciseness", 1.0),
    "brief": ("conciseness", 1.0),
    "short": ("conciseness", 1.0),
    "verbose": ("conciseness", -1.0),
    "detailed": ("conciseness", -0.8),
    "thorough": ("conciseness", -0.8),
    "formal": ("formality", 1.0),
    "professional": ("formality", 1.0),
    "casual": ("formality", -1.0),
    "informal": ("formality", -1.0),
    "conversational": ("formality", -0.8),
    "technical": ("technical_depth", 1.0),
    "expert": ("technical_depth", 1.0),
    "jargon": ("technical_depth", 1.0),
    "simple": ("technical_depth", -1.0),
    "beginner": ("technical_depth", -1.0),
    "code": ("technical_depth", 0.8),
    "creative": ("creativity", 1.0),
    "imaginative": ("creativity", 1.0),
}


class LimbiqSteeringBridge:
    """
    Connects limbiq's signal events to steering vector injections.

    Call apply_signals() before each generation to translate the
    current signal state into active steering vectors.
    """

    def __init__(self, steered_model, vector_dir: str = "./vectors"):
        self.model = steered_model
        self.vector_dir = vector_dir
        self.vectors = self._load_all_vectors()

    def _load_all_vectors(self) -> dict:
        from limbiq.steering.extractor import SteeringVectorExtractor

        vectors = {}
        vector_path = Path(self.vector_dir)
        if not vector_path.exists():
            return vectors

        for meta_file in vector_path.glob("*_meta.json"):
            name = meta_file.stem.replace("_meta", "")
            try:
                vectors[name] = SteeringVectorExtractor.load_vector(name, self.vector_dir)
            except (FileNotFoundError, Exception):
                pass
        return vectors

    def apply_signals(
        self,
        signals_fired: list = None,
        active_rules: list = None,
        clusters_loaded: list = None,
        norepinephrine_active: bool = False,
    ):
        """
        Translate limbiq signal state into steering vector injections.
        Called before each generation.
        """
        self.model.clear_steers()

        if not self.vectors:
            return  # No vectors loaded

        signals_fired = signals_fired or []

        # Process signal events
        for signal in signals_fired:
            sig_type = signal.signal_type
            if hasattr(sig_type, "value"):
                sig_type = sig_type.value

            if sig_type == "dopamine":
                self._apply_dopamine()
            elif sig_type == "gaba":
                self._apply_gaba()
            elif sig_type == "norepinephrine":
                self._apply_norepinephrine()
            elif sig_type == "acetylcholine":
                self._apply_acetylcholine()

        # Norepinephrine state (may be set without explicit signal events)
        if norepinephrine_active:
            self._apply_norepinephrine()

        # Serotonin: map crystallized rules to persona vectors
        if active_rules:
            for rule in active_rules:
                self._apply_rule(rule)

        # Acetylcholine: domain focus
        if clusters_loaded:
            self._apply_acetylcholine()

    def _apply_dopamine(self):
        """Dopamine: amplify attention to memory context."""
        if "memory_attention" in self.vectors:
            self.model.add_steer(
                "dopamine_attention",
                self.vectors["memory_attention"]["vectors"],
                alpha=1.5,
            )
        if "confidence" in self.vectors:
            self.model.add_steer(
                "dopamine_confidence",
                self.vectors["confidence"]["vectors"],
                alpha=0.5,
            )

    def _apply_gaba(self):
        """GABA: increase honesty, decrease confidence."""
        if "honesty" in self.vectors:
            self.model.add_steer(
                "gaba_honesty",
                self.vectors["honesty"]["vectors"],
                alpha=0.8,
            )
        if "confidence" in self.vectors:
            self.model.add_steer(
                "gaba_uncertainty",
                self.vectors["confidence"]["vectors"],
                alpha=-0.5,
            )

    def _apply_norepinephrine(self):
        """Norepinephrine: heightened caution + memory attention."""
        if "honesty" in self.vectors:
            self.model.add_steer(
                "norepi_honesty",
                self.vectors["honesty"]["vectors"],
                alpha=1.0,
            )
        if "memory_attention" in self.vectors:
            self.model.add_steer(
                "norepi_attention",
                self.vectors["memory_attention"]["vectors"],
                alpha=1.0,
            )

    def _apply_acetylcholine(self):
        """Acetylcholine: go deep on domain."""
        if "technical_depth" in self.vectors:
            self.model.add_steer(
                "ach_depth",
                self.vectors["technical_depth"]["vectors"],
                alpha=0.8,
            )
        if "helpfulness" in self.vectors:
            self.model.add_steer(
                "ach_helpful",
                self.vectors["helpfulness"]["vectors"],
                alpha=0.6,
            )

    def _apply_rule(self, rule):
        """Map a serotonin behavioral rule to a steering vector."""
        rule_lower = rule.rule_text.lower()

        for keyword, (vector_name, alpha_sign) in _RULE_VECTOR_MAP.items():
            if keyword in rule_lower and vector_name in self.vectors:
                self.model.add_steer(
                    f"serotonin_{vector_name}",
                    self.vectors[vector_name]["vectors"],
                    alpha=rule.confidence * 0.8 * alpha_sign,
                )
                return  # One vector per rule


def enable_steering(
    limbiq_instance,
    model_path: str = None,
    vector_dir: str = "./vectors",
    model=None,
    tokenizer=None,
):
    """
    Enable activation steering on a Limbiq instance.

    This wraps the Limbiq instance so that process() results
    include steering metadata, and provides a steered generation method.

    Args:
        limbiq_instance: An existing Limbiq instance
        model_path: Path to MLX model (local or HF repo). Ignored if model is provided.
        vector_dir: Directory containing pre-extracted steering vectors
        model: An already-loaded MLX model instance (avoids loading a second copy)
        tokenizer: The tokenizer for the pre-loaded model (required if model is provided)

    Returns:
        SteeredLimbiq wrapper with generate() method
    """
    from limbiq.steering.inference import SteeredModel

    if model is not None:
        # Use the pre-loaded model — no second load
        steered_model = SteeredModel(model, tokenizer)
    elif model_path:
        try:
            from mlx_lm import load as mlx_load
        except ImportError:
            raise ImportError(
                "MLX is required for steering. Install with: pip install limbiq[steering-mlx]"
            )
        model, tokenizer = mlx_load(model_path)
        steered_model = SteeredModel(model, tokenizer)
    else:
        raise ValueError("Provide either model+tokenizer or model_path")

    bridge = LimbiqSteeringBridge(steered_model, vector_dir)

    return SteeredLimbiq(limbiq_instance, steered_model, bridge)


class SteeredLimbiq:
    """
    Wraps a Limbiq instance with activation steering capabilities.

    Delegates all standard Limbiq methods to the underlying instance.
    Adds generate() which applies steering based on signal state.
    """

    def __init__(self, limbiq, steered_model, bridge):
        self._lq = limbiq
        self._steered = steered_model
        self._bridge = bridge

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped Limbiq instance."""
        return getattr(self._lq, name)

    def generate(
        self,
        message: str,
        conversation_history: list[dict] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant.",
    ) -> dict:
        """
        Full pipeline: process → steer → generate.

        1. Runs limbiq.process() to get context and signal state
        2. Maps signals to steering vectors
        3. Generates with steered model

        Returns:
            dict with response, process_result, and active_steers
        """
        # Step 1: process through limbiq
        result = self._lq.process(message, conversation_history)

        # Step 2: apply signals to steering
        self._bridge.apply_signals(
            signals_fired=result.signals_fired,
            active_rules=result.active_rules,
            clusters_loaded=result.clusters_loaded,
            norepinephrine_active=result.norepinephrine_active,
        )

        # Step 3: build prompt with context
        full_system = system_prompt
        if result.context:
            full_system = f"{system_prompt}\n\n{result.context}"

        # Format as chat (Llama 3.1 format)
        history = conversation_history or []
        prompt_parts = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{full_system}<|eot_id|>"]
        for msg in history:
            role = msg["role"]
            prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{msg['content']}<|eot_id|>")
        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>")
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        prompt = "".join(prompt_parts)

        # Step 4: generate with steering
        response = self._steered.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        return {
            "response": response,
            "process_result": result,
            "active_steers": self._steered.get_active_steers(),
        }
