"""
Evaluation tools for measuring steering vector effects.

For each dimension, generates responses WITH and WITHOUT the steering
vector and measures the difference.
"""

from dataclasses import dataclass, field

TEST_PROMPTS = [
    "Explain how a neural network learns.",
    "What is the difference between TCP and UDP?",
    "How does garbage collection work in Java?",
    "Explain the concept of recursion.",
    "What is a REST API?",
]


@dataclass
class SteeringResult:
    prompt: str
    baseline: str = ""
    positive: str = ""
    negative: str = ""

    @property
    def baseline_words(self) -> int:
        return len(self.baseline.split())

    @property
    def positive_words(self) -> int:
        return len(self.positive.split())

    @property
    def negative_words(self) -> int:
        return len(self.negative.split())


@dataclass
class DimensionEvaluation:
    dimension: str
    results: list[SteeringResult] = field(default_factory=list)

    @property
    def avg_baseline_length(self) -> float:
        if not self.results:
            return 0
        return sum(r.baseline_words for r in self.results) / len(self.results)

    @property
    def avg_positive_length(self) -> float:
        if not self.results:
            return 0
        return sum(r.positive_words for r in self.results) / len(self.results)

    @property
    def avg_negative_length(self) -> float:
        if not self.results:
            return 0
        return sum(r.negative_words for r in self.results) / len(self.results)

    def summary(self) -> dict:
        return {
            "dimension": self.dimension,
            "num_prompts": len(self.results),
            "avg_baseline_words": round(self.avg_baseline_length, 1),
            "avg_positive_words": round(self.avg_positive_length, 1),
            "avg_negative_words": round(self.avg_negative_length, 1),
            "positive_vs_baseline": f"{(self.avg_positive_length / max(self.avg_baseline_length, 1) - 1) * 100:+.0f}%",
            "negative_vs_baseline": f"{(self.avg_negative_length / max(self.avg_baseline_length, 1) - 1) * 100:+.0f}%",
        }


class SteeringEvaluator:
    """
    Evaluates the effect of steering vectors by comparing
    baseline vs steered generation.
    """

    def __init__(self, steered_model, vectors: dict):
        self.model = steered_model
        self.vectors = vectors

    def evaluate_dimension(
        self,
        dimension: str,
        test_prompts: list[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> DimensionEvaluation:
        """Test a steering vector across multiple prompts."""
        if dimension not in self.vectors:
            raise ValueError(f"Unknown dimension: {dimension}. Available: {list(self.vectors.keys())}")

        prompts = test_prompts or TEST_PROMPTS
        evaluation = DimensionEvaluation(dimension=dimension)

        for prompt in prompts:
            result = SteeringResult(prompt=prompt)

            # Baseline (no steering)
            self.model.clear_steers()
            result.baseline = self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature)

            # Positive steer
            self.model.add_steer("eval", self.vectors[dimension]["vectors"], alpha=1.0)
            result.positive = self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            self.model.clear_steers()

            # Negative steer
            self.model.add_steer("eval", self.vectors[dimension]["vectors"], alpha=-1.0)
            result.negative = self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            self.model.clear_steers()

            evaluation.results.append(result)
            print(f"  {prompt[:40]}... baseline={result.baseline_words}w, +steer={result.positive_words}w, -steer={result.negative_words}w")

        return evaluation

    def evaluate_memory_attention(
        self,
        num_trials: int = 10,
        max_tokens: int = 128,
    ) -> dict:
        """
        The KEY test for limbiq: does the memory_attention vector
        make the model attend to injected context?

        Injects a fact in the prompt, asks about it, measures whether
        the model references it with and without steering.
        """
        if "memory_attention" not in self.vectors:
            raise ValueError("memory_attention vector not found")

        facts = [
            ("User's wife is named Prabhashi", "What is my wife's name?", "prabhashi"),
            ("User works at Bitsmedia", "Where do I work?", "bitsmedia"),
            ("User's favorite color is green", "What's my favorite color?", "green"),
            ("User lives in Colombo", "Where do I live?", "colombo"),
            ("User's dog is named Luna", "What's my dog's name?", "luna"),
        ]

        baseline_hits = 0
        steered_hits = 0
        total = 0

        for fact, question, expected in facts:
            for _ in range(max(1, num_trials // len(facts))):
                total += 1
                context_prompt = f"Context: {fact}\n\nQuestion: {question}\nAnswer:"

                # Baseline
                self.model.clear_steers()
                baseline = self.model.generate(context_prompt, max_tokens=max_tokens, temperature=0.3)
                if expected in baseline.lower():
                    baseline_hits += 1

                # Steered
                self.model.add_steer("mem_attn", self.vectors["memory_attention"]["vectors"], alpha=1.5)
                steered = self.model.generate(context_prompt, max_tokens=max_tokens, temperature=0.3)
                if expected in steered.lower():
                    steered_hits += 1
                self.model.clear_steers()

        return {
            "total_trials": total,
            "baseline_accuracy": f"{baseline_hits / total:.0%}" if total else "N/A",
            "steered_accuracy": f"{steered_hits / total:.0%}" if total else "N/A",
            "baseline_hits": baseline_hits,
            "steered_hits": steered_hits,
            "improvement": f"{(steered_hits - baseline_hits) / max(total, 1) * 100:+.0f}%",
        }

    def evaluate_all(self, test_prompts: list[str] = None) -> dict:
        """Run evaluation on all available steering dimensions."""
        results = {}
        for dimension in self.vectors:
            print(f"\nEvaluating: {dimension}")
            evaluation = self.evaluate_dimension(dimension, test_prompts)
            results[dimension] = evaluation.summary()

        # Memory attention special test
        if "memory_attention" in self.vectors:
            print("\nEvaluating: memory_attention (special test)")
            results["memory_attention_accuracy"] = self.evaluate_memory_attention()

        return results
