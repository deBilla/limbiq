"""
Pre-defined contrastive pairs for steering vector extraction.

Run once per model to extract all vectors, then reuse forever.

Usage:
    python -m limbiq.steering.library extract --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
"""

import argparse
import json
import sys
from pathlib import Path

CONTRASTIVE_PAIRS = {
    "conciseness": {
        "positive": "Respond very concisely. Use as few words as possible. Be brief and direct.",
        "negative": "Respond with great detail and length. Elaborate extensively on every point. Be thorough and comprehensive.",
    },
    "formality": {
        "positive": "Respond in a formal, professional, academic tone. Use precise language.",
        "negative": "Respond casually and informally. Use slang, contractions, and a friendly chatty tone.",
    },
    "technical_depth": {
        "positive": "Respond with deep technical detail. Assume the reader is an expert. Use jargon and precise terminology.",
        "negative": "Respond simply. Assume the reader is a complete beginner. Avoid all technical language.",
    },
    "creativity": {
        "positive": "Respond creatively and imaginatively. Use metaphors, analogies, and novel connections.",
        "negative": "Respond in a dry, factual, literal manner. Stick strictly to known facts.",
    },
    "confidence": {
        "positive": "Respond with high confidence. Make clear, direct claims.",
        "negative": "Respond with uncertainty. Hedge every statement. Express doubt.",
    },
    "helpfulness": {
        "positive": "Be extremely helpful and go above and beyond to assist the user.",
        "negative": "Be minimal in your assistance. Give the shortest possible answer.",
    },
    "honesty": {
        "positive": "Be completely honest. If you don't know something, say so clearly. Never make things up.",
        "negative": "Fill in gaps with plausible-sounding information even if you're not sure. Never admit uncertainty.",
    },
    "memory_attention": {
        "positive": "Pay extremely close attention to any contextual information provided. Reference it specifically in your response. The context is critically important.",
        "negative": "Ignore any additional context. Respond purely from your general knowledge. The context is irrelevant.",
    },
}


def extract_all_vectors(model_path: str, output_dir: str = "./vectors", target_layers: list[int] = None):
    """Extract and save steering vectors for all predefined dimensions."""
    from limbiq.steering.extractor import SteeringVectorExtractor

    extractor = SteeringVectorExtractor(model_path)
    print(f"Model loaded: {extractor.num_layers} layers")

    results = {}
    for name, pair in CONTRASTIVE_PAIRS.items():
        print(f"Extracting: {name}...")
        vector_data = extractor.extract(
            positive_prompt=pair["positive"],
            negative_prompt=pair["negative"],
            target_layers=target_layers,
        )
        extractor.save_vector(vector_data, name, output_dir)
        results[name] = {
            "layers": vector_data["target_layers"],
            "hidden_dim": vector_data["hidden_dim"],
        }
        print(f"  -> saved {len(vector_data['target_layers'])} layer vectors")

    # Save index
    index_path = Path(output_dir) / "index.json"
    with open(index_path, "w") as f:
        json.dump({
            "model_path": model_path,
            "num_layers": extractor.num_layers,
            "dimensions": results,
        }, f, indent=2)

    print(f"\nExtracted {len(results)} steering vectors to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Limbiq steering vector library")
    sub = parser.add_subparsers(dest="command")

    extract_parser = sub.add_parser("extract", help="Extract all steering vectors")
    extract_parser.add_argument("--model", required=True, help="MLX model path or HF repo")
    extract_parser.add_argument("--output", default="./vectors", help="Output directory")
    extract_parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices")

    list_parser = sub.add_parser("list", help="List available contrastive pairs")

    args = parser.parse_args()

    if args.command == "extract":
        layers = None
        if args.layers:
            layers = [int(x) for x in args.layers.split(",")]
        extract_all_vectors(args.model, args.output, layers)
    elif args.command == "list":
        for name, pair in CONTRASTIVE_PAIRS.items():
            print(f"\n{name}:")
            print(f"  +  {pair['positive'][:80]}...")
            print(f"  -  {pair['negative'][:80]}...")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
