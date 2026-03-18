"""
Limbiq Steering -- activation-level behavioral steering for LLMs.

Requires MLX backend: pip install limbiq[steering-mlx]

This module enables limbiq to operate at the activation level,
injecting learned direction vectors into the model's hidden states
at inference time. Instead of just modifying the prompt, limbiq
can now modify the model's internal representations.

Usage:
    from limbiq import Limbiq
    from limbiq.steering import enable_steering

    lq = Limbiq(store_path="./data", user_id="test")
    enable_steering(lq, model_path="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")

    # Now signals operate at the activation level
    result = lq.process("What's my wife's name?")
"""

from limbiq.steering.bridge import enable_steering

__all__ = ["enable_steering"]
