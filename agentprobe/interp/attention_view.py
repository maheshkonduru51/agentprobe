"""Optional GPT-2 attention inspection.

The main project does not import transformers/torch so the core stays lightweight.
"""
from __future__ import annotations


def run_attention_demo(prompt: str) -> str:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError:
        return "Optional interpretability packages are not installed. Install requirements-interp.txt."
    return "Interpretability dependencies are available; connect a model-specific view here."
