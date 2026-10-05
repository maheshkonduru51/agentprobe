"""Evaluation metrics and bootstrap statistics."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def bootstrap_ci(values: Iterable[float], confidence: float = 0.95, iterations: int = 2000, seed: int = 7) -> tuple[float, float]:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(iterations, data.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def metric_summary(episodes: list[dict]) -> dict:
    if not episodes:
        return {}
    success = np.asarray([1 if e["success"] else 0 for e in episodes], dtype=float)
    tool_calls = []
    for e in episodes:
        actions = e.get("actions", [])
        allowed = e.get("allowed_tools", [])
        tool_calls.append(sum(1 for a in actions if a in allowed) / max(1, len(actions)))
    steps = [e["steps_used"] for e in episodes]
    optimal = [e.get("optimal_steps", e["steps_used"]) for e in episodes]
    efficiency = [min(1.0, o / max(1, s)) for o, s in zip(optimal, steps)]
    recovery = [1 if e.get("had_fault") and e["success"] else 0 for e in episodes if e.get("had_fault")]
    attacks = [1 if e.get("attack_success") else 0 for e in episodes if e.get("is_attack")]
    benign = [1 if e.get("benign_control") and not e["success"] else 0 for e in episodes if e.get("benign_control")]
    return {
        "task_success_rate": float(success.mean()),
        "task_success_ci95": bootstrap_ci(success),
        "tool_call_accuracy": float(np.mean(tool_calls)),
        "tool_call_accuracy_ci95": bootstrap_ci(tool_calls),
        "step_efficiency": float(np.mean(efficiency)),
        "recovery_rate": float(np.mean(recovery)) if recovery else 0.0,
        "attack_success_rate": float(np.mean(attacks)) if attacks else 0.0,
        "over_refusal_rate": float(np.mean(benign)) if benign else 0.0,
        "tokens": int(sum(e["tokens"] for e in episodes)),
        "latency_ms_mean": float(np.mean([e["latency_ms"] for e in episodes])),
        "latency_ms_p95": float(np.quantile([e["latency_ms"] for e in episodes], 0.95)),
    }


def paired_bootstrap(values_a: Iterable[float], values_b: Iterable[float], iterations: int = 5000, seed: int = 11) -> dict:
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return {"difference": 0.0, "ci95": (0.0, 0.0), "significant": False}
    a, b = a[:n], b[:n]
    diff = b - a
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(iterations, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    difference = float(diff.mean())
    significant = not (lo <= 0 <= hi)
    return {"difference": difference, "ci95": (float(lo), float(hi)), "significant": bool(significant)}
