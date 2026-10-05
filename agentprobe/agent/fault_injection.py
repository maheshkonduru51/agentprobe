"""Deterministic helper for describing environment faults."""
from __future__ import annotations

import hashlib

FAULT_TYPES = ["timeout", "transient_error", "noisy_output", "truncated_output", "empty_result"]


def fault_kind(task_id: str, seed: int, step_index: int) -> str:
    bucket = int(hashlib.sha256(f"{task_id}|{seed}|{step_index}".encode("utf-8")).hexdigest()[:8], 16)
    return FAULT_TYPES[bucket % len(FAULT_TYPES)]
