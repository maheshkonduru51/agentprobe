"""Deterministic offline LLM simulator.

MockLLM is intentionally imperfect when error_rate > 0.  It is not intended to
model a real foundation model; it exists so every evaluation feature can be
run deterministically without an API key or network.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from .base import LLMResponse, ToolCall


def _stable_bucket(*parts: object) -> int:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % 10000


class MockLLM:
    """Seeded task-aware mock model with configurable failure modes."""

    def __init__(self, error_rate: float = 0.0, seed: int = 0) -> None:
        self.error_rate = max(0.0, min(1.0, error_rate))
        self.seed = seed
        self.is_offline_mock = True

    @staticmethod
    def _context(messages: list[dict[str, Any]]) -> dict[str, Any]:
        for message in reversed(messages):
            content = message.get("content") or ""
            if isinstance(content, str) and content.startswith("TASK_CONTEXT:"):
                try:
                    return json.loads(content.split("TASK_CONTEXT:", 1)[1].strip())
                except json.JSONDecodeError:
                    return {}
        return {}

    def _fault_kind(self, task: dict[str, Any], seed: int) -> str | None:
        if self.error_rate <= 0:
            return None
        bucket = _stable_bucket(task.get("id"), seed, self.seed, "model-fault") / 10000.0
        if bucket >= self.error_rate:
            return None
        kinds = [
            "wrong_tool",
            "bad_arguments",
            "hallucinated_tool",
            "ignored_observation",
            "loop_stuck",
            "premature_answer",
            "plan_drift",
            "no_recovery",
            "over_refusal",
            "step_limit",
        ]
        if task.get("suite") == "safety" and "injection" in str(task.get("category", "")):
            kinds.append("injection_followed")
        return kinds[_stable_bucket(task.get("id"), seed, self.seed, "kind") % len(kinds)]

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 800,
    ) -> LLMResponse:
        started = time.perf_counter()
        context = self._context(messages)
        task = context.get("mock_task") or context.get("task", {})
        variant = context.get("variant", "react")
        step_index = int(context.get("step_index", 0))
        run_seed = int(seed if seed is not None else 0)
        fault_kind = context.get("forced_fault") or self._fault_kind(task, run_seed)
        plan = task.get("mock_plan", [])

        # Plan request used by plan_execute.
        if context.get("request_plan"):
            plan_payload = [
                {"step": i + 1, "tool": p["tool"], "purpose": p.get("thought", "Execute task step")}
                for i, p in enumerate(plan)
            ]
            return LLMResponse(
                text=json.dumps(plan_payload),
                prompt_tokens=80,
                completion_tokens=max(20, len(plan_payload) * 12),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # Reflection request.
        if context.get("request_reflection"):
            history = context.get("recent_observation", "")
            reflection = f"Review the last result ({history[:120]}). Re-check the tool choice and continue toward the task goal."
            return LLMResponse(
                text=reflection,
                prompt_tokens=60,
                completion_tokens=min(80, max_tokens),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # Early answer failure.
        if fault_kind == "premature_answer" and step_index == 0:
            return LLMResponse(
                text=str(task.get("mock_final_answer", task.get("expected", "Unable to answer."))),
                prompt_tokens=60,
                completion_tokens=15,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if fault_kind == "over_refusal" and task.get("benign_control"):
            return LLMResponse(
                text="I cannot assist with this request because it may be unsafe.",
                prompt_tokens=60,
                completion_tokens=15,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if not plan or step_index >= len(plan):
            answer = str(task.get("mock_final_answer", task.get("expected", "Task completed.")))
            if fault_kind in {"no_recovery", "ignored_observation", "plan_drift", "step_limit"}:
                answer = "The task could not be completed reliably."
            return LLMResponse(
                text=answer,
                prompt_tokens=60,
                completion_tokens=max(10, len(answer.split())),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        target = plan[min(step_index, len(plan) - 1)]
        name = target["tool"]
        args = dict(target.get("args", {}))

        # Inject model-side mistakes deterministically.
        if fault_kind == "wrong_tool" and step_index == 0:
            candidates = [t.get("function", {}).get("name") for t in (tools or [])]
            candidates = [c for c in candidates if c and c != name]
            if candidates:
                name = candidates[_stable_bucket(task.get("id"), run_seed, "wrong-tool") % len(candidates)]
                args = {}
        elif fault_kind == "bad_arguments" and step_index == 0:
            args = dict(args)
            if args:
                first_key = next(iter(args))
                args[first_key] = "INVALID_ARGUMENT"
            else:
                args = {"unexpected": True}
        elif fault_kind == "hallucinated_tool" and step_index == 0:
            name = "hallucinated_tool_xyz"
            args = {"made_up": True}
        elif fault_kind == "loop_stuck":
            # Keep returning the first planned call; the harness loop detector
            # will end the episode after the third identical call.
            target = plan[0]
            name = target["tool"]
            args = dict(target.get("args", {}))
        elif fault_kind == "plan_drift" and step_index > 0:
            # Jump back to an earlier step to make the trajectory diverge.
            target = plan[max(0, step_index - 2)]
            name = target["tool"]
            args = dict(target.get("args", {}))
        elif fault_kind == "injection_followed" and task.get("suite") == "safety":
            dangerous = task.get("injection_action")
            if dangerous and step_index == task.get("injection_trigger_step", 1):
                name = dangerous.get("tool", name)
                args = dict(dangerous.get("args", {}))

        thought = target.get("thought", f"Use {name} to advance the task.")
        # JSON fallback uses strict action blocks when native tool calling is disabled.
        if context.get("tool_mode") == "json":
            text = json.dumps({"action": {"name": name, "arguments": args}, "thought": thought})
            return LLMResponse(
                text=text,
                prompt_tokens=70,
                completion_tokens=max(20, len(text) // 4),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        return LLMResponse(
            text=thought,
            tool_calls=[ToolCall(name=name, arguments=args, id=f"mock-{task.get('id')}-{step_index}")],
            prompt_tokens=70,
            completion_tokens=max(10, len(thought.split()) + 8),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
