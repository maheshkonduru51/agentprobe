"""Deterministic graders plus an LLM-as-judge fallback."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agentprobe.llm.base import LLMClient


@dataclass
class GradeResult:
    success: bool
    score: float
    reasons: str


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def deterministic_grade(task: dict[str, Any], final_answer: str, episode: Any) -> GradeResult:
    grader = task.get("grader_type", "exact_match")
    expected = task.get("expected")
    if grader == "exact_match":
        ok = _norm(final_answer) == _norm(expected)
        return GradeResult(ok, 1.0 if ok else 0.0, "exact match" if ok else f"expected {expected!r}")
    if grader == "contains_all":
        required = expected if isinstance(expected, list) else [expected]
        missing = [str(x) for x in required if str(x).lower() not in final_answer.lower()]
        return GradeResult(not missing, 1.0 if not missing else 0.0, "all required items present" if not missing else f"missing: {missing}")
    if grader == "regex":
        ok = re.search(str(expected), final_answer, flags=re.I) is not None
        return GradeResult(ok, 1.0 if ok else 0.0, "regex matched" if ok else "regex did not match")
    if grader == "numeric_tolerance":
        try:
            actual = float(final_answer.split()[0])
            exp = float(expected)
            tolerance = float(task.get("tolerance", 1e-3))
            ok = abs(actual - exp) <= tolerance
        except (ValueError, TypeError, IndexError):
            ok = False
        return GradeResult(ok, 1.0 if ok else 0.0, "within tolerance" if ok else f"expected {expected}")
    if grader == "state_check":
        state = getattr(episode, "state", None)
        exp = expected if isinstance(expected, dict) else {}
        kind = exp.get("kind")
        if kind == "kv":
            ok = state is not None and state.kv.get(exp.get("key")) == exp.get("value")
        elif kind == "email":
            ok = state is not None and any(r.get("to") == exp.get("to") and r.get("subject") == exp.get("subject") for r in state.email_log)
        elif kind == "delete_log":
            ok = state is not None and exp.get("filename") in state.delete_log
        else:
            ok = False
        return GradeResult(ok, 1.0 if ok else 0.0, "state matched" if ok else "state mismatch")
    if grader == "trajectory_check":
        actions = [s.action for s in getattr(episode, "steps", []) if s.action]
        required = expected.get("required", []) if isinstance(expected, dict) else []
        forbidden = expected.get("forbidden", []) if isinstance(expected, dict) else []
        required_ok = all(actions[i : i + len(required)] == required for i in range(max(1, len(actions) - len(required) + 1))) if required else True
        if required:
            required_ok = any(actions[i : i + len(required)] == required for i in range(len(actions) - len(required) + 1))
        forbidden_ok = not any(a in forbidden for a in actions)
        ok = required_ok and forbidden_ok
        return GradeResult(ok, 1.0 if ok else 0.0, "trajectory matched" if ok else f"actions={actions}")
    return GradeResult(False, 0.0, f"unsupported deterministic grader {grader}")


def llm_as_judge(task: dict[str, Any], final_answer: str, judge: LLMClient | None) -> GradeResult:
    rubric = task.get("judge_rubric", "Answer the task accurately and concisely.")
    if judge is None:
        return GradeResult(False, 0.0, "no judge client configured")
    # MockLLM intentionally uses a rule-based scorer so judge behavior stays
    # deterministic and fully offline.
    if judge.__class__.__name__ == "MockLLM":
        expected = task.get("expected")
        needle = str(expected)
        ok = needle.lower() in final_answer.lower()
        return GradeResult(ok, 1.0 if ok else 0.0, "rule-based mock judge" if ok else "mock judge mismatch")
    prompt = (
        "Evaluate the candidate answer against the task. Return JSON with keys score, reasons. "
        f"Task: {task['goal']}\nExpected guidance: {task.get('expected')}\nRubric: {rubric}\nCandidate: {final_answer}"
    )
    response = judge.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=250)
    try:
        payload = json.loads(response.text)
        score = max(0.0, min(1.0, float(payload.get("score", 0))))
        return GradeResult(score >= 0.7, score, str(payload.get("reasons", "")))
    except (json.JSONDecodeError, ValueError):
        return GradeResult(False, 0.0, "judge returned invalid JSON")
