"""Agent variant metadata and policy."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentVariant:
    name: str
    description: str
    max_replans: int = 0
    reflects: bool = False


VARIANTS = {
    "react": AgentVariant("react", "Thought/Action/Observation loop with native tool calling."),
    "plan_execute": AgentVariant("plan_execute", "Planner produces a numbered plan, executor performs each step, replanning up to 2 times on failure.", max_replans=2),
    "react_reflect": AgentVariant("react_reflect", "ReAct loop with reflection after tool errors and every 5 steps.", reflects=True),
}


def get_variant(name: str) -> AgentVariant:
    if name == "all":
        raise ValueError("Use one concrete variant when running an episode")
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown variant: {name}") from exc
