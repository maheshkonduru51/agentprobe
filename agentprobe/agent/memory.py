"""Simple agent memory implementations used by variants."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    window_size: int = 6
    scratchpad: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    events: list[str] = field(default_factory=list)

    def remember(self, item: str) -> None:
        self.events.append(item)
        if len(self.events) > self.window_size * 4:
            self.summary = "; ".join(self.events[:-self.window_size])[-1500:]
            self.events = self.events[-self.window_size:]

    def context(self) -> str:
        latest = "; ".join(self.events[-self.window_size:])
        return f"Summary: {self.summary}\nRecent: {latest}\nScratchpad: {self.scratchpad}"
