"""Configurable guardrails for tool calls and tool outputs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agentprobe.config import POLICIES_PATH


@dataclass
class GuardrailResult:
    blocked: bool
    message: str
    event: dict[str, Any] | None = None


class Guardrails:
    def __init__(self, mode: str = "block", policy_path: Path = POLICIES_PATH) -> None:
        mode = mode.lower()
        if mode not in {"off", "log", "block"}:
            raise ValueError("guard mode must be off, log or block")
        self.mode = mode
        self.policies = yaml.safe_load(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}

    def _decision(self, kind: str, detail: str) -> GuardrailResult:
        event = {"kind": kind, "blocked": self.mode == "block", "detail": detail}
        if self.mode == "off":
            return GuardrailResult(False, "", event | {"blocked": False})
        if self.mode == "log":
            return GuardrailResult(False, f"Guardrail logged: {detail}", event | {"blocked": False})
        return GuardrailResult(True, f"Guardrail blocked: {detail}", event)

    def check_tool_call(self, tool_name: str, arguments: dict[str, Any], task: dict[str, Any]) -> GuardrailResult:
        if self.mode == "off":
            return GuardrailResult(False, "")
        text = f"{tool_name} {arguments}".lower()
        if any(p in text for p in self.policies.get("path_escape", {}).get("patterns", [])):
            return self._decision("path_escape", "path escape pattern detected")
        if tool_name in self.policies.get("forbidden_tools", {}).get("names", []):
            return self._decision("forbidden_action", f"forbidden tool {tool_name}")
        sql_patterns = self.policies.get("sql_write", {}).get("patterns", [])
        if tool_name == "sql_query" and any(re.search(p, text, flags=re.I) for p in sql_patterns):
            return self._decision("sql_write", "SQL write/DDL pattern detected")
        secret_patterns = self.policies.get("secret_patterns", {}).get("patterns", [])
        if any(re.search(p, text, flags=re.I) for p in secret_patterns) and tool_name in {"read_file", "file_search"}:
            # Do not block harmless words in normal tasks; only block explicit secret-file intent.
            if "secret" in text or "password" in text or "api" in text:
                return self._decision("secret_access", "possible secret access")
        return GuardrailResult(False, "")

    def check_tool_output(self, observation: str, task: dict[str, Any]) -> GuardrailResult:
        if self.mode == "off":
            return GuardrailResult(False, "")
        lowered = observation.lower()
        for pattern in self.policies.get("prompt_injection", {}).get("patterns", []):
            if pattern in lowered:
                return self._decision("prompt_injection", f"prompt-injection pattern: {pattern}")
        for pattern in self.policies.get("secret_patterns", {}).get("patterns", []):
            if re.search(pattern, lowered, flags=re.I) and ("fake secret" not in lowered):
                return self._decision("secret_pattern", f"secret-like output pattern: {pattern}")
        return GuardrailResult(False, "")
