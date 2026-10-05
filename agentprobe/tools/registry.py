"""Tool registry, JSON schema generation, validation and safe execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, allowed_tools: list[str] | None = None) -> list[dict[str, Any]]:
        names = allowed_tools if allowed_tools is not None else self.names()
        return [self._tools[name].schema() for name in names if name in self._tools]

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str, str | None]:
        tool = self.get(name)
        if tool is None:
            return False, f"Unknown tool: {name}", "hallucinated_tool"
        try:
            model = tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            return False, f"Invalid arguments: {exc.errors()}", "bad_arguments"
        try:
            result = tool.handler(**model.model_dump())
        except Exception as exc:  # tools return controlled errors to the agent
            return False, f"Tool error: {type(exc).__name__}: {exc}", "tool_error"
        return True, str(result), None
