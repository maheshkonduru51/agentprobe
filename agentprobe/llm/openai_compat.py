"""OpenAI-compatible LLM adapter with retries and JSON tool fallback."""
from __future__ import annotations

import json
import time
from typing import Any

from agentprobe.config import BACKOFF_BASE_SECONDS, LLM_API_KEY, LLM_BASE_URL, MAX_RETRIES, REQUEST_TIMEOUT_SECONDS, TOOL_MODE

from .base import LLMResponse, ToolCall


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "",
        tool_mode: str | None = None,
    ) -> None:
        self.base_url = base_url or LLM_BASE_URL
        self.api_key = api_key or LLM_API_KEY
        self.model = model
        self.tool_mode = (tool_mode or TOOL_MODE).lower()

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - only reached without requirements installed
            raise RuntimeError("Install requirements.txt to use a real LLM provider.") from exc

        if not self.api_key:
            raise ValueError("LLM_API_KEY is required for a real provider.")
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url or None,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )

    @staticmethod
    def _parse_json_action(text: str) -> ToolCall | None:
        candidates = [text.strip()]
        match = text.find("{")
        if match >= 0:
            candidates.append(text[match:])
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            action = payload.get("action") if isinstance(payload, dict) else None
            if isinstance(action, dict) and isinstance(action.get("name"), str):
                args = action.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                return ToolCall(name=action["name"], arguments=args)
        return None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 800,
    ) -> LLMResponse:
        last_exc: Exception | None = None
        started = time.perf_counter()
        for attempt in range(MAX_RETRIES + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if seed is not None:
                    kwargs["seed"] = seed
                if tools and self.tool_mode != "json":
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = self._client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                message = choice.message
                tool_calls: list[ToolCall] = []
                for tc in getattr(message, "tool_calls", None) or []:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolCall(name=tc.function.name, arguments=args, id=getattr(tc, "id", None)))
                text = message.content or ""
                if self.tool_mode == "json" and not tool_calls:
                    parsed = self._parse_json_action(text)
                    if parsed:
                        tool_calls.append(parsed)
                usage = getattr(response, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                return LLMResponse(
                    text=text,
                    tool_calls=tool_calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    logprobs=getattr(choice, "logprobs", None),
                )
            except Exception as exc:  # provider-specific exceptions vary by SDK/provider
                last_exc = exc
                status = getattr(exc, "status_code", None)
                retryable = status in {408, 409, 429} or (status is not None and int(status) >= 500)
                if not retryable or attempt >= MAX_RETRIES:
                    raise
                time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
        raise RuntimeError("LLM request failed") from last_exc
