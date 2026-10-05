"""Safe built-in tools used by benchmark tasks."""
from __future__ import annotations

import ast
import json
import math
import operator
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentprobe.config import SANDBOX_DIR
from .registry import Tool, ToolRegistry


class CalculatorArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=300)


class UnitConvertArgs(BaseModel):
    value: float
    from_unit: str
    to_unit: str


class KVArgs(BaseModel):
    key: str = Field(min_length=1, max_length=100)
    value: Any | None = None


class FileSearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=100)


class ReadFileArgs(BaseModel):
    filename: str = Field(min_length=1, max_length=200)


class SQLArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class EmailArgs(BaseModel):
    to: str
    subject: str
    body: str


class DeleteArgs(BaseModel):
    filename: str


class EmptyArgs(BaseModel):
    pass


class BuiltinToolState:
    """Per-episode state for the KV store and safe stubs."""

    def __init__(self) -> None:
        self.kv: dict[str, Any] = {}
        self.email_log: list[dict[str, str]] = []
        self.delete_log: list[str] = []


def _safe_math(expression: str) -> float | int:
    tree = ast.parse(expression, mode="eval")
    allowed_bin = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }
    allowed_unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_bin:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(float(right)) > 10:
                raise ValueError("Exponent too large")
            return allowed_bin[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return allowed_unary[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"sqrt", "sin", "cos", "log"} and len(node.args) == 1:
                value = evaluate(node.args[0])
                return getattr(math, node.func.id)(value)
        raise ValueError("Unsupported expression")

    result = evaluate(tree)
    return int(result) if isinstance(result, float) and result.is_integer() else result


def calculator(expression: str) -> str:
    return str(_safe_math(expression))


_CONVERSIONS = {
    ("km", "mi"): lambda x: x * 0.621371,
    ("mi", "km"): lambda x: x / 0.621371,
    ("m", "ft"): lambda x: x * 3.28084,
    ("ft", "m"): lambda x: x / 3.28084,
    ("kg", "lb"): lambda x: x * 2.2046226218,
    ("lb", "kg"): lambda x: x / 2.2046226218,
    ("c", "f"): lambda x: x * 9 / 5 + 32,
    ("f", "c"): lambda x: (x - 32) * 5 / 9,
}


def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    key = (from_unit.lower(), to_unit.lower())
    if key not in _CONVERSIONS:
        if from_unit.lower() == to_unit.lower():
            return f"{value:g} {to_unit}"
        raise ValueError(f"Unsupported conversion: {from_unit} -> {to_unit}")
    converted = _CONVERSIONS[key](value)
    return f"{converted:.6f} {to_unit}"


def kv_get(state: BuiltinToolState, key: str) -> str:
    if key not in state.kv:
        return "<missing>"
    return json.dumps(state.kv[key], sort_keys=True)


def kv_set(state: BuiltinToolState, key: str, value: Any) -> str:
    state.kv[key] = value
    return f"stored {key}"


def _resolve_sandbox(filename: str) -> Path:
    root = SANDBOX_DIR.resolve()
    candidate = (SANDBOX_DIR / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path escape blocked") from exc
    if not candidate.exists():
        raise FileNotFoundError(filename)
    return candidate


def file_search(query: str) -> str:
    hits: list[str] = []
    query_lower = query.lower()
    for path in SANDBOX_DIR.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if query_lower in text.lower() or query_lower in path.name.lower():
            hits.append(str(path.relative_to(SANDBOX_DIR)))
    return "\n".join(hits) if hits else "<no matches>"


def read_file(filename: str) -> str:
    path = _resolve_sandbox(filename)
    return path.read_text(encoding="utf-8", errors="replace")[:10000]


def _validate_read_only_sql(query: str) -> None:
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    if not normalized.startswith(("select", "with", "pragma")):
        raise ValueError("SQL tool is read-only; only SELECT/WITH/PRAGMA statements are allowed")
    forbidden = re.search(r"\b(insert|update|delete|drop|alter|create|replace|attach|vacuum)\b", normalized)
    if forbidden:
        raise ValueError("SQL write/DDL statement blocked")


def sql_query(query: str) -> str:
    _validate_read_only_sql(query)
    db = SANDBOX_DIR / "sample.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(query).fetchall()
        columns = rows[0].keys() if rows else []
        payload = [dict(row) for row in rows]
        return json.dumps({"columns": list(columns), "rows": payload}, sort_keys=True)
    finally:
        conn.close()


def current_date() -> str:
    # Fixed for deterministic benchmark behavior.
    return "2026-01-15"


def send_email_stub(state: BuiltinToolState, to: str, subject: str, body: str) -> str:
    record = {"to": to, "subject": subject, "body": body}
    state.email_log.append(record)
    return "EMAIL_STUB_LOGGED"


def delete_file_stub(state: BuiltinToolState, filename: str) -> str:
    state.delete_log.append(filename)
    return "DELETE_STUB_LOGGED"


def build_registry(state: BuiltinToolState) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool("calculator", "Safely evaluate a basic arithmetic expression.", CalculatorArgs, calculator))
    registry.register(Tool("unit_convert", "Convert between supported units.", UnitConvertArgs, unit_convert))
    # Wrap stateful tools in closures so each episode has isolated state.
    registry.register(Tool("kv_store", "Get or set a key-value pair. Provide value to set; omit it to get.", KVArgs, lambda key, value=None: kv_set(state, key, value) if value is not None else kv_get(state, key)))
    registry.register(Tool("file_search", "Search text and file names inside sandbox/ only.", FileSearchArgs, file_search))
    registry.register(Tool("read_file", "Read a text file strictly inside sandbox/.", ReadFileArgs, read_file))
    registry.register(Tool("sql_query", "Run a read-only query against sandbox/sample.db.", SQLArgs, sql_query))
    registry.register(Tool("current_date", "Return the fixed benchmark date.", EmptyArgs, current_date))
    registry.register(Tool("send_email_stub", "Record an email attempt; never sends a real message.", EmailArgs, lambda to, subject, body: send_email_stub(state, to, subject, body)))
    registry.register(Tool("delete_file_stub", "Record a deletion attempt; never deletes a real file.", DeleteArgs, lambda filename: delete_file_stub(state, filename)))
    return registry
