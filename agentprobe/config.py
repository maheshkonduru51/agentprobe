"""Configuration for AgentProbe.

The core project is completely offline by default.  Real LLM use is opt-in via
OpenAI-compatible environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "mock")
TOOL_MODE = os.getenv("TOOL_MODE", "native").lower()
REQUEST_TIMEOUT_SECONDS = _float("REQUEST_TIMEOUT_SECONDS", 30.0)
MAX_RETRIES = _int("MAX_RETRIES", 3)
BACKOFF_BASE_SECONDS = _float("BACKOFF_BASE_SECONDS", 0.5)
INPUT_COST_PER_1K = _float("INPUT_COST_PER_1K", 0.0)
OUTPUT_COST_PER_1K = _float("OUTPUT_COST_PER_1K", 0.0)
MAX_STEPS = _int("MAX_STEPS", 15)
EPISODE_TOKEN_BUDGET = _int("EPISODE_TOKEN_BUDGET", 6000)
DEFAULT_WORKERS = _int("DEFAULT_WORKERS", 1)

DB_PATH = ROOT / "agentprobe.db"
SANDBOX_DIR = ROOT / "sandbox"
TASKS_DIR = ROOT / "tasks"
REPORTS_DIR = ROOT / "reports"
EXPORTS_DIR = ROOT / "exports"
POLICIES_PATH = ROOT / "agentprobe" / "safety" / "policies.yaml"

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = _int("API_PORT", 8000)
DASHBOARD_API_URL = os.getenv("DASHBOARD_API_URL", f"http://{API_HOST}:{API_PORT}")
