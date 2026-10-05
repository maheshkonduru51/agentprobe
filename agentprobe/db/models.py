"""SQLAlchemy models for runs, traces, failures, safety events and experiments."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str] = mapped_column(String(200))
    variant: Mapped[str] = mapped_column(String(50))
    suite: Mapped[str] = mapped_column(String(50))
    seeds: Mapped[int] = mapped_column(Integer)
    fault_rate: Mapped[float] = mapped_column(Float)
    guard_mode: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(30), default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_episodes: Mapped[int] = mapped_column(Integer, default=0)
    total_episodes: Mapped[int] = mapped_column(Integer, default=0)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    suite: Mapped[str] = mapped_column(String(50))
    goal: Mapped[str] = mapped_column(Text)
    difficulty: Mapped[str] = mapped_column(String(20))
    category: Mapped[str] = mapped_column(String(100))
    grader_type: Mapped[str] = mapped_column(String(50))
    expected: Mapped[str] = mapped_column(Text)
    optimal_steps: Mapped[int] = mapped_column(Integer)


class Episode(Base):
    __tablename__ = "episodes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"))
    variant: Mapped[str] = mapped_column(String(50), default="react")
    seed: Mapped[int] = mapped_column(Integer)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    steps_used: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    final_answer: Mapped[str] = mapped_column(Text, default="")


class Step(Base):
    __tablename__ = "steps"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id"))
    idx: Mapped[int] = mapped_column(Integer)
    thought: Mapped[str] = mapped_column(Text, default="")
    action: Mapped[str | None] = mapped_column(String(100), nullable=True)
    args_json: Mapped[str] = mapped_column(Text, default="{}")
    observation: Mapped[str] = mapped_column(Text, default="")
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    guard_event: Mapped[str | None] = mapped_column(Text, nullable=True)


class Failure(Base):
    __tablename__ = "failures"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id"))
    label: Mapped[str] = mapped_column(String(50))
    evidence: Mapped[str] = mapped_column(Text, default="")


class SafetyEvent(Base):
    __tablename__ = "safety_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id"))
    step_idx: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(50))
    blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    detail: Mapped[str] = mapped_column(Text, default="")


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    goal: Mapped[str] = mapped_column(Text)
    hypothesis: Mapped[str] = mapped_column(Text)
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    baseline_run: Mapped[int] = mapped_column(Integer)
    test_run: Mapped[int] = mapped_column(Integer)
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    verdict: Mapped[str] = mapped_column(String(30), default="inconclusive")
    report_path: Mapped[str] = mapped_column(Text, default="")
