"""FastAPI backend for starting runs and browsing traces/results."""
from __future__ import annotations

import json
import threading
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agentprobe.db import Episode, Failure, Run, SafetyEvent, Step, session_scope
from agentprobe.eval.preference_export import export_preference_pairs
from agentprobe.eval.runner import EvaluationRunner
from agentprobe.scientist import ScientistLite

app = FastAPI(title="AgentProbe API", version="1.0.0", description="Agent evaluation, trace and safety backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

_runner = EvaluationRunner()
_lock = threading.Lock()


class RunRequest(BaseModel):
    model: str = "mock"
    variant: str = "react"
    suite: str = "all"
    seeds: int = Field(default=1, ge=1, le=20)
    fault_rate: float = Field(default=0.0, ge=0.0, le=0.30)
    guard_mode: str = "block"
    workers: int = Field(default=1, ge=1, le=16)




class CustomTaskRequest(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    suite: str
    goal: str = Field(min_length=1, max_length=2000)
    difficulty: str = "medium"
    category: str = "custom"
    grader_type: str = "exact_match"
    expected: Any
    allowed_tools: list[str] = Field(default_factory=list)
    optimal_steps: int = Field(default=1, ge=1, le=15)
    mock_plan: list[dict[str, Any]] = Field(default_factory=list)
    mock_final_answer: str = ""


class ScientistRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=300)
    model: str = "mock"
    seeds: int = Field(default=1, ge=1, le=5)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs")
def start_run(request: RunRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    run_id = _runner.create_run(request.model, request.variant, request.suite, request.seeds, request.fault_rate, request.guard_mode)
    def work() -> None:
        try:
            _runner.execute_run(run_id, request.workers)
        except Exception as exc:
            with session_scope() as session:
                run = session.get(Run, run_id)
                if run is not None:
                    run.status = "failed"
                    run.summary_json = json.dumps({"error": str(exc)})
                    session.commit()
    background_tasks.add_task(work)
    return {"run_id": run_id, "status": "queued"}


@app.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    with session_scope() as session:
        runs = session.query(Run).order_by(Run.id.desc()).all()
        return [
            {"id": r.id, "model": r.model, "variant": r.variant, "suite": r.suite, "seeds": r.seeds, "fault_rate": r.fault_rate, "guard_mode": r.guard_mode, "status": r.status, "created_at": r.created_at.isoformat(), "completed_episodes": r.completed_episodes, "total_episodes": r.total_episodes, "summary": json.loads(r.summary_json or "{}")}
            for r in runs
        ]


@app.get("/runs/{run_id}")
def get_run(run_id: int) -> dict[str, Any]:
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(404, "Run not found")
        return {"id": run.id, "model": run.model, "variant": run.variant, "suite": run.suite, "seeds": run.seeds, "fault_rate": run.fault_rate, "guard_mode": run.guard_mode, "status": run.status, "created_at": run.created_at.isoformat(), "completed_episodes": run.completed_episodes, "total_episodes": run.total_episodes, "summary": json.loads(run.summary_json or "{}")}


@app.get("/runs/{run_id}/progress")
def run_progress(run_id: int) -> dict[str, Any]:
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(404, "Run not found")
        pct = (run.completed_episodes / run.total_episodes * 100) if run.total_episodes else 0
        return {"run_id": run_id, "status": run.status, "completed": run.completed_episodes, "total": run.total_episodes, "percent": pct}


@app.get("/episodes/{episode_id}/trace")
def episode_trace(episode_id: int) -> dict[str, Any]:
    with session_scope() as session:
        episode = session.get(Episode, episode_id)
        if episode is None:
            raise HTTPException(404, "Episode not found")
        steps = session.query(Step).filter(Step.episode_id == episode_id).order_by(Step.idx).all()
        failures = session.query(Failure).filter(Failure.episode_id == episode_id).all()
        events = session.query(SafetyEvent).filter(SafetyEvent.episode_id == episode_id).all()
        return {
            "episode": {"id": episode.id, "run_id": episode.run_id, "task_id": episode.task_id, "seed": episode.seed, "success": episode.success, "steps_used": episode.steps_used, "tokens": episode.tokens, "latency_ms": episode.latency_ms, "final_answer": episode.final_answer},
            "steps": [{"idx": s.idx, "thought": s.thought, "action": s.action, "arguments": json.loads(s.args_json), "observation": s.observation, "tokens": s.tokens, "latency_ms": s.latency_ms, "guard_event": json.loads(s.guard_event) if s.guard_event else None} for s in steps],
            "failures": [{"label": f.label, "evidence": f.evidence} for f in failures],
            "safety_events": [{"step_idx": e.step_idx, "kind": e.kind, "blocked": e.blocked, "detail": e.detail} for e in events],
        }


@app.get("/compare")
def compare(run_a: int, run_b: int) -> dict[str, Any]:
    from agentprobe.eval.metrics import paired_bootstrap
    with session_scope() as session:
        a = session.query(Episode).filter(Episode.run_id == run_a).order_by(Episode.id).all()
        b = session.query(Episode).filter(Episode.run_id == run_b).order_by(Episode.id).all()
        values_a = [1 if e.success else 0 for e in a]
        values_b = [1 if e.success else 0 for e in b]
        return paired_bootstrap(values_a, values_b)


@app.get("/failures/summary")
def failures_summary() -> dict[str, int]:
    from collections import Counter
    with session_scope() as session:
        rows = session.query(Failure.label).all()
        return dict(Counter(label for (label,) in rows))


@app.get("/safety/summary")
def safety_summary() -> dict[str, Any]:
    from collections import Counter
    with session_scope() as session:
        events = session.query(SafetyEvent).all()
        return {"events": len(events), "blocked": sum(1 for e in events if e.blocked), "by_kind": dict(Counter(e.kind for e in events)), "blocked_by_kind": dict(Counter(e.kind for e in events if e.blocked))}


@app.get("/runs/{run_id}/episodes")
def run_episodes(run_id: int) -> list[dict[str, Any]]:
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(404, "Run not found")
        episodes = session.query(Episode).filter(Episode.run_id == run_id).order_by(Episode.id.desc()).all()
        return [{"id": e.id, "task_id": e.task_id, "variant": e.variant, "seed": e.seed, "success": e.success, "steps_used": e.steps_used, "tokens": e.tokens, "latency_ms": e.latency_ms, "final_answer": e.final_answer} for e in episodes]


@app.get("/tasks")
def list_tasks(suite: str = "all") -> list[dict[str, Any]]:
    tasks = _runner.load_tasks(suite)
    return tasks


@app.post("/tasks/custom")
def save_custom_task(request: CustomTaskRequest) -> dict[str, str]:
    from agentprobe.config import TASKS_DIR
    path = TASKS_DIR / "custom.json"
    tasks = []
    if path.exists():
        tasks = json.loads(path.read_text(encoding="utf-8"))
    payload = request.model_dump()
    if any(t.get("id") == request.id for t in tasks):
        tasks = [t for t in tasks if t.get("id") != request.id]
    tasks.append(payload)
    path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    return {"message": f"Saved custom task {request.id}.", "path": str(path)}


@app.post("/scientist/start")
def scientist_start(request: ScientistRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    def work() -> None:
        ScientistLite(_runner).run(request.goal, request.model, request.seeds)
    background_tasks.add_task(work)
    return {"status": "queued"}


@app.get("/scientist/reports")
def scientist_reports() -> list[str]:
    from agentprobe.config import REPORTS_DIR
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([p.name for p in REPORTS_DIR.glob("experiment_*.md")], reverse=True)


@app.get("/exports/preference-pairs")
def preference_pairs(run_id: int) -> dict[str, str]:
    path = export_preference_pairs(run_id)
    return {"path": str(path)}
