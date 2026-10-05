"""Export chosen/rejected trajectories for tasks with both outcomes."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from agentprobe.config import EXPORTS_DIR
from agentprobe.db import Episode, Step, session_scope


def export_preference_pairs(run_id: int) -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = EXPORTS_DIR / "preference_pairs.jsonl"
    with session_scope() as session, output.open("w", encoding="utf-8") as fh:
        episodes = session.query(Episode).filter(Episode.run_id == run_id).all()
        by_task = defaultdict(list)
        for ep in episodes:
            by_task[ep.task_id].append(ep)
        count = 0
        for task_id, eps in by_task.items():
            chosen = next((e for e in eps if e.success), None)
            rejected = next((e for e in eps if not e.success), None)
            if not chosen or not rejected:
                continue
            def trajectory(ep):
                return [
                    {"idx": s.idx, "action": s.action, "arguments": json.loads(s.args_json), "observation": s.observation}
                    for s in session.query(Step).filter(Step.episode_id == ep.id).order_by(Step.idx)
                ]
            fh.write(json.dumps({"task_id": task_id, "chosen": trajectory(chosen), "rejected": trajectory(rejected)}) + "\n")
            count += 1
    return output
