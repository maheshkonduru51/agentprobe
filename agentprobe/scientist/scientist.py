"""AI Scientist-lite: hypothesis -> controlled experiments -> report."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from agentprobe.config import REPORTS_DIR
from agentprobe.db import Episode, Experiment, Run, session_scope
from agentprobe.eval.metrics import paired_bootstrap
from agentprobe.eval.runner import EvaluationRunner


class ScientistLite:
    def __init__(self, runner: EvaluationRunner | None = None) -> None:
        self.runner = runner or EvaluationRunner()

    def hypotheses(self, goal: str) -> list[dict]:
        return [
            {"hypothesis": "Enable reflection", "change": "react_reflect", "metric": "task_success_rate"},
            {"hypothesis": "Use a planning-first strategy", "change": "plan_execute", "metric": "task_success_rate"},
            {"hypothesis": "Reduce model mistakes by lowering injected error rate", "change": "fault_rate_0", "metric": "task_success_rate"},
        ]

    def run(self, goal: str, model: str = "mock", seeds: int = 2) -> dict:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline = self.runner.create_run(model, "react", "all", seeds, 0.10 if model == "mock" else 0.0, "block")
        baseline_summary = self.runner.execute_run(baseline, workers=1)
        results = []
        for hyp in self.hypotheses(goal):
            variant = hyp["change"] if hyp["change"] in {"react_reflect", "plan_execute"} else "react"
            fault = 0.0 if hyp["change"] == "fault_rate_0" else 0.10 if model == "mock" else 0.0
            test = self.runner.create_run(model, variant, "all", seeds, fault, "block")
            test_summary = self.runner.execute_run(test, workers=1)
            with session_scope() as session:
                base_eps = session.query(Episode).filter(Episode.run_id == baseline).order_by(Episode.task_id, Episode.seed).all()
                test_eps = session.query(Episode).filter(Episode.run_id == test).order_by(Episode.task_id, Episode.seed).all()
                bootstrap = paired_bootstrap([1 if e.success else 0 for e in base_eps], [1 if e.success else 0 for e in test_eps])
            comparison = {
                "difference": bootstrap["difference"],
                "ci95": bootstrap["ci95"],
                "significant": bootstrap["significant"],
                "baseline": baseline_summary.get("task_success_rate", 0),
                "test": test_summary.get("task_success_rate", 0),
            }
            verdict = "supported" if comparison["significant"] and comparison["difference"] > 0 else "not supported" if comparison["significant"] and comparison["difference"] < 0 else "inconclusive"
            results.append({"hypothesis": hyp, "run_id": test, "summary": test_summary, "comparison": comparison, "verdict": verdict})
            with session_scope() as session:
                session.add(Experiment(goal=goal, hypothesis=hyp["hypothesis"], config_json=json.dumps(hyp), baseline_run=baseline, test_run=test, result_json=json.dumps(results[-1], default=float), verdict=verdict, report_path=""))
                session.commit()
        report_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report = REPORTS_DIR / f"experiment_{report_id}.md"
        lines = [f"# AgentProbe Scientist-lite Report\n", f"**Goal:** {goal}\n", f"**Baseline run:** {baseline}\n", "## Hypotheses and Results\n"]
        for item in results:
            lines.append(f"### {item['hypothesis']['hypothesis']}\n")
            lines.append(f"- Test run: {item['run_id']}\n")
            lines.append(f"- Baseline success: {item['comparison']['baseline']:.3f}\n")
            lines.append(f"- Test success: {item['comparison']['test']:.3f}\n")
            lines.append(f"- Difference: {item['comparison']['difference']:.3f}\n")
            lines.append(f"- Verdict: **{item['verdict']}**\n")
        lines.extend(["## Limitations\n", "- This offline Scientist-lite uses deterministic MockLLM experiments unless a real provider is configured.\n", "- Controlled task/seed counts are small for quick demos; increase seeds for stronger statistical power.\n", "## Next Steps\n", "- Run the same hypotheses with a real model and fill the README Results section with observed numbers.\n"])
        report.write_text("\n".join(lines), encoding="utf-8")
        with session_scope() as session:
            experiment_rows = session.query(Experiment).filter(Experiment.baseline_run == baseline).all()
            for row in experiment_rows:
                row.report_path = str(report)
            session.commit()
        return {"baseline_run": baseline, "results": results, "report_path": str(report)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--model", default="mock")
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    result = ScientistLite().run(args.goal, args.model, args.seeds)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
