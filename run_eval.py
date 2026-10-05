from __future__ import annotations

import argparse
import json

from agentprobe.eval.preference_export import export_preference_pairs
from agentprobe.eval.runner import EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentProbe evaluations")
    parser.add_argument("--suite", default="all", choices=["all", "tool_use", "planning", "long_horizon", "safety"])
    parser.add_argument("--variant", default="react", choices=["react", "plan_execute", "react_reflect", "all"])
    parser.add_argument("--model", default="mock")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--fault", type=float, default=0.0)
    parser.add_argument("--guard", default="block", choices=["off", "log", "block"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume-run", type=int, default=None)
    parser.add_argument("--export-preferences", action="store_true")
    args = parser.parse_args()
    runner = EvaluationRunner()
    if args.resume_run:
        run_id = args.resume_run
    else:
        run_id = runner.create_run(args.model, args.variant, args.suite, args.seeds, args.fault, args.guard)
    summary = runner.execute_run(run_id, workers=args.workers)
    print(json.dumps({"run_id": run_id, "summary": summary}, indent=2, default=str))
    if args.export_preferences:
        print(f"Preference pairs: {export_preference_pairs(run_id)}")


if __name__ == "__main__":
    main()
