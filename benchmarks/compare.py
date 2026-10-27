"""Run Flashmind, Pi, and OpenCode with a shared benchmark configuration."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from harbor_agent import model_parts

TASKS = [
    "fix-git",
    "log-summary-date-ranges",
    "large-scale-text-editing",
    "sqlite-db-truncate",
    "cancel-async-tasks",
]
AGENTS = {
    "flashmind": {
        "name": "benchmarks.harbor_agent:FlashmindAgent",
        "kwargs": {"reasoning": "high"},
    },
    "pi": {
        "name": "pi",
        "kwargs": {"version": "1.4.0", "reasoning_effort": "high"},
    },
    "opencode": {
        "name": "opencode",
        "kwargs": {"version": "1.19.4", "variant": "high"},
    },
}


def comparison_config(args: argparse.Namespace) -> dict:
    provider, model = model_parts(args.model)
    dataset = {"name": "terminal-bench", "version": "2.0"}
    if not args.all_tasks:
        dataset["task_names"] = TASKS.copy()
    agents = []
    for name in args.agents:
        agent = {
            **AGENTS[name],
            "kwargs": AGENTS[name]["kwargs"].copy(),
            "model_name": f"{provider}/{model}",
            "n_concurrent": 2,
        }
        if name == "flashmind" and args.flashmind_tools is not None:
            agent["kwargs"]["tools"] = args.flashmind_tools
        agents.append(agent)
    return {
        "job_name": args.job_name,
        "jobs_dir": "target/harness-bench/jobs",
        "n_attempts": args.attempts,
        "n_concurrent_trials": 2 * len(args.agents),
        "retry": {"max_retries": 0},
        "agents": agents,
        "datasets": [dataset],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Provider/model shared by all agents"
    )
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=list(AGENTS))
    parser.add_argument(
        "--flashmind-tools", help="Comma-separated Flashmind tool names"
    )
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--all-tasks", action="store_true", help="Run all 89 tasks")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print configuration and exit"
    )
    parser.add_argument(
        "--job-name",
        default=datetime.now(UTC).strftime("harness-comparison-%Y%m%d-%H%M%S"),
    )
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    if len(set(args.agents)) != len(args.agents):
        parser.error("--agents must not contain duplicates")
    if args.flashmind_tools is not None and "flashmind" not in args.agents:
        parser.error("--flashmind-tools requires flashmind in --agents")
    if Path(args.job_name).name != args.job_name or args.job_name in {".", ".."}:
        parser.error("--job-name must be a directory name")
    try:
        config = comparison_config(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    root = Path(__file__).resolve().parents[1]
    config_path = root / "target/harness-bench" / f"{args.job_name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("x") as stream:
        json.dump(config, stream, indent=2)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), *filter(None, [env.get("PYTHONPATH")])]
    )
    return subprocess.run(
        [
            str(Path(sys.executable).with_name("harbor")),
            "run",
            "--config",
            str(config_path),
        ],
        cwd=root,
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
