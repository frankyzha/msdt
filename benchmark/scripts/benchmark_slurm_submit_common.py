#!/usr/bin/env python3
"""Shared Slurm submission helpers for cached benchmark runners."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.benchmark_paths import BENCHMARK_ARTIFACTS_ROOT, SCRIPT_ROOT
from benchmark.scripts.experiment_utils import canonical_dataset_list


def _resolve_python(python_arg: Path | None) -> Path:
    if python_arg is not None:
        return python_arg.expanduser().resolve()
    analysis_python = REPO_ROOT / ".analysis-venv" / "bin" / "python"
    if analysis_python.exists():
        return analysis_python
    return Path(sys.executable).resolve()


def _normalize_benchmark_args(raw_args: list[str]) -> list[str]:
    if raw_args and raw_args[0] == "--":
        return raw_args[1:]
    return list(raw_args)


def _contains_option(args: list[str], option: str) -> bool:
    for token in args:
        if token == option or token.startswith(f"{option}="):
            return True
    return False


def _build_parser(*, description: str, default_log_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Exactly one dataset slug to benchmark.")
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Python interpreter to run remotely. Defaults to .analysis-venv/bin/python when present.",
    )
    parser.add_argument("--partition", default="compsci-gpu")
    parser.add_argument("--nodelist", default="compsci-cluster-fitz-[45-49]")
    parser.add_argument("--time-limit", default="24:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="100G")
    parser.add_argument(
        "--mail-user",
        default="frank.zhang@duke.edu",
        help="Email address for Slurm job notifications.",
    )
    parser.add_argument(
        "--mail-type",
        default="BEGIN,END,FAIL",
        help="Comma-separated Slurm mail events.",
    )
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wait for the submitted Slurm job to finish before returning.",
    )
    parser.add_argument(
        "--exclusive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request an exclusive node allocation for cleaner timing.",
    )
    parser.add_argument(
        "--timing-mode",
        choices=("fair", "fast"),
        default="fair",
        help="Timing policy forwarded to the benchmark runner.",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Optional Slurm job name. Defaults to a dataset-specific name.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=default_log_dir,
        help="Directory for generated batch scripts and Slurm stdout/stderr logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated batch script and sbatch command without submitting.",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the benchmark script. Prefix with -- to separate them.",
    )
    return parser


def _render_batch_script(
    *,
    args: argparse.Namespace,
    run_name: str,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> str:
    slurm_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.job_name or run_name}",
        f"#SBATCH --output={stdout_path}",
        f"#SBATCH --error={stderr_path}",
        f"#SBATCH -p {args.partition}",
        f"#SBATCH --nodelist={args.nodelist}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={int(args.cpus_per_task)}",
        f"#SBATCH --time={args.time_limit}",
    ]
    if args.mem:
        slurm_lines.append(f"#SBATCH --mem={args.mem}")
    if args.mail_user:
        slurm_lines.append(f"#SBATCH --mail-user={args.mail_user}")
    if args.mail_type:
        slurm_lines.append(f"#SBATCH --mail-type={args.mail_type}")
    if bool(args.exclusive):
        slurm_lines.append("#SBATCH --exclusive")

    body_lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        "export PYTHONUNBUFFERED=1",
        'echo "Job started on $(hostname) at $(date)"',
        f"echo {shlex.quote('Command: ' + shlex.join(command))}",
        shlex.join(command),
        'echo "Job finished on $(hostname) at $(date)"',
    ]
    return "\n".join([*slurm_lines, "", *body_lines, ""])


def _submit_batch(batch_script_path: Path, *, wait: bool) -> str:
    submit_cmd = ["sbatch", "--parsable"]
    if bool(wait):
        submit_cmd.append("--wait")
    submit_cmd.append(str(batch_script_path))
    result = subprocess.run(submit_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Slurm submission failed.\n"
            f"command: {shlex.join(submit_cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def submit_single_dataset_benchmark(
    *,
    script_name: str,
    run_prefix: str,
    description: str,
    default_log_subdir: str,
) -> int:
    default_log_dir = BENCHMARK_ARTIFACTS_ROOT / "slurm_jobs" / default_log_subdir
    parser = _build_parser(description=description, default_log_dir=default_log_dir)
    args = parser.parse_args()
    if args.cpus_per_task < 1:
        raise ValueError("--cpus-per-task must be at least 1")
    if str(args.timing_mode) == "fair" and not bool(args.exclusive):
        raise ValueError(
            "--timing-mode fair requires --exclusive. "
            "Use --timing-mode fast for throughput-oriented shared-node runs."
        )

    dataset = canonical_dataset_list([args.dataset])[0]
    script_path = (SCRIPT_ROOT / script_name).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Benchmark script not found: {script_path}")
    python_path = _resolve_python(args.python)
    if not python_path.exists():
        raise FileNotFoundError(f"Python interpreter not found: {python_path}")

    extra_args = _normalize_benchmark_args(args.benchmark_args)
    forbidden_options = {"--dataset", "--datasets", "--timing-mode"}
    for option in forbidden_options:
        if _contains_option(extra_args, option):
            raise ValueError(
                f"{option} should not be passed through benchmark_args; use the launcher flag instead."
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset}_{run_prefix}_{timestamp}"
    if not _contains_option(extra_args, "--run-name"):
        extra_args.extend(["--run-name", run_name])
    command = [
        str(python_path),
        str(script_path),
        "--datasets",
        dataset,
        "--timing-mode",
        str(args.timing_mode),
        *extra_args,
    ]

    log_dir = args.log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{run_name}_%j.out"
    stderr_path = log_dir / f"{run_name}_%j.err"
    batch_script_text = _render_batch_script(
        args=args,
        run_name=run_name,
        command=command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"{run_name}_",
        suffix=".sbatch",
        dir=log_dir,
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(batch_script_text)
        batch_script_path = Path(handle.name)

    if args.dry_run:
        print(f"batch_script: {batch_script_path}")
        print(batch_script_text)
        dry_run_cmd = ["sbatch", "--parsable"]
        if bool(args.wait):
            dry_run_cmd.append("--wait")
        dry_run_cmd.append(str(batch_script_path))
        print(f"sbatch_command: {shlex.join(dry_run_cmd)}")
        return 0

    job_id = _submit_batch(batch_script_path, wait=bool(args.wait))
    if bool(args.wait):
        print(f"Submitted and waited for Slurm job {job_id}")
    else:
        print(f"Submitted Slurm job {job_id}")
    print(f"batch_script: {batch_script_path}")
    print(f"stdout_log: {stdout_path}")
    print(f"stderr_log: {stderr_path}")
    return 0
