from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import os
from pathlib import Path
import time
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESS_WARMUP_DONE = False


@dataclass(frozen=True)
class TimingGuardConfig:
    enabled: bool
    lock_path: Path
    lock_timeout_sec: float
    quiet_timeout_sec: float
    poll_interval_sec: float
    sample_window_sec: float
    max_cpu_utilization: float
    max_runnable_tasks: int
    warmup_runs: int
    measure_runs: int
    max_relative_spread: float
    max_absolute_spread_sec: float
    small_fit_cutoff_sec: float

    @classmethod
    def from_env(cls) -> "TimingGuardConfig":
        return cls(
            enabled=_env_bool("MSDT_BENCHMARK_GUARD", True),
            lock_path=Path(os.environ.get("MSDT_BENCHMARK_LOCK_PATH", "/tmp/msdt_msplit_benchmark.lock")),
            lock_timeout_sec=_env_float("MSDT_BENCHMARK_LOCK_TIMEOUT_SEC", 900.0),
            quiet_timeout_sec=_env_float("MSDT_BENCHMARK_QUIET_TIMEOUT_SEC", 900.0),
            poll_interval_sec=_env_float("MSDT_BENCHMARK_POLL_INTERVAL_SEC", 5.0),
            sample_window_sec=_env_float("MSDT_BENCHMARK_SAMPLE_WINDOW_SEC", 0.25),
            max_cpu_utilization=_env_float("MSDT_BENCHMARK_MAX_CPU_UTIL", 0.25),
            max_runnable_tasks=_env_int("MSDT_BENCHMARK_MAX_RUNNABLE_TASKS", 1),
            warmup_runs=_env_int("MSDT_BENCHMARK_WARMUP_RUNS", 1),
            measure_runs=max(1, _env_int("MSDT_BENCHMARK_MEASURE_RUNS", 2)),
            max_relative_spread=_env_float("MSDT_BENCHMARK_MAX_REL_SPREAD", 0.12),
            max_absolute_spread_sec=_env_float("MSDT_BENCHMARK_MAX_ABS_SPREAD_SEC", 0.05),
            small_fit_cutoff_sec=_env_float("MSDT_BENCHMARK_SMALL_FIT_CUTOFF_SEC", 1.0),
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _read_proc_stat() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [int(field) for field in fields]
    idle = values[3] + values[4]
    total = sum(values)
    return idle, total


def _sample_cpu_utilization(window_sec: float) -> float | None:
    try:
        idle_before, total_before = _read_proc_stat()
        time.sleep(max(0.01, float(window_sec)))
        idle_after, total_after = _read_proc_stat()
    except Exception:
        return None
    delta_total = total_after - total_before
    delta_idle = idle_after - idle_before
    if delta_total <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - (float(delta_idle) / float(delta_total))))


def _current_runnable_tasks() -> int | None:
    try:
        parts = Path("/proc/loadavg").read_text(encoding="utf-8").split()
        runnable, _ = parts[3].split("/", maxsplit=1)
        return int(runnable)
    except Exception:
        return None


def _same_user_repo_python_pids(repo_root: Path) -> list[int]:
    repo_str = str(repo_root)
    uid = os.getuid()
    pids: list[int] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == os.getpid():
            continue
        try:
            status_text = (proc_dir / "status").read_text(encoding="utf-8")
            uid_line = next((line for line in status_text.splitlines() if line.startswith("Uid:")), None)
            if uid_line is None:
                continue
            proc_uid = int(uid_line.split()[1])
            if proc_uid != uid:
                continue
            cmd_raw = (proc_dir / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            continue
        if not cmd_raw:
            continue
        argv = [part.decode("utf-8", errors="replace") for part in cmd_raw.split(b"\0") if part]
        if not argv:
            continue
        exe_name = Path(argv[0]).name.lower()
        if "python" not in exe_name and not any("python" in token.lower() for token in argv[:2]):
            continue
        joined = " ".join(argv)
        if repo_str in joined:
            pids.append(pid)
    return sorted(pids)


def collect_host_snapshot(repo_root: Path | None = None, *, sample_window_sec: float = 0.0) -> dict[str, Any]:
    if repo_root is None:
        repo_root = REPO_ROOT
    cpu_count = os.cpu_count() or 1
    cpu_util = _sample_cpu_utilization(sample_window_sec) if sample_window_sec > 0.0 else None
    snapshot = {
        "timestamp_unix_sec": float(time.time()),
        "cpu_count": int(cpu_count),
        "cpu_utilization": None if cpu_util is None else float(cpu_util),
        "runnable_tasks": _current_runnable_tasks(),
        "repo_python_pids": _same_user_repo_python_pids(repo_root),
    }
    snapshot["repo_python_process_count"] = int(len(snapshot["repo_python_pids"]))
    return snapshot


def _quiet_host_reason(snapshot: dict[str, Any], cfg: TimingGuardConfig) -> str | None:
    repo_pids = [int(pid) for pid in snapshot.get("repo_python_pids", [])]
    if repo_pids:
        return f"other repo python processes are active: {repo_pids}"
    runnable_tasks = snapshot.get("runnable_tasks")
    if runnable_tasks is not None and int(runnable_tasks) > int(cfg.max_runnable_tasks):
        return (
            f"host has {int(runnable_tasks)} runnable tasks, "
            f"threshold is {int(cfg.max_runnable_tasks)}"
        )
    cpu_util = snapshot.get("cpu_utilization")
    if cpu_util is not None and float(cpu_util) > float(cfg.max_cpu_utilization):
        return (
            f"host CPU utilization {float(cpu_util):.3f} exceeds "
            f"threshold {float(cfg.max_cpu_utilization):.3f}"
        )
    return None


@contextmanager
def _acquire_lock(lock_path: Path, *, timeout_sec: float):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    started = time.perf_counter()
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle.seek(0)
                handle.truncate(0)
                handle.write(f"pid={os.getpid()}\n")
                handle.write(f"started_unix_sec={time.time():.6f}\n")
                handle.flush()
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                if time.perf_counter() - started >= float(timeout_sec):
                    raise TimeoutError(
                        f"Timed out waiting for benchmark lock {lock_path} after {timeout_sec:.1f}s"
                    ) from exc
                time.sleep(1.0)
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _wait_for_quiet_host(cfg: TimingGuardConfig, *, repo_root: Path) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    last_snapshot: dict[str, Any] | None = None
    last_reason = "unknown"
    while True:
        snapshot = collect_host_snapshot(repo_root, sample_window_sec=cfg.sample_window_sec)
        reason = _quiet_host_reason(snapshot, cfg)
        if reason is None:
            return float(time.perf_counter() - started), snapshot
        last_snapshot = snapshot
        last_reason = reason
        if time.perf_counter() - started >= float(cfg.quiet_timeout_sec):
            raise RuntimeError(
                "Benchmark timing guard could not obtain a quiet host before timeout. "
                f"Last observation: {last_reason}. Snapshot={last_snapshot}"
            )
        time.sleep(float(cfg.poll_interval_sec))


def guarded_fit(
    run_once: Callable[[], dict[str, Any]],
    *,
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    global _PROCESS_WARMUP_DONE

    if repo_root is None:
        repo_root = REPO_ROOT
    cfg = TimingGuardConfig.from_env()
    if not cfg.enabled:
        started = time.perf_counter()
        result = run_once()
        fit_seconds = float(time.perf_counter() - started)
        return result, fit_seconds, {
            "enabled": False,
            "fit_seconds_runs": [fit_seconds],
            "selected_fit_seconds": fit_seconds,
        }

    with _acquire_lock(cfg.lock_path, timeout_sec=cfg.lock_timeout_sec):
        quiet_wait_seconds, pre_snapshot = _wait_for_quiet_host(cfg, repo_root=repo_root)
        warmup_runs_completed = 0
        if not _PROCESS_WARMUP_DONE:
            for _ in range(max(0, int(cfg.warmup_runs))):
                run_once()
                warmup_runs_completed += 1
            _PROCESS_WARMUP_DONE = True

        fit_seconds_runs: list[float] = []
        measured_results: list[dict[str, Any]] = []
        for _ in range(int(cfg.measure_runs)):
            started = time.perf_counter()
            result = run_once()
            fit_seconds_runs.append(float(time.perf_counter() - started))
            measured_results.append(result)

        selected_idx = min(range(len(fit_seconds_runs)), key=fit_seconds_runs.__getitem__)
        selected_fit_seconds = float(fit_seconds_runs[selected_idx])
        min_fit = selected_fit_seconds
        max_fit = float(max(fit_seconds_runs))
        absolute_spread_sec = float(max_fit - min_fit)
        relative_spread = 0.0 if min_fit <= 0.0 else float((max_fit - min_fit) / min_fit)
        post_snapshot = collect_host_snapshot(repo_root, sample_window_sec=0.0)
        is_small_fit = bool(min_fit < float(cfg.small_fit_cutoff_sec))
        spread_too_large = (
            absolute_spread_sec > float(cfg.max_absolute_spread_sec)
            if is_small_fit
            else relative_spread > float(cfg.max_relative_spread)
        )
        if spread_too_large:
            raise RuntimeError(
                "Benchmark timing guard observed unstable fit timings even after waiting for a quiet host. "
                f"fit_seconds_runs={fit_seconds_runs}, "
                f"absolute_spread_sec={absolute_spread_sec:.4f}, "
                f"relative_spread={relative_spread:.4f}, "
                f"relative_threshold={cfg.max_relative_spread:.4f}, "
                f"absolute_threshold_sec={cfg.max_absolute_spread_sec:.4f}, "
                f"small_fit_cutoff_sec={cfg.small_fit_cutoff_sec:.4f}, "
                f"pre_snapshot={pre_snapshot}, post_snapshot={post_snapshot}"
            )

        report = {
            "enabled": True,
            "lock_path": str(cfg.lock_path),
            "lock_timeout_sec": float(cfg.lock_timeout_sec),
            "quiet_timeout_sec": float(cfg.quiet_timeout_sec),
            "quiet_wait_seconds": float(quiet_wait_seconds),
            "sample_window_sec": float(cfg.sample_window_sec),
            "max_cpu_utilization": float(cfg.max_cpu_utilization),
            "max_runnable_tasks": int(cfg.max_runnable_tasks),
            "warmup_runs_configured": int(cfg.warmup_runs),
            "warmup_runs_completed": int(warmup_runs_completed),
            "measure_runs": int(cfg.measure_runs),
            "fit_seconds_runs": fit_seconds_runs,
            "selected_fit_seconds": float(selected_fit_seconds),
            "selected_run_index": int(selected_idx),
            "absolute_spread_sec": float(absolute_spread_sec),
            "relative_spread": float(relative_spread),
            "max_absolute_spread_sec": float(cfg.max_absolute_spread_sec),
            "small_fit_cutoff_sec": float(cfg.small_fit_cutoff_sec),
            "pre_snapshot": pre_snapshot,
            "post_snapshot": post_snapshot,
        }
        return measured_results[selected_idx], selected_fit_seconds, report
