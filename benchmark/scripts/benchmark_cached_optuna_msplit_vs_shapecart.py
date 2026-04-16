#!/usr/bin/env python3
"""Tune MSPLIT and ShapeCART with nested cross-validation and Optuna."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.scripts.benchmark_cached_common import (
    aggregate_results,
    best_depth_table,
    configure_timing_mode,
    resolve_search_jobs,
    write_csv_tables,
)
from benchmark.scripts.benchmark_cached_optuna_support import (
    append_progress_log,
    build_run_config,
    checkpoint_benchmark_state,
    iter_fold_runs,
    parse_args,
    resolve_requested_run,
    study_storage_uri,
    write_result_tables,
)
from benchmark.scripts.benchmark_cached_optuna_tuning import (
    flush_artifact_jobs,
    require_optuna,
    tune_msplit,
    tune_shapecart,
)


def main() -> int:
    args = parse_args()
    optuna = require_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.environ["MSPLIT_BUILD_DIR"] = str(args.msplit_build_dir)
    configure_timing_mode(args.timing_mode)

    datasets, depths, fold_seed, deprecated_seed_args = resolve_requested_run(args)
    if deprecated_seed_args:
        print(
            "[note] --seeds is deprecated for this benchmark; "
            f"using outer-fold CV with --fold-seed={int(fold_seed)}.",
            flush=True,
        )
    search_jobs = resolve_search_jobs(args.search_jobs, args.timing_mode)
    final_fit_guard_enabled = bool(args.timing_mode == "fair")

    out_dir = Path(args.results_root) / (
        args.run_name or datetime.now().strftime("cached_optuna_msplit_vs_shapecart_%Y%m%d_%H%M%S")
    )
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            build_run_config(
                args=args,
                datasets=datasets,
                depths=depths,
                fold_seed=fold_seed,
                search_jobs=search_jobs,
                final_fit_guard_enabled=final_fit_guard_enabled,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    selected_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = [] if args.save_trials else []
    cache_manifest_rows: list[dict[str, object]] = [] if args.save_cache_manifest else []
    artifact_jobs: list[dict[str, object]] = []
    storage_uri = study_storage_uri(
        out_dir=out_dir,
        search_jobs=int(search_jobs),
        persist_study=bool(args.persist_optuna_study),
    )
    append_progress_log(out_dir, f"[start] datasets={datasets} depths={depths} outer_folds={int(args.outer_folds)} seed={int(fold_seed)}")

    for fold_run in iter_fold_runs(
        args=args,
        datasets=datasets,
        fold_seed=fold_seed,
        out_dir=out_dir,
        collect_manifest=bool(args.save_cache_manifest),
    ):
        print(
            f"[fold] dataset={fold_run.dataset} outer_fold={fold_run.fold_index}/{fold_run.outer_folds} "
            f"seed={int(fold_seed)}",
            flush=True,
        )
        if args.save_cache_manifest:
            cache_manifest_rows.extend(fold_run.manifest_rows)
        for depth in depths:
            append_progress_log(
                out_dir,
                f"[run] dataset={fold_run.dataset} outer_fold={fold_run.fold_index}/{fold_run.outer_folds} depth={depth} algorithm=msplit",
            )
            msplit_row, msplit_trials, msplit_artifact_job = tune_msplit(
                tune_protocols=fold_run.tune_protocols,
                final_protocol=fold_run.final_protocol,
                fold_index=fold_run.fold_index,
                outer_folds=fold_run.outer_folds,
                depth=int(depth),
                cv_folds=int(args.cv_folds),
                n_trials=int(args.n_trials),
                search_jobs=int(search_jobs),
                final_fit_guard_enabled=final_fit_guard_enabled,
                sampler_seed=int(100000 + 1000 * int(fold_seed) + 100 * int(fold_run.fold_index) + 10 * int(depth) + 1),
                study_storage_uri=storage_uri,
                study_name=f"msplit::{fold_run.dataset}::fold{fold_run.fold_index}::depth{int(depth)}",
                shared_min_leaf_values=[int(v) for v in args.shared_min_leaf_values],
                shared_split_multipliers=[int(v) for v in args.shared_split_multipliers],
                msplit_lookahead_depth_values=[int(v) for v in args.msplit_lookahead_depth_values],
                msplit_exactify_top_k_values=[int(v) for v in args.msplit_exactify_top_k_values],
                msplit_max_branching_values=[int(v) for v in args.msplit_max_branching_values],
                msplit_reg_values=[float(v) for v in args.msplit_reg_values],
                msplit_worker_limit=int(args.msplit_worker_limit),
                out_dir=out_dir,
                save_model_artifacts=bool(args.save_model_artifacts),
                collect_trial_rows=bool(args.save_trials),
            )
            selected_rows.append(msplit_row)
            if args.save_trials:
                trial_rows.extend(msplit_trials)
            if msplit_artifact_job is not None:
                artifact_jobs.append(msplit_artifact_job)
            checkpoint_benchmark_state(
                out_dir=out_dir,
                selected_rows=selected_rows,
                trial_rows=trial_rows if args.save_trials else None,
                cache_manifest_rows=cache_manifest_rows if args.save_cache_manifest else None,
                aggregate_results=aggregate_results,
                best_depth_table=best_depth_table,
                write_csv_tables=write_csv_tables,
            )
            append_progress_log(
                out_dir,
                f"[done] dataset={fold_run.dataset} outer_fold={fold_run.fold_index}/{fold_run.outer_folds} depth={depth} algorithm=msplit test_accuracy={float(msplit_row['test_accuracy']):.6f}",
            )

            append_progress_log(
                out_dir,
                f"[run] dataset={fold_run.dataset} outer_fold={fold_run.fold_index}/{fold_run.outer_folds} depth={depth} algorithm=shapecart",
            )
            shapecart_row, shapecart_trials, shapecart_artifact_job = tune_shapecart(
                tune_protocols=fold_run.tune_protocols,
                final_protocol=fold_run.final_protocol,
                fold_index=fold_run.fold_index,
                outer_folds=fold_run.outer_folds,
                depth=int(depth),
                cv_folds=int(args.cv_folds),
                n_trials=int(args.n_trials),
                search_jobs=int(search_jobs),
                final_fit_guard_enabled=final_fit_guard_enabled,
                sampler_seed=int(200000 + 1000 * int(fold_seed) + 100 * int(fold_run.fold_index) + 10 * int(depth) + 2),
                study_storage_uri=storage_uri,
                study_name=f"shapecart::{fold_run.dataset}::fold{fold_run.fold_index}::depth{int(depth)}",
                shared_min_leaf_values=[int(v) for v in args.shared_min_leaf_values],
                shared_split_multipliers=[int(v) for v in args.shared_split_multipliers],
                shape_criterion_values=list(args.shape_criterion_values),
                shape_inner_max_depth=int(args.shape_inner_max_depth),
                shape_inner_max_leaf_values=[int(v) for v in args.shape_inner_max_leaf_values],
                shape_inner_min_leaf_values=list(args.shape_inner_min_leaf_values),
                shape_k=int(args.shape_k),
                shape_max_iter=int(args.shape_max_iter),
                shape_pairwise_candidates=float(args.shape_pairwise_candidates),
                shape_smart_init=bool(args.shape_smart_init),
                shape_random_pairs=bool(args.shape_random_pairs),
                shape_use_dpdt=bool(args.shape_use_dpdt),
                shape_use_tao=bool(args.shape_use_tao),
                shape_branching_penalty_values=[float(v) for v in args.shape_branching_penalty_values],
                shape_tao_reg_values=[float(v) for v in args.shape_tao_reg_values],
                shape_tao_n_runs=int(args.shape_tao_n_runs),
                shape_tao_pair_scale=float(args.shape_tao_pair_scale),
                out_dir=out_dir,
                save_model_artifacts=bool(args.save_model_artifacts),
                collect_trial_rows=bool(args.save_trials),
            )
            selected_rows.append(shapecart_row)
            if args.save_trials:
                trial_rows.extend(shapecart_trials)
            if shapecart_artifact_job is not None:
                artifact_jobs.append(shapecart_artifact_job)
            checkpoint_benchmark_state(
                out_dir=out_dir,
                selected_rows=selected_rows,
                trial_rows=trial_rows if args.save_trials else None,
                cache_manifest_rows=cache_manifest_rows if args.save_cache_manifest else None,
                aggregate_results=aggregate_results,
                best_depth_table=best_depth_table,
                write_csv_tables=write_csv_tables,
            )
            append_progress_log(
                out_dir,
                f"[done] dataset={fold_run.dataset} outer_fold={fold_run.fold_index}/{fold_run.outer_folds} depth={depth} algorithm=shapecart test_accuracy={float(shapecart_row['test_accuracy']):.6f}",
            )

    if artifact_jobs:
        append_progress_log(out_dir, f"[artifacts] writing {len(artifact_jobs)} selected model artifacts")
        flush_artifact_jobs(artifact_jobs)
    checkpoint_benchmark_state(
        out_dir=out_dir,
        selected_rows=selected_rows,
        trial_rows=trial_rows if args.save_trials else None,
        cache_manifest_rows=cache_manifest_rows if args.save_cache_manifest else None,
        aggregate_results=aggregate_results,
        best_depth_table=best_depth_table,
        write_csv_tables=write_csv_tables,
    )
    append_progress_log(out_dir, f"[done] wrote results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
