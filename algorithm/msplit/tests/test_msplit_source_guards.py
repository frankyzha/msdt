from __future__ import annotations

from pathlib import Path


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "libgosdt" / "src"


def _current_solver_source_text() -> dict[str, str]:
    root = _source_root()
    core_source = root / "msplit_core.cpp"
    selector_source = root / "msplit_nonlinear.cpp"
    support_source = root / "msplit_atomized_support.cpp"
    atomized_source = root / "msplit_atomized.cpp"
    heuristic_source = root / "msplit_atomized_heuristic.cpp"
    return {
        "core": core_source.read_text(encoding="utf-8"),
        "selector": selector_source.read_text(encoding="utf-8"),
        "support": support_source.read_text(encoding="utf-8"),
        "atomized": atomized_source.read_text(encoding="utf-8"),
        "heuristic": heuristic_source.read_text(encoding="utf-8"),
    }


def _dataset_source_text() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1] / "src" / "libgosdt"
    header = root / "include" / "dataset.hpp"
    source = root / "src" / "dataset.cpp"
    return {
        "header": header.read_text(encoding="utf-8"),
        "source": source.read_text(encoding="utf-8"),
    }


def test_current_sources_have_no_force_legacy_or_atom_descent_symbols():
    texts = _current_solver_source_text()
    text = "\n".join(texts.values())

    assert "MSPLIT_FORCE_RUSH_LEGACY" not in text
    assert "ensure_teacher_prior_hierarchy_for_prep_legacy" not in text
    assert "ensure_distilled_legacy_shape_for_prep" not in text
    assert "run_teacher_atom_descent(" not in text
    assert "run_teacher_atom_descent_family(" not in text

    py_source = Path(__file__).resolve().parents[1] / "src" / "split" / "MSPLIT.py"
    py_text = py_source.read_text(encoding="utf-8")
    assert "legacy_mean_logit" not in py_text


def test_current_sources_have_no_legacy_selector_or_path_lp_helpers():
    texts = _current_solver_source_text()
    text = "\n".join(texts.values())

    forbidden = [
        "path_bound_for_indices",
        "state_lower_bound_for_indices",
        "tighten_candidate_lower_bound_with_path",
        "initialize_candidate_lower_bounds",
        "candidate_leaf_completion_objective",
        "CanonicalSignatureState",
        "profiling_signature_bound_calls",
        "profiling_signature_bound_sec",
        "signature_state_cache_entries",
        "profiling_path_bound_calls",
        "profiling_path_bound_sec",
        "profiling_path_bound_skip_trivial",
        "profiling_path_bound_skip_disabled",
        "profiling_path_bound_skip_small_state",
        "profiling_path_bound_skip_too_many_blocks",
        "profiling_path_bound_skip_large_child",
        "profiling_path_bound_tighten_attempts",
        "profiling_path_bound_tighten_effective",
        "lp_skip_reason_not_promising",
        "lp_skip_reason_depth_gate",
        "lp_skip_reason_tighten_cap",
        "greedy_state_block_count_histogram",
        "per_node_block_count",
        "count_signature_blocks(",
    ]
    for token in forbidden:
        assert token not in text


def test_current_solver_uses_single_active_selector_source():
    root = _source_root()
    core_text = _current_solver_source_text()["core"]
    heuristic_text = _current_solver_source_text()["heuristic"]
    nonlinear_text = _current_solver_source_text()["selector"]
    init_text = (Path(__file__).resolve().parents[1] / "src" / "split" / "__init__.py").read_text(encoding="utf-8")

    assert '#include "msplit_nonlinear.cpp"' in core_text
    assert '#include "msplit_atomized.cpp"' in nonlinear_text
    assert "MSPLIT_USE_BACKUP_SELECTOR" not in core_text
    assert "current_depth == effective_lookahead_depth_" not in heuristic_text
    assert (root / "msplit_linear.cpp").exists()
    assert not (root / "msplit_exact_lazy.cpp").exists()
    assert "MSPLIT_RUSHDP" not in init_text


def test_current_sources_have_no_unused_family_trace_sink():
    text = "\n".join(_current_solver_source_text().values())

    assert "record_family1_hard_loss_inversion_trace" not in text
    assert "family1_hard_loss_inversion_traces" not in text


def test_current_core_default_lookahead_is_dynamic_half_depth():
    text = _current_solver_source_text()["core"]
    assert "std::max(1, (full_depth_budget_ + 1) / 2)" in text
    assert "effective_lookahead_depth_ =" in text


def test_dataset_uses_compact_reverse_feature_lookup():
    texts = _dataset_source_text()
    header = texts["header"]
    source = texts["source"]

    assert "m_binarized_to_original_feature" in header
    assert "std::vector<std::set<size_t>> m_feature_map" not in header
    assert "m_binarized_to_original_feature[binarized_feature_index]" in source
    assert "m_feature_map[i].find(binarized_feature_index)" not in source


def test_signature_summary_groups_rows_without_string_row_copies():
    core = _current_solver_source_text()["core"]

    assert "row_pattern_id_.assign(static_cast<size_t>(n_rows_), -1);" in core
    assert "std::unordered_map<int, CanonicalSignatureBlock> blocks_by_pattern;" in core
    assert "return row_pattern_less(row_ptr(lhs_row), row_ptr(rhs_row));" in core
    assert "encode_signature_code(" not in core


def test_nonlinear_support_canonicalizes_group_assignments():
    support = _current_solver_source_text()["support"]

    assert "static bool canonicalize_group_assignment(" in support
    assert "std::vector<int> canonical_assign = assign;" in support
    assert "if (!canonicalize_group_assignment(canonical_assign, groups))" in support
    assert "out.assignment = std::move(canonical_assign);" in support
    assert "if (lhs.assignment == rhs.assignment)" in support
    assert "atomized_candidate_dominates(impurity, misclassification)" in support
    assert "atomized_candidate_dominates(misclassification, impurity)" in support


def test_nonlinear_selector_above_lookahead_uses_anchor_exactification_not_pair_shortlist():
    heuristic = _current_solver_source_text()["heuristic"]

    assert "std::vector<CandidateEval> reference_candidates;" not in heuristic
    assert "resolve_pair_budget" not in heuristic
    assert "ensure_anchor" in heuristic
    assert "best_impurity_idx" in heuristic
    assert "best_hardloss_idx" in heuristic
    assert "early_stop_idx" not in heuristic


def test_nonlinear_selector_caches_duplicate_exact_child_evaluations():
    heuristic = _current_solver_source_text()["heuristic"]

    assert "struct CachedExactChildren" in heuristic
    assert "exact_child_eval_cache" in heuristic
    assert "nominee_child_partition_key" in heuristic


def test_nonlinear_impurity_weight_is_solver_state_not_env_switch():
    texts = _current_solver_source_text()
    text = "\n".join(texts.values())

    assert "MSPLIT_IMPURITY_METRIC" not in text
    assert "MSPLIT_SOFT_IMPURITY_WEIGHT" not in text
    assert "soft_impurity_weight_" in texts["core"]
