import numpy as np

from benchmark.scripts.benchmark_cached_msplit import run_cached_msplit
from benchmark.scripts.cache_utils import (
    DEFAULT_CACHE_SEED,
    DEFAULT_MAX_BINS,
    DEFAULT_MIN_CHILD_SIZE,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    default_cache_path,
    load_cache,
)
from benchmark.scripts.msplit_benchmark_defaults import (
    DEFAULT_EXACTIFY_TOP_K,
    DEFAULT_LOOKAHEAD_DEPTH,
    DEFAULT_MAX_BRANCHING,
    DEFAULT_MIN_SPLIT_SIZE,
    DEFAULT_REG,
)
from split import MSPLIT


def test_native_wrapper_weighted_cpp_smoke():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)
    w = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    teacher = np.array([-3.0, -2.0, 2.0, 3.0, -1.5, 1.5, -0.5, 0.5], dtype=np.float64)

    model = MSPLIT(
        full_depth_budget=4,
        lookahead_depth_budget=2,
        reg=0.02,
        min_child_size=1,
        max_branching=4,
        use_cpp_solver=True,
    )
    model.fit(X, y, sample_weight=w, teacher_logit=teacher)

    assert model.lower_bound_ <= model.upper_bound_ + 1e-9
    assert model.objective_ <= model.upper_bound_ + 1e-9
    assert model.objective_ >= model.lower_bound_ - 1e-9
    assert model.native_binary_mode_ is True


def test_native_wrapper_default_lookahead_is_half_depth():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
    teacher = np.array([-3.0, -2.0, 2.0, 3.0, -1.0, 1.0], dtype=np.float64)

    model = MSPLIT(
        full_depth_budget=5,
        min_child_size=1,
        max_branching=4,
        use_cpp_solver=True,
    )
    model.fit(X, y, teacher_logit=teacher)

    assert model.effective_lookahead_depth_ == 3
    assert model.lower_bound_ <= model.upper_bound_ + 1e-9


def test_native_wrapper_exactify_metrics_exposed():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
            [4, 0],
            [4, 1],
            [5, 0],
            [5, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32)
    teacher = np.array([-4.0, -3.0, -2.0, 1.0, 2.0, 3.0, 2.5, -1.0, -2.5, 1.5, 3.5, -3.5], dtype=np.float64)

    model = MSPLIT(
        full_depth_budget=3,
        lookahead_depth_budget=2,
        reg=0.0,
        min_child_size=1,
        min_split_size=2,
        max_branching=3,
        exactify_top_k=1,
        use_cpp_solver=True,
    )
    model.fit(X, y, teacher_logit=teacher)

    assert model.native_teacher_class_count_ == 1
    assert model.nominee_exactify_prefix_total_ >= 0
    assert 1 <= model.nominee_exactify_prefix_max_ <= 2
    assert isinstance(model.nominee_exactify_prefix_histogram_, list)


def test_native_wrapper_lookahead_depth_one_is_greedy_from_root():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    teacher = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)

    cpp_model = MSPLIT(
        full_depth_budget=3,
        lookahead_depth_budget=1,
        reg=0.0,
        min_child_size=1,
        max_branching=4,
        use_cpp_solver=True,
    ).fit(X, y, teacher_logit=teacher)

    py_model = MSPLIT(
        full_depth_budget=3,
        lookahead_depth_budget=1,
        reg=0.0,
        min_child_size=1,
        max_branching=4,
        use_cpp_solver=False,
    ).fit(X, y, teacher_logit=teacher)

    assert cpp_model.effective_lookahead_depth_ == 1
    assert cpp_model.exact_dp_subproblem_calls_above_lookahead_ == 0
    assert cpp_model.greedy_subproblem_calls_ > 0

    assert py_model.effective_lookahead_depth_ == 1
    assert py_model.exact_dp_subproblem_calls_above_lookahead_ == 0
    assert py_model.greedy_subproblem_calls_ > 0


def test_native_wrapper_rejects_non_positive_lookahead_depth():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    for kwargs, expected_message in [
        ({"lookahead_depth": 0}, "lookahead_depth"),
        ({"lookahead_depth_budget": -1}, "lookahead_depth_budget"),
    ]:
        model = MSPLIT(
            full_depth_budget=3,
            reg=0.0,
            min_child_size=1,
            max_branching=4,
            use_cpp_solver=False,
            **kwargs,
        )
        try:
            model.fit(X, y)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError("Expected non-positive lookahead depth to raise ValueError.")


def test_native_cached_electricity_depth6_smoke(monkeypatch):
    monkeypatch.setenv("MSDT_BENCHMARK_GUARD", "0")
    monkeypatch.setenv("MSPLIT_BUILD_DIR", "build-nonlinear-py")

    cache_path = default_cache_path(
        dataset="electricity",
        seed=DEFAULT_CACHE_SEED,
        test_size=DEFAULT_TEST_SIZE,
        val_size=DEFAULT_VAL_SIZE,
        max_bins=DEFAULT_MAX_BINS,
        min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
        min_child_size=DEFAULT_MIN_CHILD_SIZE,
    )
    cache = load_cache(cache_path)

    result = run_cached_msplit(
        cache=cache,
        depth=6,
        lookahead_depth=DEFAULT_LOOKAHEAD_DEPTH,
        reg=DEFAULT_REG,
        exactify_top_k=DEFAULT_EXACTIFY_TOP_K,
        min_split_size=DEFAULT_MIN_SPLIT_SIZE,
        min_child_size=DEFAULT_MIN_CHILD_SIZE,
        max_branching=DEFAULT_MAX_BRANCHING,
        worker_limit=1,
        include_tree=False,
        include_diagnostics=False,
        verbose=False,
    )

    assert result["fit_seconds"] < 30.0
    assert result["n_internal"] >= 1
    assert result["test_accuracy"] >= 0.87


def test_native_wrapper_parallel_exactify_matches_serial():
    X = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [2, 0, 0],
            [2, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32)
    teacher = np.array(
        [-4.0, -3.0, 1.0, 2.0, -2.5, 2.5, 3.0, -1.0, -3.5, 2.0, 3.5, -2.0],
        dtype=np.float64,
    )

    kwargs = dict(
        full_depth_budget=3,
        lookahead_depth_budget=2,
        reg=0.0,
        min_child_size=1,
        min_split_size=2,
        max_branching=3,
        exactify_top_k=2,
        use_cpp_solver=True,
    )

    serial_model = MSPLIT(worker_limit=1, **kwargs).fit(X, y, teacher_logit=teacher)
    parallel_model = MSPLIT(worker_limit=2, **kwargs).fit(X, y, teacher_logit=teacher)

    assert serial_model.nominee_exactify_prefix_max_ >= 2
    assert abs(serial_model.objective_ - parallel_model.objective_) <= 1e-12
    assert serial_model.tree == parallel_model.tree
    assert np.array_equal(serial_model.predict(X), parallel_model.predict(X))


def test_native_wrapper_parallel_exactify_repeated_fit_matches_serial():
    X = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [2, 0, 0],
            [2, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
        ],
        dtype=np.int32,
    )
    y_first = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32)
    y_second = 1 - y_first
    teacher = np.array(
        [-4.0, -3.0, 1.0, 2.0, -2.5, 2.5, 3.0, -1.0, -3.5, 2.0, 3.5, -2.0],
        dtype=np.float64,
    )

    kwargs = dict(
        full_depth_budget=3,
        lookahead_depth_budget=2,
        reg=0.0,
        min_child_size=1,
        min_split_size=2,
        max_branching=3,
        exactify_top_k=2,
        use_cpp_solver=True,
    )

    MSPLIT(worker_limit=2, **kwargs).fit(X, y_first, teacher_logit=teacher)
    parallel_second = MSPLIT(worker_limit=2, **kwargs).fit(X, y_second, teacher_logit=teacher)
    serial_second = MSPLIT(worker_limit=1, **kwargs).fit(X, y_second, teacher_logit=teacher)

    assert abs(serial_second.objective_ - parallel_second.objective_) <= 1e-12
    assert serial_second.tree == parallel_second.tree
    assert np.array_equal(serial_second.predict(X), parallel_second.predict(X))


def test_native_wrapper_parallel_child_recursion_matches_serial():
    X = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [2, 0, 0],
            [2, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
        ],
        dtype=np.int32,
    )
    y = np.array(
        [
            0, 0, 1, 1,  # group 0 prefers feature 1
            0, 1, 0, 1,  # group 1 prefers feature 2
            1, 1, 0, 0,  # group 2 prefers inverted feature 1
        ],
        dtype=np.int32,
    )
    teacher = np.where(y == 1, 4.0, -4.0).astype(np.float64)

    kwargs = dict(
        full_depth_budget=3,
        lookahead_depth_budget=2,
        reg=0.0,
        min_child_size=1,
        min_split_size=2,
        max_branching=3,
        exactify_top_k=1,
        use_cpp_solver=True,
    )

    serial_model = MSPLIT(worker_limit=1, **kwargs).fit(X, y, teacher_logit=teacher)
    parallel_model = MSPLIT(worker_limit=4, **kwargs).fit(X, y, teacher_logit=teacher)

    def count_internal(node):
        if getattr(node, "is_leaf", False):
            return 0
        return 1 + sum(count_internal(child) for child in getattr(node, "children", []))

    assert serial_model.nominee_exactify_prefix_max_ == 1
    assert count_internal(serial_model.tree_) >= 3
    assert abs(serial_model.objective_ - parallel_model.objective_) <= 1e-12
    assert serial_model.tree == parallel_model.tree
    assert np.array_equal(serial_model.predict(X), parallel_model.predict(X))


def test_native_wrapper_cpp_requires_teacher_guidance():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    model = MSPLIT(
        full_depth_budget=3,
        lookahead_depth_budget=2,
        min_child_size=1,
        min_split_size=2,
        max_branching=3,
        use_cpp_solver=True,
    )

    try:
        model.fit(X, y)
    except ValueError as exc:
        assert "teacher_logit" in str(exc)
    else:
        raise AssertionError("Expected the native reference-guided solver to require teacher_logit.")


def test_native_wrapper_binary_teacher_matrix_matches_margin_vector():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.int32)
    margin = np.array([-3.0, -2.5, -1.5, 1.0, 2.0, 2.5, -0.5, 1.5], dtype=np.float64)
    teacher_matrix = np.column_stack([-0.5 * margin, 0.5 * margin]).astype(np.float64)

    kwargs = dict(
        full_depth_budget=3,
        lookahead_depth_budget=2,
        reg=0.0,
        min_child_size=1,
        min_split_size=2,
        max_branching=3,
        use_cpp_solver=True,
    )

    margin_model = MSPLIT(**kwargs).fit(X, y, teacher_logit=margin)
    matrix_model = MSPLIT(**kwargs).fit(X, y, teacher_logit=teacher_matrix)

    assert margin_model.native_binary_mode_ is True
    assert matrix_model.native_binary_mode_ is True
    assert margin_model.native_teacher_class_count_ == 1
    assert matrix_model.native_teacher_class_count_ == 2
    assert np.array_equal(margin_model.predict(X), matrix_model.predict(X))
    assert np.allclose(margin_model.predict_proba(X), matrix_model.predict_proba(X))
    assert abs(margin_model.objective_ - matrix_model.objective_) <= 1e-12
