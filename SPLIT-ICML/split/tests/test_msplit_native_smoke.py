import numpy as np

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

    model = MSPLIT(
        full_depth_budget=4,
        lookahead_depth_budget=2,
        reg=0.02,
        min_child_size=1,
        max_branching=4,
        use_cpp_solver=True,
    )
    model.fit(X, y, sample_weight=w)

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

    model = MSPLIT(
        full_depth_budget=5,
        min_child_size=1,
        max_branching=4,
        use_cpp_solver=True,
    )
    model.fit(X, y)

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
    assert model.nominee_exactify_prefix_max_ <= 1
    assert isinstance(model.nominee_exactify_prefix_histogram_, list)


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
