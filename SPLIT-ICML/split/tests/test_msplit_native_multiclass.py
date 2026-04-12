import numpy as np

from split import MSPLIT


def test_native_multiclass_teacher_logits_smoke():
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
    y = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0], dtype=np.int32)
    teacher = np.array(
        [
            [4.0, 1.0, -1.0],
            [3.0, 1.0, -1.0],
            [2.5, 1.0, -0.5],
            [0.5, 3.5, -1.0],
            [0.0, 4.0, -1.0],
            [0.0, 4.5, -1.0],
            [-0.5, 3.5, 0.0],
            [-1.0, 1.0, 3.5],
            [-1.0, 0.5, 4.0],
            [-1.0, 0.0, 4.5],
            [-1.0, 0.0, 5.0],
            [3.5, 0.0, -0.5],
        ],
        dtype=np.float64,
    )

    model = MSPLIT(
        full_depth_budget=3,
        lookahead_depth=2,
        min_child_size=1,
        min_split_size=2,
        max_branching=3,
        time_limit=10,
    )
    model.fit(X, y, teacher_logit=teacher)

    proba = model.predict_proba(X)
    assert model.native_n_classes_ == 3
    assert model.native_teacher_class_count_ == 3
    assert model.native_binary_mode_ is False
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert model.atomized_features_prepared_ >= 1


def test_binary_teacher_matrix_matches_scalar_margin():
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
        lookahead_depth=2,
        min_child_size=1,
        min_split_size=2,
        max_branching=3,
        time_limit=10,
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
    assert margin_model.tree == matrix_model.tree


def test_exactify_top_k_smoke():
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
        lookahead_depth=2,
        min_child_size=1,
        min_split_size=2,
        max_branching=3,
        exactify_top_k=1,
        time_limit=10,
    )
    model.fit(X, y, teacher_logit=teacher)

    assert model.nominee_exactify_prefix_total_ >= 0
    assert model.nominee_exactify_prefix_max_ <= 1
