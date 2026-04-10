#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <matrix.hpp>
#include <configuration.hpp>
#include <gosdt.hpp>
#include <dataset.hpp>
#include <msplit.hpp>
#include <string>
#include <vector>
#include <cstring>

// #define STRINGIFY(x) #x
// #define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_libgosdt, m) {

    using BoolMatrix = Matrix<bool>;
    using FloatMatrix = Matrix<float>;

    // Input binary matrix class
    py::class_<BoolMatrix>(m, "BoolMatrix", py::buffer_protocol())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, bool>())
        .def("__getitem__",
                [](const BoolMatrix& bm, std::pair<size_t, size_t> tup) {
                    return bm(tup.first, tup.second);
                })
        .def("__setitem__",
                [](BoolMatrix& bm, std::pair<size_t, size_t> tup, bool value) {
                    bm(tup.first, tup.second) = value;
                })
        .def_buffer([](BoolMatrix &m) -> py::buffer_info {
                return py::buffer_info(
                        m.data(),
                        sizeof(bool),
                        py::format_descriptor<bool>::format(),
                        2,
                        { m.n_rows(), m.n_columns() },
                        { sizeof(bool) * m.n_columns(), sizeof(bool) }
                );
        });

    // float matrix class
    py::class_<FloatMatrix>(m, "FloatMatrix", py::buffer_protocol())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, float>())
        .def("__getitem__",
                [](const FloatMatrix& bm, std::pair<size_t, size_t> tup) {
                    return bm(tup.first, tup.second);
                })
        .def("__setitem__",
                [](FloatMatrix& bm, std::pair<size_t, size_t> tup, bool value) {
                    bm(tup.first, tup.second) = value;
                })
        .def_buffer([](FloatMatrix &m) -> py::buffer_info {
                return py::buffer_info(
                        m.data(),
                        sizeof(float),
                        py::format_descriptor<float>::format(),
                        2,
                        { m.n_rows(), m.n_columns() },
                        { sizeof(float) * m.n_columns(), sizeof(float) }
                );
        });

    // Configuration class
    py::class_<Configuration>(m, "Configuration")
        .def(py::init<>())
        .def_readwrite("regularization",                &Configuration::regularization)
        .def_readwrite("upperbound",                    &Configuration::upperbound_guess)
        .def_readwrite("time_limit",                    &Configuration::time_limit)
        .def_readwrite("worker_limit",                  &Configuration::worker_limit)
        .def_readwrite("model_limit",                   &Configuration::model_limit)
        .def_readwrite("verbose",                       &Configuration::verbose)
        .def_readwrite("diagnostics",                   &Configuration::diagnostics)
        .def_readwrite("depth_budget",                  &Configuration::depth_budget)
        .def_readwrite("reference_LB",                  &Configuration::reference_LB)
        .def_readwrite("look_ahead",                    &Configuration::look_ahead)
        .def_readwrite("similar_support",               &Configuration::similar_support)
        .def_readwrite("cancellation",                  &Configuration::cancellation)
        .def_readwrite("feature_transform",             &Configuration::feature_transform)
        .def_readwrite("rule_list",                     &Configuration::rule_list)
        .def_readwrite("non_binary",                    &Configuration::non_binary)
        .def_readwrite("trace",                         &Configuration::trace)
        .def_readwrite("tree",                          &Configuration::tree)
        .def_readwrite("profile",                       &Configuration::profile)
        .def_readwrite("cart_lookahead_depth",          &Configuration::cart_lookahead_depth)
        .def("__repr__", [](const Configuration& config) { return config.to_json().dump(); })
        // Provides Pickling support for the Configuration class:
        .def(py::pickle(
            // __getstate__
            [](const Configuration& config) {
                // Return a tuple that fully encodes the state of the object
                return py::make_tuple(
                    config.regularization,
                    config.upperbound_guess,
                    config.time_limit,
                    config.worker_limit,
                    config.model_limit,
                    config.verbose,
                    config.diagnostics,
                    config.depth_budget,
                    config.reference_LB,
                    config.look_ahead,
                    config.similar_support,
                    config.cancellation,
                    config.feature_transform,
                    config.rule_list,
                    config.non_binary,
                    config.trace,
                    config.tree,
                    config.profile, 
                    config.cart_lookahead_depth
                );
            },
            // __setstate__
            [](const py::tuple& t) {
                if (t.size() != 18) {
                    throw std::runtime_error("Invalid state!");
                }
                Configuration config;
                config.regularization = t[0].cast<float>();
                config.upperbound_guess = t[1].cast<float>();
                config.time_limit = t[2].cast<unsigned int>();
                config.worker_limit = t[3].cast<unsigned int>();
                config.model_limit = t[4].cast<unsigned int>();
                config.verbose = t[5].cast<bool>();
                config.diagnostics = t[6].cast<bool>();
                config.depth_budget = t[7].cast<unsigned char>();
                config.reference_LB = t[8].cast<bool>();
                config.look_ahead = t[9].cast<bool>();
                config.similar_support = t[10].cast<bool>();
                config.cancellation = t[11].cast<bool>();
                config.feature_transform = t[12].cast<bool>();
                config.rule_list = t[13].cast<bool>();
                config.non_binary = t[14].cast<bool>();
                config.trace = t[15].cast<std::string>();
                config.tree = t[16].cast<std::string>();
                config.profile = t[17].cast<std::string>();
                config.cart_lookahead_depth = t[18].cast<unsigned char>();
                return config;
            }
        ))
        .def("save", &Configuration::save);

    // gosdt::Result Class
    py::class_<gosdt::Result>(m, "GOSDTResult")
        .def(py::init<gosdt::Result>())
        .def_readonly("model",          &gosdt::Result::model)
        .def_readonly("graph_size",     &gosdt::Result::graph_size)
        .def_readonly("n_iterations",   &gosdt::Result::n_iterations)
        .def_readonly("lowerbound",     &gosdt::Result::lower_bound)
        .def_readonly("upperbound",     &gosdt::Result::upper_bound)
        .def_readonly("model_loss",     &gosdt::Result::model_loss)
        .def_readonly("time",           &gosdt::Result::time_elapsed)
        .def_readonly("status",         &gosdt::Result::status)
        .def(py::pickle(
            [](const gosdt::Result& result) {
                return py::make_tuple(
                    result.model,
                    result.graph_size,
                    result.n_iterations,
                    result.lower_bound,
                    result.upper_bound,
                    result.model_loss,
                    result.time_elapsed,
                    result.status
                );
            },
            [](const py::tuple& t) {
                if (t.size() != 8) {
                    throw std::runtime_error("Invalid state!");
                }
                gosdt::Result result;
                result.model = t[0].cast<std::string>();
                result.graph_size = t[1].cast<size_t>();
                result.n_iterations = t[2].cast<size_t>();
                result.lower_bound = t[3].cast<double>();
                result.upper_bound = t[4].cast<double>();
                result.model_loss = t[5].cast<double>();
                result.time_elapsed = t[6].cast<double>();
                result.status = t[7].cast<gosdt::Status>();
                return result;
            }
        ));

    // gosdt::fit function
    m.def("gosdt_fit", &gosdt::fit);
    m.def(
        "msplit_fit",
        [](py::array_t<int, py::array::c_style | py::array::forcecast> z,
           py::array_t<int, py::array::c_style | py::array::forcecast> y,
           py::object sample_weight,
           py::object teacher_logit,
           py::object teacher_boundary_gain,
           py::object teacher_boundary_cover,
           py::object teacher_boundary_value_jump,
           int full_depth_budget,
           int lookahead_depth,
           double regularization,
           int min_split_size,
           int min_child_size,
           double time_limit_seconds,
           int max_branching,
           int exactify_top_k) {
            if (z.ndim() != 2) {
                throw std::runtime_error("msplit_fit expects z to be a 2D int array.");
            }
            if (y.ndim() != 1) {
                throw std::runtime_error("msplit_fit expects y to be a 1D int array.");
            }
            if (z.shape(0) != y.shape(0)) {
                throw std::runtime_error("msplit_fit expects z.shape[0] == y.shape[0].");
            }

            const int n_rows = static_cast<int>(z.shape(0));
            const int n_features = static_cast<int>(z.shape(1));

            std::vector<int> z_flat(static_cast<size_t>(n_rows) * static_cast<size_t>(n_features));
            std::memcpy(z_flat.data(), z.data(), z_flat.size() * sizeof(int));

            std::vector<int> y_vec(static_cast<size_t>(n_rows));
            std::memcpy(y_vec.data(), y.data(), y_vec.size() * sizeof(int));

            std::vector<double> sample_weight_vec;
            if (!sample_weight.is_none()) {
                py::array_t<double, py::array::c_style | py::array::forcecast> sw =
                    sample_weight.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (sw.ndim() != 1 || sw.shape(0) != z.shape(0)) {
                    throw std::runtime_error(
                        "msplit_fit expects sample_weight to be None or a 1D float array with shape[0] == z.shape[0].");
                }
                sample_weight_vec.resize(static_cast<size_t>(n_rows));
                std::memcpy(sample_weight_vec.data(), sw.data(), sample_weight_vec.size() * sizeof(double));
            }

            std::vector<double> teacher_logit_vec;
            int teacher_class_count = 0;
            if (!teacher_logit.is_none()) {
                py::array_t<double, py::array::c_style | py::array::forcecast> teacher =
                    teacher_logit.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (teacher.ndim() == 1) {
                    if (teacher.shape(0) != z.shape(0)) {
                        throw std::runtime_error(
                            "msplit_fit expects 1D teacher_logit to have shape[0] == z.shape[0].");
                    }
                    teacher_class_count = 1;
                    teacher_logit_vec.resize(static_cast<size_t>(n_rows));
                    std::memcpy(teacher_logit_vec.data(), teacher.data(), teacher_logit_vec.size() * sizeof(double));
                } else if (teacher.ndim() == 2) {
                    if (teacher.shape(0) != z.shape(0) || teacher.shape(1) <= 0) {
                        throw std::runtime_error(
                            "msplit_fit expects 2D teacher_logit to have shape (n_rows, n_classes).");
                    }
                    teacher_class_count = static_cast<int>(teacher.shape(1));
                    teacher_logit_vec.resize(static_cast<size_t>(teacher.shape(0) * teacher.shape(1)));
                    std::memcpy(teacher_logit_vec.data(), teacher.data(), teacher_logit_vec.size() * sizeof(double));
                } else {
                    throw std::runtime_error(
                        "msplit_fit expects teacher_logit to be None, a 1D float array, or a 2D float array.");
                }
            }

            auto load_boundary_prior =
                [&](py::object obj, const char *name, std::vector<double> &out_vec, int &n_cols) {
                    if (obj.is_none()) {
                        return;
                    }
                    py::array_t<double, py::array::c_style | py::array::forcecast> arr =
                        obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                    if (arr.ndim() != 2 || arr.shape(0) != z.shape(1)) {
                        throw std::runtime_error(std::string("msplit_fit expects ") + name +
                                                 " to be None or a 2D float array with shape[0] == z.shape[1].");
                    }
                    if (n_cols < 0) {
                        n_cols = static_cast<int>(arr.shape(1));
                    } else if (n_cols != static_cast<int>(arr.shape(1))) {
                        throw std::runtime_error("Teacher boundary prior arrays must share the same shape.");
                    }
                    out_vec.resize(static_cast<size_t>(arr.shape(0) * arr.shape(1)));
                    std::memcpy(out_vec.data(), arr.data(), out_vec.size() * sizeof(double));
                };

            int teacher_boundary_cols = -1;
            std::vector<double> teacher_boundary_gain_vec;
            std::vector<double> teacher_boundary_cover_vec;
            std::vector<double> teacher_boundary_value_jump_vec;
            load_boundary_prior(teacher_boundary_gain, "teacher_boundary_gain", teacher_boundary_gain_vec, teacher_boundary_cols);
            load_boundary_prior(teacher_boundary_cover, "teacher_boundary_cover", teacher_boundary_cover_vec, teacher_boundary_cols);
            load_boundary_prior(teacher_boundary_value_jump, "teacher_boundary_value_jump", teacher_boundary_value_jump_vec, teacher_boundary_cols);
            if (teacher_boundary_cols < 0) {
                teacher_boundary_cols = 0;
            }

            msplit::FitResult solved = msplit::fit(
                z_flat,
                n_rows,
                n_features,
                y_vec,
                sample_weight_vec,
                teacher_logit_vec,
                teacher_class_count,
                teacher_boundary_gain_vec,
                teacher_boundary_cover_vec,
                teacher_boundary_value_jump_vec,
                teacher_boundary_cols,
                full_depth_budget,
                lookahead_depth,
                regularization,
                min_split_size,
                min_child_size,
                time_limit_seconds,
                max_branching,
                exactify_top_k);

            py::dict out;
            out["tree"] = py::str(solved.tree.dump());
            out["objective"] = solved.objective;
            out["greedy_internal_nodes"] = solved.greedy_internal_nodes;
            out["greedy_subproblem_calls"] = solved.greedy_subproblem_calls;
            out["exact_dp_subproblem_calls_above_lookahead"] = solved.exact_dp_subproblem_calls_above_lookahead;
            out["greedy_cache_hits"] = solved.greedy_cache_hits;
            out["greedy_unique_states"] = solved.greedy_unique_states;
            out["greedy_cache_entries_peak"] = solved.greedy_cache_entries_peak;
            out["greedy_cache_bytes_peak"] = solved.greedy_cache_bytes_peak;
            out["greedy_interval_evals"] = solved.greedy_interval_evals;
            out["elapsed_time_sec"] = solved.elapsed_time_sec;
            out["debr_refine_calls"] = solved.debr_refine_calls;
            out["debr_refine_improved"] = solved.debr_refine_improved;
            out["debr_total_moves"] = solved.debr_total_moves;
            out["debr_bridge_policy_calls"] = solved.debr_bridge_policy_calls;
            out["debr_refine_windowed_calls"] = solved.debr_refine_windowed_calls;
            out["debr_refine_unwindowed_calls"] = solved.debr_refine_unwindowed_calls;
            out["debr_refine_overlap_segments"] = solved.debr_refine_overlap_segments;
            out["debr_refine_calls_with_overlap"] = solved.debr_refine_calls_with_overlap;
            out["debr_refine_calls_without_overlap"] = solved.debr_refine_calls_without_overlap;
            out["debr_candidate_total"] = solved.debr_candidate_total;
            out["debr_candidate_legal"] = solved.debr_candidate_legal;
            out["debr_candidate_source_size_rejects"] = solved.debr_candidate_source_size_rejects;
            out["debr_candidate_target_size_rejects"] = solved.debr_candidate_target_size_rejects;
            out["debr_candidate_descent_eligible"] = solved.debr_candidate_descent_eligible;
            out["debr_candidate_descent_rejected"] = solved.debr_candidate_descent_rejected;
            out["debr_candidate_bridge_eligible"] = solved.debr_candidate_bridge_eligible;
            out["debr_candidate_bridge_window_blocked"] = solved.debr_candidate_bridge_window_blocked;
            out["debr_candidate_bridge_used_blocked"] = solved.debr_candidate_bridge_used_blocked;
            out["debr_candidate_bridge_guide_rejected"] = solved.debr_candidate_bridge_guide_rejected;
            out["debr_candidate_cleanup_eligible"] = solved.debr_candidate_cleanup_eligible;
            out["debr_candidate_cleanup_primary_rejected"] = solved.debr_candidate_cleanup_primary_rejected;
            out["debr_candidate_cleanup_complexity_rejected"] = solved.debr_candidate_cleanup_complexity_rejected;
            out["debr_candidate_score_rejected"] = solved.debr_candidate_score_rejected;
            out["debr_descent_moves"] = solved.debr_descent_moves;
            out["debr_bridge_moves"] = solved.debr_bridge_moves;
            out["debr_simplify_moves"] = solved.debr_simplify_moves;
            out["debr_source_group_row_size_histogram"] = solved.debr_source_group_row_size_histogram;
            out["debr_source_component_atom_size_histogram"] = solved.debr_source_component_atom_size_histogram;
            out["debr_source_component_row_size_histogram"] = solved.debr_source_component_row_size_histogram;
            out["debr_total_hard_gain"] = solved.debr_total_hard_gain;
            out["debr_total_soft_gain"] = solved.debr_total_soft_gain;
            out["debr_total_delta_j"] = solved.debr_total_delta_j;
            out["debr_total_component_delta"] = solved.debr_total_component_delta;
            out["debr_final_geo_wins"] = solved.debr_final_geo_wins;
            out["debr_final_block_wins"] = solved.debr_final_block_wins;
            out["family_compare_total"] = solved.family_compare_total;
            out["family_compare_equivalent"] = solved.family_compare_equivalent;
            out["family1_both_wins"] = solved.family1_both_wins;
            out["family2_hard_loss_wins"] = solved.family2_hard_loss_wins;
            out["family2_hard_impurity_wins"] = solved.family2_hard_impurity_wins;
            out["family2_both_wins"] = solved.family2_both_wins;
            out["family_metric_disagreement"] = solved.family_metric_disagreement;
            out["family_hard_loss_ties"] = solved.family_hard_loss_ties;
            out["family_hard_impurity_ties"] = solved.family_hard_impurity_ties;
            out["family_joint_impurity_ties"] = solved.family_joint_impurity_ties;
            out["family_neither_both_wins"] = solved.family_neither_both_wins;
            out["family1_selected_by_equivalence"] = solved.family1_selected_by_equivalence;
            out["family1_selected_by_dominance"] = solved.family1_selected_by_dominance;
            out["family2_selected_by_dominance"] = solved.family2_selected_by_dominance;
            out["family_sent_both"] = solved.family_sent_both;
            out["family1_hard_loss_sum"] = solved.family1_hard_loss_sum;
            out["family2_hard_loss_sum"] = solved.family2_hard_loss_sum;
            out["family_hard_loss_delta_sum"] = solved.family_hard_loss_delta_sum;
            out["family1_hard_impurity_sum"] = solved.family1_hard_impurity_sum;
            out["family2_hard_impurity_sum"] = solved.family2_hard_impurity_sum;
            out["family_hard_impurity_delta_sum"] = solved.family_hard_impurity_delta_sum;
            out["family1_joint_impurity_sum"] = solved.family1_joint_impurity_sum;
            out["family2_joint_impurity_sum"] = solved.family2_joint_impurity_sum;
            out["family_joint_impurity_delta_sum"] = solved.family_joint_impurity_delta_sum;
            out["family1_soft_impurity_sum"] = solved.family1_soft_impurity_sum;
            out["family2_soft_impurity_sum"] = solved.family2_soft_impurity_sum;
            out["family_soft_impurity_delta_sum"] = solved.family_soft_impurity_delta_sum;
            out["family2_joint_impurity_wins"] = solved.family2_joint_impurity_wins;
            out["teacher_available"] = solved.teacher_available;
            out["native_teacher_available"] = solved.teacher_available;
            out["n_classes"] = solved.n_classes;
            out["teacher_class_count"] = solved.teacher_class_count;
            out["binary_mode"] = solved.binary_mode;
            out["native_n_classes"] = solved.n_classes;
            out["native_teacher_class_count"] = solved.teacher_class_count;
            out["native_binary_mode"] = solved.binary_mode;
            out["atomized_features_prepared"] = solved.atomized_features_prepared;
            out["atomized_coarse_candidates"] = solved.atomized_coarse_candidates;
            out["atomized_final_candidates"] = solved.atomized_final_candidates;
            out["atomized_coarse_pruned_candidates"] = solved.atomized_coarse_pruned_candidates;
            out["atomized_compression_features_applied"] = solved.atomized_compression_features_applied;
            out["atomized_compression_features_collapsed_to_single_block"] =
                solved.atomized_compression_features_collapsed_to_single_block;
            out["atomized_compression_atoms_before_total"] = solved.atomized_compression_atoms_before_total;
            out["atomized_compression_blocks_after_total"] = solved.atomized_compression_blocks_after_total;
            out["atomized_compression_atoms_merged_total"] = solved.atomized_compression_atoms_merged_total;
            out["greedy_feature_survivor_histogram"] = solved.greedy_feature_survivor_histogram;
            out["nominee_unique_total"] = solved.nominee_unique_total;
            out["nominee_child_interval_lookups"] = solved.nominee_child_interval_lookups;
            out["nominee_child_interval_unique"] = solved.nominee_child_interval_unique;
            out["nominee_exactified_total"] = solved.nominee_exactified_total;
            out["nominee_incumbent_updates"] = solved.nominee_incumbent_updates;
            out["nominee_threatening_samples"] = solved.nominee_threatening_samples;
            out["nominee_threatening_sum"] = solved.nominee_threatening_sum;
            out["nominee_threatening_max"] = solved.nominee_threatening_max;
            out["nominee_certificate_nodes"] = solved.nominee_certificate_nodes;
            out["nominee_certificate_exhausted_nodes"] = solved.nominee_certificate_exhausted_nodes;
            out["nominee_exactified_until_certificate_total"] = solved.nominee_exactified_until_certificate_total;
            out["nominee_exactified_until_certificate_max"] = solved.nominee_exactified_until_certificate_max;
            out["nominee_certificate_min_remaining_lower_bound_sum"] =
                solved.nominee_certificate_min_remaining_lower_bound_sum;
            out["nominee_certificate_min_remaining_lower_bound_max"] =
                solved.nominee_certificate_min_remaining_lower_bound_max;
            out["nominee_certificate_incumbent_exact_score_sum"] =
                solved.nominee_certificate_incumbent_exact_score_sum;
            out["nominee_certificate_incumbent_exact_score_max"] =
                solved.nominee_certificate_incumbent_exact_score_max;
            out["nominee_exactified_until_certificate_histogram"] =
                solved.nominee_exactified_until_certificate_histogram;
            out["nominee_certificate_stop_depth_histogram"] =
                solved.nominee_certificate_stop_depth_histogram;
            out["nominee_elbow_prefix_total"] = solved.nominee_elbow_prefix_total;
            out["nominee_elbow_prefix_max"] = solved.nominee_elbow_prefix_max;
            out["nominee_elbow_prefix_histogram"] = solved.nominee_elbow_prefix_histogram;
            out["nominee_exact_child_eval_sec"] = solved.profiling_recursive_child_eval_sec;
            out["nominee_debr_sec"] = solved.profiling_refine_sec;
            out["atomized_feature_atom_count_histogram"] = solved.atomized_feature_atom_count_histogram;
            out["atomized_feature_block_atom_count_histogram"] = solved.atomized_feature_block_atom_count_histogram;
            out["atomized_feature_q_effective_histogram"] = solved.atomized_feature_q_effective_histogram;
            out["greedy_feature_preserved_histogram"] = solved.greedy_feature_preserved_histogram;
            out["greedy_candidate_count_histogram"] = solved.greedy_candidate_count_histogram;
            out["per_node_prepared_features"] = solved.per_node_prepared_features;
            out["per_node_candidate_count"] = solved.per_node_candidate_count;
            out["per_node_total_weight"] = solved.per_node_total_weight;
            out["per_node_mu_node"] = solved.per_node_mu_node;
            out["per_node_candidate_upper_bounds"] = solved.per_node_candidate_upper_bounds;
            out["per_node_candidate_lower_bounds"] = solved.per_node_candidate_lower_bounds;
            out["per_node_candidate_hard_loss"] = solved.per_node_candidate_hard_loss;
            out["per_node_candidate_impurity_objective"] = solved.per_node_candidate_impurity_objective;
            out["per_node_candidate_hard_impurity"] = solved.per_node_candidate_hard_impurity;
            out["per_node_candidate_soft_impurity"] = solved.per_node_candidate_soft_impurity;
            out["per_node_candidate_boundary_penalty"] = solved.per_node_candidate_boundary_penalty;
            out["per_node_candidate_components"] = solved.per_node_candidate_components;
            out["profiling_lp_solve_calls"] = solved.profiling_lp_solve_calls;
            out["profiling_lp_solve_sec"] = solved.profiling_lp_solve_sec;
            out["profiling_pricing_calls"] = solved.profiling_pricing_calls;
            out["profiling_pricing_sec"] = solved.profiling_pricing_sec;
            out["profiling_greedy_complete_calls"] = solved.profiling_greedy_complete_calls;
            out["profiling_greedy_complete_sec"] = solved.profiling_greedy_complete_sec;
            out["profiling_greedy_complete_calls_by_depth"] = solved.profiling_greedy_complete_calls_by_depth;
            out["profiling_feature_prepare_sec"] = solved.profiling_feature_prepare_sec;
            out["profiling_candidate_nomination_sec"] = solved.profiling_candidate_nomination_sec;
            out["profiling_candidate_shortlist_sec"] = solved.profiling_candidate_shortlist_sec;
            out["profiling_candidate_generation_sec"] = solved.profiling_candidate_generation_sec;
            out["profiling_recursive_child_eval_sec"] = solved.profiling_recursive_child_eval_sec;
            out["heuristic_selector_nodes"] = solved.heuristic_selector_nodes;
            out["heuristic_selector_candidate_total"] = solved.heuristic_selector_candidate_total;
            out["heuristic_selector_candidate_pruned_total"] =
                solved.heuristic_selector_candidate_pruned_total;
            out["heuristic_selector_survivor_total"] = solved.heuristic_selector_survivor_total;
            out["heuristic_selector_leaf_optimal_nodes"] = solved.heuristic_selector_leaf_optimal_nodes;
            out["heuristic_selector_improving_split_nodes"] = solved.heuristic_selector_improving_split_nodes;
            out["heuristic_selector_improving_split_retained_nodes"] =
                solved.heuristic_selector_improving_split_retained_nodes;
            out["heuristic_selector_improving_split_margin_sum"] =
                solved.heuristic_selector_improving_split_margin_sum;
            out["heuristic_selector_improving_split_margin_max"] =
                solved.heuristic_selector_improving_split_margin_max;
            out["heuristic_selector_nodes_by_depth"] = solved.heuristic_selector_nodes_by_depth;
            out["heuristic_selector_candidate_total_by_depth"] =
                solved.heuristic_selector_candidate_total_by_depth;
            out["heuristic_selector_candidate_pruned_total_by_depth"] =
                solved.heuristic_selector_candidate_pruned_total_by_depth;
            out["heuristic_selector_survivor_total_by_depth"] =
                solved.heuristic_selector_survivor_total_by_depth;
            out["heuristic_selector_leaf_optimal_nodes_by_depth"] =
                solved.heuristic_selector_leaf_optimal_nodes_by_depth;
            out["heuristic_selector_improving_split_nodes_by_depth"] =
                solved.heuristic_selector_improving_split_nodes_by_depth;
            out["heuristic_selector_improving_split_retained_nodes_by_depth"] =
                solved.heuristic_selector_improving_split_retained_nodes_by_depth;
            out["heuristic_selector_improving_split_margin_sum_by_depth"] =
                solved.heuristic_selector_improving_split_margin_sum_by_depth;
            out["heuristic_selector_improving_split_margin_max_by_depth"] =
                solved.heuristic_selector_improving_split_margin_max_by_depth;
            out["profiling_refine_calls"] = solved.profiling_refine_calls;
            out["profiling_refine_sec"] = solved.profiling_refine_sec;
            return out;
        },
        py::arg("z"),
        py::arg("y"),
        py::arg("sample_weight") = py::none(),
        py::arg("teacher_logit") = py::none(),
        py::arg("teacher_boundary_gain") = py::none(),
        py::arg("teacher_boundary_cover") = py::none(),
        py::arg("teacher_boundary_value_jump") = py::none(),
        py::arg("full_depth_budget"),
        py::arg("lookahead_depth") = 3,
        py::arg("regularization"),
        py::arg("min_split_size") = 0,
        py::arg("min_child_size"),
        py::arg("time_limit_seconds") = 0.0,
        py::arg("max_branching") = 0,
        py::arg("exactify_top_k") = 0);

    m.def(
        "msplit_debug_run_atomized_smoke_cases",
        []() { return py::str(msplit::debug_run_atomized_smoke_cases().dump()); });

    // Define Status enum
    py::enum_<gosdt::Status>(m, "Status")
        .value("CONVERGED",         gosdt::Status::CONVERGED)
        .value("TIMEOUT",           gosdt::Status::TIMEOUT)
        .value("NON_CONVERGENCE",   gosdt::Status::NON_CONVERGENCE)
        .value("FALSE_CONVERGENCE", gosdt::Status::FALSE_CONVERGENCE)
        .value("UNINITIALIZED",     gosdt::Status::UNINITIALIZED)
        .export_values();

    // Encoding class for translating between original features and binarized features.
    py::class_<Dataset>(m, "Dataset")
        .def(py::init<const Configuration&, const Matrix<bool>&, const Matrix<float>&, const std::vector<std::set<size_t>>&>())
        .def(py::init<const Configuration&, const Matrix<bool>&, const Matrix<float>&, const std::vector<std::set<size_t>>&, const Matrix<bool>&>())
        .def_readonly("n_rows",     &Dataset::m_number_rows)
        .def_readonly("n_features", &Dataset::m_number_features)
        .def_readonly("n_targets",  &Dataset::m_number_targets)
        .def("save",                &Dataset::save);
        // .def_static("load",         &Dataset::load);

}
