#include "msplit.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <list>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

namespace msplit {

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kEpsUpdate = 1e-12;

static bool env_flag_enabled(const char *name) {
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }
    return !(raw[0] == '0' && raw[1] == '\0');
}

struct ScopedTimer {
    double *accumulator = nullptr;
    Clock::time_point start{};

    explicit ScopedTimer(double &acc, bool enabled = true)
        : accumulator(enabled ? &acc : nullptr),
          start(enabled ? Clock::now() : Clock::time_point{}) {}

    ~ScopedTimer() {
        if (accumulator != nullptr) {
            *accumulator += std::chrono::duration<double>(Clock::now() - start).count();
        }
    }
};

static double hard_label_impurity(double pos_weight, double neg_weight) {
    const double total = pos_weight + neg_weight;
    if (total <= kEpsUpdate) {
        return 0.0;
    }
    return total - ((pos_weight * pos_weight + neg_weight * neg_weight) / total);
}

static double split_leaf_loss(double pos_weight, double neg_weight) {
    return std::min(pos_weight, neg_weight);
}

static double hard_label_impurity(const std::vector<double> &class_weight) {
    double total = 0.0;
    double sum_sq = 0.0;
    for (double value : class_weight) {
        total += value;
        sum_sq += value * value;
    }
    if (total <= kEpsUpdate) {
        return 0.0;
    }
    return total - (sum_sq / total);
}

static double split_leaf_loss(const std::vector<double> &class_weight) {
    double total = 0.0;
    double best = 0.0;
    for (double value : class_weight) {
        total += value;
        best = std::max(best, value);
    }
    return (total > best) ? (total - best) : 0.0;
}

static int argmax_index(const std::vector<double> &values) {
    if (values.empty()) {
        return 0;
    }
    return static_cast<int>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

static double hard_label_impurity_flat(
    const std::vector<double> &storage,
    int n_classes,
    size_t base
) {
    double total = 0.0;
    double sum_sq = 0.0;
    for (int cls = 0; cls < n_classes; ++cls) {
        const double value = storage[base + static_cast<size_t>(cls)];
        total += value;
        sum_sq += value * value;
    }
    if (total <= kEpsUpdate) {
        return 0.0;
    }
    return total - (sum_sq / total);
}

static double split_leaf_loss_flat(
    const std::vector<double> &storage,
    int n_classes,
    size_t base
) {
    double total = 0.0;
    double best = 0.0;
    for (int cls = 0; cls < n_classes; ++cls) {
        const double value = storage[base + static_cast<size_t>(cls)];
        total += value;
        best = std::max(best, value);
    }
    return (total > best) ? (total - best) : 0.0;
}

static double split_leaf_loss_flat_delta(
    const std::vector<double> &storage,
    int n_classes,
    size_t base,
    const std::vector<double> &delta,
    bool add_delta
) {
    double total = 0.0;
    double best = 0.0;
    for (int cls = 0; cls < n_classes; ++cls) {
        const double raw = storage[base + static_cast<size_t>(cls)];
        const double value = add_delta ? (raw + delta[(size_t)cls]) : (raw - delta[(size_t)cls]);
        total += value;
        best = std::max(best, value);
    }
    return (total > best) ? (total - best) : 0.0;
}

static double hard_label_impurity_flat_delta(
    const std::vector<double> &storage,
    int n_classes,
    size_t base,
    const std::vector<double> &delta,
    bool add_delta
) {
    double total = 0.0;
    double sum_sq = 0.0;
    for (int cls = 0; cls < n_classes; ++cls) {
        const double raw = storage[base + static_cast<size_t>(cls)];
        const double value = add_delta ? (raw + delta[(size_t)cls]) : (raw - delta[(size_t)cls]);
        total += value;
        sum_sq += value * value;
    }
    if (total <= kEpsUpdate) {
        return 0.0;
    }
    return total - (sum_sq / total);
}

static double hard_label_impurity_prefix_segment(
    const std::vector<double> &prefix,
    int n_classes,
    int start,
    int end_exclusive
) {
    const size_t start_base = static_cast<size_t>(start) * static_cast<size_t>(n_classes);
    const size_t end_base = static_cast<size_t>(end_exclusive) * static_cast<size_t>(n_classes);
    double total = 0.0;
    double sum_sq = 0.0;
    for (int cls = 0; cls < n_classes; ++cls) {
        const double value =
            prefix[end_base + static_cast<size_t>(cls)] -
            prefix[start_base + static_cast<size_t>(cls)];
        total += value;
        sum_sq += value * value;
    }
    if (total <= kEpsUpdate) {
        return 0.0;
    }
    return total - (sum_sq / total);
}

static double split_leaf_loss_prefix_segment(
    const std::vector<double> &prefix,
    int n_classes,
    int start,
    int end_exclusive
) {
    const size_t start_base = static_cast<size_t>(start) * static_cast<size_t>(n_classes);
    const size_t end_base = static_cast<size_t>(end_exclusive) * static_cast<size_t>(n_classes);
    double total = 0.0;
    double best = 0.0;
    for (int cls = 0; cls < n_classes; ++cls) {
        const double value =
            prefix[end_base + static_cast<size_t>(cls)] -
            prefix[start_base + static_cast<size_t>(cls)];
        total += value;
        best = std::max(best, value);
    }
    return (total > best) ? (total - best) : 0.0;
}

struct Node {
    bool is_leaf = true;

    int prediction = 0;
    double loss = 0.0;
    int n_samples = 0;
    int neg_count = 0;
    int pos_count = 0;
    std::vector<int> class_counts;

    int feature = -1;
    int fallback_bin = -1;
    int fallback_prediction = 0;
    int group_count = 0;
    std::vector<std::vector<std::pair<int, int>>> group_bin_spans;
    std::vector<std::shared_ptr<Node>> group_nodes;
};

struct GreedyResult {
    double objective = 0.0;
    std::shared_ptr<Node> tree;
};

struct OrderedBins {
    std::vector<int> values;
    std::vector<std::vector<int>> members;
};

struct GreedyCacheEntry {
    GreedyResult result;
    size_t bytes = 0;
};

class Solver {
   public:
    struct CanonicalSignatureBlock {
        std::string code_key;
        int row_count = 0;
        double pos_weight = 0.0;
        double neg_weight = 0.0;
        std::vector<double> class_weight;
    };

    struct CanonicalSignatureSummary {
        double leaf_objective = 0.0;
        double signature_bound = 0.0;
        size_t block_count = 0U;
    };

    struct AtomizedTelemetry {
        long long &debr_refine_calls;
        long long &debr_refine_improved;
        long long &debr_total_moves;
        long long &debr_bridge_policy_calls;
        long long &debr_refine_windowed_calls;
        long long &debr_refine_unwindowed_calls;
        long long &debr_refine_overlap_segments;
        long long &debr_refine_calls_with_overlap;
        long long &debr_refine_calls_without_overlap;
        long long &debr_candidate_total;
        long long &debr_candidate_legal;
        long long &debr_candidate_source_size_rejects;
        long long &debr_candidate_target_size_rejects;
        long long &debr_candidate_descent_eligible;
        long long &debr_candidate_descent_rejected;
        long long &debr_candidate_bridge_eligible;
        long long &debr_candidate_bridge_window_blocked;
        long long &debr_candidate_bridge_used_blocked;
        long long &debr_candidate_bridge_guide_rejected;
        long long &debr_candidate_cleanup_eligible;
        long long &debr_candidate_cleanup_primary_rejected;
        long long &debr_candidate_cleanup_complexity_rejected;
        long long &debr_candidate_score_rejected;
        long long &debr_descent_moves;
        long long &debr_bridge_moves;
        long long &debr_simplify_moves;
        std::vector<long long> &debr_source_group_row_size_histogram;
        std::vector<long long> &debr_source_component_atom_size_histogram;
        std::vector<long long> &debr_source_component_row_size_histogram;
        double &debr_total_hard_gain;
        double &debr_total_soft_gain;
        double &debr_total_delta_j;
        long long &debr_total_component_delta;
        long long &greedy_interval_evals;
        long long &atomized_features_prepared;
        long long &atomized_coarse_candidates;
        long long &atomized_final_candidates;
        long long &atomized_coarse_pruned_candidates;
        long long &atomized_compression_features_applied;
        long long &atomized_compression_features_collapsed_to_single_block;
        long long &atomized_compression_atoms_before_total;
        long long &atomized_compression_blocks_after_total;
        long long &atomized_compression_atoms_merged_total;
        std::vector<long long> &greedy_feature_survivor_histogram;
        std::vector<long long> &atomized_feature_atom_count_histogram;
        std::vector<long long> &atomized_feature_block_atom_count_histogram;
        std::vector<long long> &atomized_feature_q_effective_histogram;
        long long &debr_final_geo_wins;
        long long &debr_final_block_wins;
        long long &family_compare_total;
        long long &family_compare_equivalent;
        long long &family1_both_wins;
        long long &family2_hard_loss_wins;
        long long &family2_hard_impurity_wins;
        long long &family2_joint_impurity_wins;
        long long &family2_both_wins;
        long long &family_metric_disagreement;
        long long &family_hard_loss_ties;
        long long &family_hard_impurity_ties;
        long long &family_joint_impurity_ties;
        long long &family_neither_both_wins;
        long long &family1_selected_by_equivalence;
        long long &family1_selected_by_dominance;
        long long &family2_selected_by_dominance;
        long long &family_sent_both;
        double &family1_hard_loss_sum;
        double &family2_hard_loss_sum;
        double &family_hard_loss_delta_sum;
        double &family1_hard_impurity_sum;
        double &family2_hard_impurity_sum;
        double &family_hard_impurity_delta_sum;
        double &family1_joint_impurity_sum;
        double &family2_joint_impurity_sum;
        double &family_joint_impurity_delta_sum;
        double &family1_soft_impurity_sum;
        double &family2_soft_impurity_sum;
        double &family_soft_impurity_delta_sum;
        std::vector<nlohmann::json> &family1_hard_loss_inversion_traces;
        long long &heuristic_selector_nodes;
        long long &heuristic_selector_candidate_total;
        long long &heuristic_selector_candidate_pruned_total;
        long long &heuristic_selector_survivor_total;
        long long &heuristic_selector_leaf_optimal_nodes;
        long long &heuristic_selector_improving_split_nodes;
        long long &heuristic_selector_improving_split_retained_nodes;
        double &heuristic_selector_improving_split_margin_sum;
        double &heuristic_selector_improving_split_margin_max;
        std::vector<long long> &heuristic_selector_nodes_by_depth;
        std::vector<long long> &heuristic_selector_candidate_total_by_depth;
        std::vector<long long> &heuristic_selector_candidate_pruned_total_by_depth;
        std::vector<long long> &heuristic_selector_survivor_total_by_depth;
        std::vector<long long> &heuristic_selector_leaf_optimal_nodes_by_depth;
        std::vector<long long> &heuristic_selector_improving_split_nodes_by_depth;
        std::vector<long long> &heuristic_selector_improving_split_retained_nodes_by_depth;
        std::vector<double> &heuristic_selector_improving_split_margin_sum_by_depth;
        std::vector<double> &heuristic_selector_improving_split_margin_max_by_depth;
        std::vector<double> &per_node_total_weight;
        std::vector<double> &per_node_mu_node;
        std::vector<std::vector<double>> &per_node_candidate_upper_bounds;
        std::vector<std::vector<double>> &per_node_candidate_lower_bounds;
        std::vector<std::vector<double>> &per_node_candidate_hard_loss;
        std::vector<std::vector<double>> &per_node_candidate_impurity_objective;
        std::vector<std::vector<double>> &per_node_candidate_hard_impurity;
        std::vector<std::vector<double>> &per_node_candidate_soft_impurity;
        std::vector<std::vector<double>> &per_node_candidate_boundary_penalty;
        std::vector<std::vector<long long>> &per_node_candidate_components;
        long long &nominee_certificate_nodes;
        long long &nominee_certificate_exhausted_nodes;
        long long &nominee_exactified_until_certificate_total;
        long long &nominee_exactified_until_certificate_max;
        long long &nominee_exactify_prefix_total;
        long long &nominee_exactify_prefix_max;
        double &nominee_certificate_min_remaining_lower_bound_sum;
        double &nominee_certificate_min_remaining_lower_bound_max;
        double &nominee_certificate_incumbent_exact_score_sum;
        double &nominee_certificate_incumbent_exact_score_max;
        std::vector<long long> &nominee_exactified_until_certificate_histogram;
        std::vector<long long> &nominee_certificate_stop_depth_histogram;
        std::vector<long long> &nominee_exactify_prefix_histogram;
    };

    Solver(
        const std::vector<int> &x_flat,
        int n_rows,
        int n_features,
        const std::vector<int> &y,
        const std::vector<double> &sample_weight,
        const std::vector<double> &teacher_logit,
        int teacher_class_count,
        const std::vector<double> &teacher_boundary_gain,
        const std::vector<double> &teacher_boundary_cover,
        const std::vector<double> &teacher_boundary_value_jump,
        int teacher_boundary_cols,
        int full_depth_budget,
        int lookahead_depth,
        double regularization,
        int min_split_size,
        int min_child_size,
        double time_limit_seconds,
        int max_branching,
        int exactify_top_k
    )
        : x_flat_(x_flat),
          n_rows_(n_rows),
          n_features_(n_features),
          y_(y),
          sample_weight_raw_(sample_weight),
          teacher_logit_raw_(teacher_logit),
          teacher_class_count_(teacher_class_count),
          teacher_boundary_gain_raw_(teacher_boundary_gain),
          teacher_boundary_cover_raw_(teacher_boundary_cover),
          teacher_boundary_value_jump_raw_(teacher_boundary_value_jump),
          teacher_boundary_cols_(teacher_boundary_cols),
          full_depth_budget_(full_depth_budget),
          lookahead_depth_(lookahead_depth),
          regularization_(regularization),
          min_split_size_(min_split_size),
          min_child_size_(min_child_size),
          time_limit_seconds_(time_limit_seconds),
          max_branching_(max_branching),
          exactify_top_k_(exactify_top_k),
          atomized_telemetry_(
              debr_refine_calls_,
              debr_refine_improved_,
              debr_total_moves_,
              debr_bridge_policy_calls_,
              debr_refine_windowed_calls_,
              debr_refine_unwindowed_calls_,
              debr_refine_overlap_segments_,
              debr_refine_calls_with_overlap_,
              debr_refine_calls_without_overlap_,
              debr_candidate_total_,
              debr_candidate_legal_,
              debr_candidate_source_size_rejects_,
              debr_candidate_target_size_rejects_,
              debr_candidate_descent_eligible_,
              debr_candidate_descent_rejected_,
              debr_candidate_bridge_eligible_,
              debr_candidate_bridge_window_blocked_,
              debr_candidate_bridge_used_blocked_,
              debr_candidate_bridge_guide_rejected_,
              debr_candidate_cleanup_eligible_,
              debr_candidate_cleanup_primary_rejected_,
              debr_candidate_cleanup_complexity_rejected_,
              debr_candidate_score_rejected_,
              debr_descent_moves_,
              debr_bridge_moves_,
              debr_simplify_moves_,
              debr_source_group_row_size_histogram_,
              debr_source_component_atom_size_histogram_,
              debr_source_component_row_size_histogram_,
              debr_total_hard_gain_,
              debr_total_soft_gain_,
              debr_total_delta_j_,
              debr_total_component_delta_,
              greedy_interval_evals_,
              atomized_features_prepared_,
              atomized_coarse_candidates_,
              atomized_final_candidates_,
              atomized_coarse_pruned_candidates_,
              atomized_compression_features_applied_,
              atomized_compression_features_collapsed_to_single_block_,
              atomized_compression_atoms_before_total_,
              atomized_compression_blocks_after_total_,
              atomized_compression_atoms_merged_total_,
              greedy_feature_survivor_histogram_,
              atomized_feature_atom_count_histogram_,
              atomized_feature_block_atom_count_histogram_,
              atomized_feature_q_effective_histogram_,
              debr_final_geo_wins_,
              debr_final_block_wins_,
              family_compare_total_,
              family_compare_equivalent_,
              family1_both_wins_,
              family2_hard_loss_wins_,
              family2_hard_impurity_wins_,
              family2_joint_impurity_wins_,
              family2_both_wins_,
              family_metric_disagreement_,
              family_hard_loss_ties_,
              family_hard_impurity_ties_,
              family_joint_impurity_ties_,
              family_neither_both_wins_,
              family1_selected_by_equivalence_,
              family1_selected_by_dominance_,
              family2_selected_by_dominance_,
              family_sent_both_,
              family1_hard_loss_sum_,
              family2_hard_loss_sum_,
              family_hard_loss_delta_sum_,
              family1_hard_impurity_sum_,
              family2_hard_impurity_sum_,
              family_hard_impurity_delta_sum_,
              family1_joint_impurity_sum_,
              family2_joint_impurity_sum_,
              family_joint_impurity_delta_sum_,
              family1_soft_impurity_sum_,
              family2_soft_impurity_sum_,
              family_soft_impurity_delta_sum_,
              family1_hard_loss_inversion_traces_,
              heuristic_selector_nodes_,
              heuristic_selector_candidate_total_,
              heuristic_selector_candidate_pruned_total_,
              heuristic_selector_survivor_total_,
              heuristic_selector_leaf_optimal_nodes_,
              heuristic_selector_improving_split_nodes_,
              heuristic_selector_improving_split_retained_nodes_,
              heuristic_selector_improving_split_margin_sum_,
              heuristic_selector_improving_split_margin_max_,
              heuristic_selector_nodes_by_depth_,
              heuristic_selector_candidate_total_by_depth_,
              heuristic_selector_candidate_pruned_total_by_depth_,
              heuristic_selector_survivor_total_by_depth_,
              heuristic_selector_leaf_optimal_nodes_by_depth_,
              heuristic_selector_improving_split_nodes_by_depth_,
              heuristic_selector_improving_split_retained_nodes_by_depth_,
              heuristic_selector_improving_split_margin_sum_by_depth_,
              heuristic_selector_improving_split_margin_max_by_depth_,
              per_node_total_weight_,
              per_node_mu_node_,
              per_node_candidate_upper_bounds_,
              per_node_candidate_lower_bounds_,
              per_node_candidate_hard_loss_,
              per_node_candidate_impurity_objective_,
              per_node_candidate_hard_impurity_,
              per_node_candidate_soft_impurity_,
              per_node_candidate_boundary_penalty_,
              per_node_candidate_components_,
              nominee_certificate_nodes_,
              nominee_certificate_exhausted_nodes_,
              nominee_exactified_until_certificate_total_,
              nominee_exactified_until_certificate_max_,
              nominee_exactify_prefix_total_,
              nominee_exactify_prefix_max_,
              nominee_certificate_min_remaining_lower_bound_sum_,
              nominee_certificate_min_remaining_lower_bound_max_,
              nominee_certificate_incumbent_exact_score_sum_,
              nominee_certificate_incumbent_exact_score_max_,
              nominee_exactified_until_certificate_histogram_,
              nominee_certificate_stop_depth_histogram_,
              nominee_exactify_prefix_histogram_),
          start_time_(Clock::now()) {
        if (n_rows_ <= 0 || n_features_ <= 0) {
            throw std::invalid_argument("MSPLIT requires a non-empty binned matrix.");
        }
        if ((int)y_.size() != n_rows_) {
            throw std::invalid_argument("MSPLIT y length must match number of rows in X.");
        }
        if ((int)x_flat_.size() != n_rows_ * n_features_) {
            throw std::invalid_argument("MSPLIT x_flat size must match n_rows * n_features.");
        }
        if (full_depth_budget_ < 1) {
            throw std::invalid_argument("MSPLIT full_depth_budget must be at least 1.");
        }
        if (lookahead_depth_ < 0) {
            throw std::invalid_argument("MSPLIT lookahead_depth must be non-negative.");
        }
        // Exact-prefix memoization is cheap enough to keep on by default.
        greedy_cache_max_depth_ = full_depth_budget_;
        // When lookahead is unset, follow the SPLIT-style default of roughly half the tree depth.
        const int requested_lookahead_depth =
            (lookahead_depth_ > 0)
                ? lookahead_depth_
                : std::max(1, (full_depth_budget_ + 1) / 2);
        effective_lookahead_depth_ =
            std::max(1, std::min(full_depth_budget_, requested_lookahead_depth));
        if (regularization_ < 0.0) {
            throw std::invalid_argument("MSPLIT regularization must be non-negative.");
        }
        if (exactify_top_k_ < 0) {
            throw std::invalid_argument("MSPLIT exactify_top_k must be non-negative.");
        }
        if (min_child_size_ < 1) {
            throw std::invalid_argument("MSPLIT min_child_size must be at least 1.");
        }
        if (min_split_size_ <= 0) {
            min_split_size_ = std::max(1, 2 * min_child_size_);
        }
        if (min_split_size_ < 1) {
            throw std::invalid_argument("MSPLIT min_split_size must be at least 1.");
        }
        if (max_branching_ < 0) {
            throw std::invalid_argument("MSPLIT max_branching must be >= 0.");
        }
        if (!sample_weight_raw_.empty() && (int)sample_weight_raw_.size() != n_rows_) {
            throw std::invalid_argument("MSPLIT sample_weight must have length n_rows when provided.");
        }
        initialize_trace();
        initialize_class_info();
        initialize_weights();
        initialize_teacher_prob();
        initialize_feature_bin_max();
        initialize_teacher_boundary_strength();
        profiling_greedy_complete_calls_by_depth_.assign(
            static_cast<size_t>(full_depth_budget_) + 1U,
            0);
        heuristic_selector_nodes_by_depth_.assign(
            static_cast<size_t>(full_depth_budget_) + 1U,
            0);
        heuristic_selector_candidate_total_by_depth_.assign(
            static_cast<size_t>(full_depth_budget_) + 1U,
            0);
        heuristic_selector_candidate_pruned_total_by_depth_.assign(
            static_cast<size_t>(full_depth_budget_) + 1U,
            0);
        heuristic_selector_survivor_total_by_depth_.assign(
            static_cast<size_t>(full_depth_budget_) + 1U,
            0);
        heuristic_selector_leaf_optimal_nodes_by_depth_.assign(
            static_cast<size_t>(full_depth_budget_) + 1U,
            0);
        heuristic_selector_improving_split_nodes_by_depth_.assign(
            static_cast<size_t>(full_depth_budget_) + 1U,
            0);
        heuristic_selector_improving_split_retained_nodes_by_depth_.assign(
            static_cast<size_t>(full_depth_budget_) + 1U,
            0);
        heuristic_selector_improving_split_margin_sum_by_depth_.assign(
            static_cast<size_t>(full_depth_budget_) + 1U,
            0.0);
        heuristic_selector_improving_split_margin_max_by_depth_.assign(
            static_cast<size_t>(full_depth_budget_) + 1U,
            0.0);
    }

    FitResult fit() {
        std::vector<int> root_indices((size_t)n_rows_);
        std::iota(root_indices.begin(), root_indices.end(), 0);

        GreedyResult solved = solve_subproblem(std::move(root_indices), full_depth_budget_);
        FitResult out;
        out.tree = to_json(solved.tree);
        out.objective = solved.objective;
        out.greedy_internal_nodes = greedy_internal_nodes_;
        out.greedy_subproblem_calls = greedy_subproblem_calls_;
        out.exact_dp_subproblem_calls_above_lookahead = exact_dp_subproblem_calls_above_lookahead_;
        out.greedy_cache_hits = greedy_cache_hits_;
        out.greedy_unique_states = greedy_cache_states_;
        out.greedy_cache_entries_peak = greedy_cache_entries_peak_;
        out.greedy_cache_bytes_peak = static_cast<long long>(greedy_cache_bytes_peak_);
        out.greedy_interval_evals = greedy_interval_evals_;
        out.elapsed_time_sec = std::chrono::duration<double>(Clock::now() - start_time_).count();
        out.profiling_lp_solve_calls = profiling_lp_solve_calls_;
        out.profiling_lp_solve_sec = profiling_lp_solve_sec_;
        out.profiling_pricing_calls = profiling_pricing_calls_;
        out.profiling_pricing_sec = profiling_pricing_sec_;
        out.profiling_greedy_complete_calls = profiling_greedy_complete_calls_;
        out.profiling_greedy_complete_sec = profiling_greedy_complete_sec_;
        out.profiling_greedy_complete_calls_by_depth = profiling_greedy_complete_calls_by_depth_;
        out.profiling_feature_prepare_sec = profiling_feature_prepare_sec_;
        out.profiling_candidate_nomination_sec = profiling_candidate_nomination_sec_;
        out.profiling_candidate_shortlist_sec = profiling_candidate_shortlist_sec_;
        out.profiling_candidate_generation_sec = profiling_candidate_generation_sec_;
        out.profiling_recursive_child_eval_sec = profiling_recursive_child_eval_sec_;
        out.heuristic_selector_nodes = heuristic_selector_nodes_;
        out.heuristic_selector_candidate_total = heuristic_selector_candidate_total_;
        out.heuristic_selector_candidate_pruned_total = heuristic_selector_candidate_pruned_total_;
        out.heuristic_selector_survivor_total = heuristic_selector_survivor_total_;
        out.heuristic_selector_leaf_optimal_nodes = heuristic_selector_leaf_optimal_nodes_;
        out.heuristic_selector_improving_split_nodes = heuristic_selector_improving_split_nodes_;
        out.heuristic_selector_improving_split_retained_nodes = heuristic_selector_improving_split_retained_nodes_;
        out.heuristic_selector_improving_split_margin_sum = heuristic_selector_improving_split_margin_sum_;
        out.heuristic_selector_improving_split_margin_max = heuristic_selector_improving_split_margin_max_;
        out.above_lookahead_impurity_pairs_total = above_lookahead_impurity_pairs_total_;
        out.above_lookahead_hardloss_pairs_total = above_lookahead_hardloss_pairs_total_;
        out.above_lookahead_impurity_bucket_before_prune_total =
            above_lookahead_impurity_bucket_before_prune_total_;
        out.above_lookahead_impurity_bucket_after_prune_total =
            above_lookahead_impurity_bucket_after_prune_total_;
        out.above_lookahead_hardloss_bucket_before_prune_total =
            above_lookahead_hardloss_bucket_before_prune_total_;
        out.above_lookahead_hardloss_bucket_after_prune_total =
            above_lookahead_hardloss_bucket_after_prune_total_;
        out.heuristic_selector_nodes_by_depth = heuristic_selector_nodes_by_depth_;
        out.heuristic_selector_candidate_total_by_depth =
            heuristic_selector_candidate_total_by_depth_;
        out.heuristic_selector_candidate_pruned_total_by_depth =
            heuristic_selector_candidate_pruned_total_by_depth_;
        out.heuristic_selector_survivor_total_by_depth = heuristic_selector_survivor_total_by_depth_;
        out.heuristic_selector_leaf_optimal_nodes_by_depth =
            heuristic_selector_leaf_optimal_nodes_by_depth_;
        out.heuristic_selector_improving_split_nodes_by_depth =
            heuristic_selector_improving_split_nodes_by_depth_;
        out.heuristic_selector_improving_split_retained_nodes_by_depth =
            heuristic_selector_improving_split_retained_nodes_by_depth_;
        out.heuristic_selector_improving_split_margin_sum_by_depth =
            heuristic_selector_improving_split_margin_sum_by_depth_;
        out.heuristic_selector_improving_split_margin_max_by_depth =
            heuristic_selector_improving_split_margin_max_by_depth_;
        out.profiling_refine_calls = profiling_refine_calls_;
        out.profiling_refine_sec = profiling_refine_sec_;
        out.debr_refine_calls = debr_refine_calls_;
        out.debr_refine_improved = debr_refine_improved_;
        out.debr_total_moves = debr_total_moves_;
        out.debr_bridge_policy_calls = debr_bridge_policy_calls_;
        out.debr_refine_windowed_calls = debr_refine_windowed_calls_;
        out.debr_refine_unwindowed_calls = debr_refine_unwindowed_calls_;
        out.debr_refine_overlap_segments = debr_refine_overlap_segments_;
        out.debr_refine_calls_with_overlap = debr_refine_calls_with_overlap_;
        out.debr_refine_calls_without_overlap = debr_refine_calls_without_overlap_;
        out.debr_candidate_total = debr_candidate_total_;
        out.debr_candidate_legal = debr_candidate_legal_;
        out.debr_candidate_source_size_rejects = debr_candidate_source_size_rejects_;
        out.debr_candidate_target_size_rejects = debr_candidate_target_size_rejects_;
        out.debr_candidate_descent_eligible = debr_candidate_descent_eligible_;
        out.debr_candidate_descent_rejected = debr_candidate_descent_rejected_;
        out.debr_candidate_bridge_eligible = debr_candidate_bridge_eligible_;
        out.debr_candidate_bridge_window_blocked = debr_candidate_bridge_window_blocked_;
        out.debr_candidate_bridge_used_blocked = debr_candidate_bridge_used_blocked_;
        out.debr_candidate_bridge_guide_rejected = debr_candidate_bridge_guide_rejected_;
        out.debr_candidate_cleanup_eligible = debr_candidate_cleanup_eligible_;
        out.debr_candidate_cleanup_primary_rejected = debr_candidate_cleanup_primary_rejected_;
        out.debr_candidate_cleanup_complexity_rejected = debr_candidate_cleanup_complexity_rejected_;
        out.debr_candidate_score_rejected = debr_candidate_score_rejected_;
        out.debr_descent_moves = debr_descent_moves_;
        out.debr_bridge_moves = debr_bridge_moves_;
        out.debr_simplify_moves = debr_simplify_moves_;
        out.debr_source_group_row_size_histogram = debr_source_group_row_size_histogram_;
        out.debr_source_component_atom_size_histogram = debr_source_component_atom_size_histogram_;
        out.debr_source_component_row_size_histogram = debr_source_component_row_size_histogram_;
        out.debr_total_hard_gain = debr_total_hard_gain_;
        out.debr_total_soft_gain = debr_total_soft_gain_;
        out.debr_total_delta_j = debr_total_delta_j_;
        out.debr_total_component_delta = debr_total_component_delta_;
        out.debr_final_geo_wins = debr_final_geo_wins_;
        out.debr_final_block_wins = debr_final_block_wins_;
        out.family_compare_total = family_compare_total_;
        out.family_compare_equivalent = family_compare_equivalent_;
        out.family1_both_wins = family1_both_wins_;
        out.family2_hard_loss_wins = family2_hard_loss_wins_;
        out.family2_hard_impurity_wins = family2_hard_impurity_wins_;
        out.family2_both_wins = family2_both_wins_;
        out.family_metric_disagreement = family_metric_disagreement_;
        out.family_hard_loss_ties = family_hard_loss_ties_;
        out.family_hard_impurity_ties = family_hard_impurity_ties_;
        out.family_joint_impurity_ties = family_joint_impurity_ties_;
        out.family_neither_both_wins = family_neither_both_wins_;
        out.family1_selected_by_equivalence = family1_selected_by_equivalence_;
        out.family1_selected_by_dominance = family1_selected_by_dominance_;
        out.family2_selected_by_dominance = family2_selected_by_dominance_;
        out.family_sent_both = family_sent_both_;
        out.family1_hard_loss_sum = family1_hard_loss_sum_;
        out.family2_hard_loss_sum = family2_hard_loss_sum_;
        out.family_hard_loss_delta_sum = family_hard_loss_delta_sum_;
        out.family1_hard_impurity_sum = family1_hard_impurity_sum_;
        out.family2_hard_impurity_sum = family2_hard_impurity_sum_;
        out.family_hard_impurity_delta_sum = family_hard_impurity_delta_sum_;
        out.family1_joint_impurity_sum = family1_joint_impurity_sum_;
        out.family2_joint_impurity_sum = family2_joint_impurity_sum_;
        out.family_joint_impurity_delta_sum = family_joint_impurity_delta_sum_;
        out.family1_soft_impurity_sum = family1_soft_impurity_sum_;
        out.family2_soft_impurity_sum = family2_soft_impurity_sum_;
        out.family_soft_impurity_delta_sum = family_soft_impurity_delta_sum_;
        out.family2_joint_impurity_wins = family2_joint_impurity_wins_;
        out.teacher_available = teacher_available_;
        out.n_classes = n_classes_;
        out.teacher_class_count = teacher_class_count_;
        out.binary_mode = binary_mode_;
        out.atomized_features_prepared = atomized_features_prepared_;
        out.atomized_coarse_candidates = atomized_coarse_candidates_;
        out.atomized_final_candidates = atomized_final_candidates_;
        out.atomized_coarse_pruned_candidates = atomized_coarse_pruned_candidates_;
        out.atomized_compression_features_applied = atomized_compression_features_applied_;
        out.atomized_compression_features_collapsed_to_single_block =
            atomized_compression_features_collapsed_to_single_block_;
        out.atomized_compression_atoms_before_total = atomized_compression_atoms_before_total_;
        out.atomized_compression_blocks_after_total = atomized_compression_blocks_after_total_;
        out.atomized_compression_atoms_merged_total = atomized_compression_atoms_merged_total_;
        out.greedy_feature_survivor_histogram = greedy_feature_survivor_histogram_;
        out.nominee_unique_total = nominee_unique_total_;
        out.nominee_child_interval_lookups = nominee_child_interval_lookups_;
        out.nominee_child_interval_unique = nominee_child_interval_unique_;
        out.nominee_exactified_total = nominee_exactified_total_;
        out.nominee_incumbent_updates = nominee_incumbent_updates_;
        out.nominee_threatening_samples = nominee_threatening_samples_;
        out.nominee_threatening_sum = nominee_threatening_sum_;
        out.nominee_threatening_max = nominee_threatening_max_;
        out.nominee_certificate_nodes = nominee_certificate_nodes_;
        out.nominee_certificate_exhausted_nodes = nominee_certificate_exhausted_nodes_;
        out.nominee_exactified_until_certificate_total = nominee_exactified_until_certificate_total_;
        out.nominee_exactified_until_certificate_max = nominee_exactified_until_certificate_max_;
        out.nominee_exactify_prefix_total = nominee_exactify_prefix_total_;
        out.nominee_exactify_prefix_max = nominee_exactify_prefix_max_;
        out.nominee_certificate_min_remaining_lower_bound_sum =
            nominee_certificate_min_remaining_lower_bound_sum_;
        out.nominee_certificate_min_remaining_lower_bound_max =
            nominee_certificate_min_remaining_lower_bound_max_;
        out.nominee_certificate_incumbent_exact_score_sum =
            nominee_certificate_incumbent_exact_score_sum_;
        out.nominee_certificate_incumbent_exact_score_max =
            nominee_certificate_incumbent_exact_score_max_;
        out.nominee_exactified_until_certificate_histogram =
            nominee_exactified_until_certificate_histogram_;
        out.nominee_certificate_stop_depth_histogram = nominee_certificate_stop_depth_histogram_;
        out.nominee_exactify_prefix_histogram = nominee_exactify_prefix_histogram_;
        out.atomized_feature_atom_count_histogram = atomized_feature_atom_count_histogram_;
        out.atomized_feature_block_atom_count_histogram = atomized_feature_block_atom_count_histogram_;
        out.atomized_feature_q_effective_histogram = atomized_feature_q_effective_histogram_;
        out.greedy_feature_survivor_histogram = greedy_feature_survivor_histogram_;
        out.per_node_prepared_features = greedy_feature_preserved_histogram_;
        out.per_node_candidate_count = greedy_candidate_count_histogram_;
        out.per_node_total_weight = per_node_total_weight_;
        out.per_node_mu_node = per_node_mu_node_;
        out.per_node_candidate_upper_bounds = per_node_candidate_upper_bounds_;
        out.per_node_candidate_lower_bounds = per_node_candidate_lower_bounds_;
        out.per_node_candidate_hard_loss = per_node_candidate_hard_loss_;
        out.per_node_candidate_impurity_objective = per_node_candidate_impurity_objective_;
        out.per_node_candidate_hard_impurity = per_node_candidate_hard_impurity_;
        out.per_node_candidate_soft_impurity = per_node_candidate_soft_impurity_;
        out.per_node_candidate_boundary_penalty = per_node_candidate_boundary_penalty_;
        out.per_node_candidate_components = per_node_candidate_components_;
        out.greedy_feature_preserved_histogram = greedy_feature_preserved_histogram_;
        out.greedy_candidate_count_histogram = greedy_candidate_count_histogram_;
        return out;
    }

    AtomizedTelemetry &atomized_telemetry() {
        return atomized_telemetry_;
    }

   private:
    const std::vector<int> &x_flat_;
    int n_rows_;
    int n_features_;
    const std::vector<int> &y_;
    const std::vector<double> &sample_weight_raw_;
    const std::vector<double> &teacher_logit_raw_;
    int teacher_class_count_ = 0;
    const std::vector<double> &teacher_boundary_gain_raw_;
    const std::vector<double> &teacher_boundary_cover_raw_;
    const std::vector<double> &teacher_boundary_value_jump_raw_;
    std::vector<double> sample_weight_;
    bool sample_weight_uniform_ = false;
    std::vector<double> teacher_prob_;
    std::vector<double> teacher_prob_flat_;
    std::vector<int> teacher_prediction_;
    bool teacher_available_ = false;
    int n_classes_ = 0;
    bool binary_mode_ = true;
    int teacher_boundary_cols_ = 0;
    std::vector<std::vector<double>> feature_boundary_prefix_;

    int full_depth_budget_;
    int lookahead_depth_;
    int effective_lookahead_depth_ = 0;
    double regularization_;
    int min_split_size_;
    int min_child_size_;
    double time_limit_seconds_;
    int max_branching_;
    int exactify_top_k_ = 0;
    bool disable_coarse_pruning_ = env_flag_enabled("MSPLIT_DISABLE_COARSE_PRUNING");

    long long greedy_internal_nodes_ = 0;
    mutable long long greedy_subproblem_calls_ = 0;
    mutable long long exact_dp_subproblem_calls_above_lookahead_ = 0;
    mutable long long greedy_cache_hits_ = 0;
    mutable long long greedy_cache_states_ = 0;
    mutable long long greedy_cache_entries_peak_ = 0;
    mutable size_t greedy_cache_entries_current_ = 0;
    mutable size_t greedy_cache_bytes_current_ = 0;
    mutable size_t greedy_cache_bytes_peak_ = 0;
    // Keep exact-prefix memoization shallow and bounded; the rollout cache is
    // stored separately as objective-only values.
    int greedy_cache_max_depth_ = -1;
    long long greedy_interval_evals_ = 0;
    mutable long long profiling_lp_solve_calls_ = 0;
    mutable double profiling_lp_solve_sec_ = 0.0;
    mutable long long profiling_pricing_calls_ = 0;
    mutable double profiling_pricing_sec_ = 0.0;
    long long profiling_greedy_complete_calls_ = 0;
    double profiling_greedy_complete_sec_ = 0.0;
    std::vector<long long> profiling_greedy_complete_calls_by_depth_;
    mutable double profiling_feature_prepare_sec_ = 0.0;
    mutable double profiling_candidate_nomination_sec_ = 0.0;
    mutable double profiling_candidate_shortlist_sec_ = 0.0;
    mutable double profiling_candidate_generation_sec_ = 0.0;
    mutable double profiling_recursive_child_eval_sec_ = 0.0;
    mutable long long heuristic_selector_nodes_ = 0;
    mutable long long heuristic_selector_candidate_total_ = 0;
    mutable long long heuristic_selector_candidate_pruned_total_ = 0;
    mutable long long heuristic_selector_survivor_total_ = 0;
    mutable long long heuristic_selector_leaf_optimal_nodes_ = 0;
    mutable long long heuristic_selector_improving_split_nodes_ = 0;
    mutable long long heuristic_selector_improving_split_retained_nodes_ = 0;
    mutable double heuristic_selector_improving_split_margin_sum_ = 0.0;
    mutable double heuristic_selector_improving_split_margin_max_ = 0.0;
    mutable long long above_lookahead_impurity_pairs_total_ = 0;
    mutable long long above_lookahead_hardloss_pairs_total_ = 0;
    mutable long long above_lookahead_impurity_bucket_before_prune_total_ = 0;
    mutable long long above_lookahead_impurity_bucket_after_prune_total_ = 0;
    mutable long long above_lookahead_hardloss_bucket_before_prune_total_ = 0;
    mutable long long above_lookahead_hardloss_bucket_after_prune_total_ = 0;
    std::vector<long long> heuristic_selector_nodes_by_depth_;
    std::vector<long long> heuristic_selector_candidate_total_by_depth_;
    std::vector<long long> heuristic_selector_candidate_pruned_total_by_depth_;
    std::vector<long long> heuristic_selector_survivor_total_by_depth_;
    std::vector<long long> heuristic_selector_leaf_optimal_nodes_by_depth_;
    std::vector<long long> heuristic_selector_improving_split_nodes_by_depth_;
    std::vector<long long> heuristic_selector_improving_split_retained_nodes_by_depth_;
    std::vector<double> heuristic_selector_improving_split_margin_sum_by_depth_;
    std::vector<double> heuristic_selector_improving_split_margin_max_by_depth_;
    mutable long long profiling_refine_calls_ = 0;
    mutable double profiling_refine_sec_ = 0.0;
    const bool profiling_enabled_ = env_flag_enabled("MSPLIT_ENABLE_PROFILING");
    const bool detailed_selector_telemetry_enabled_ =
        env_flag_enabled("MSPLIT_ENABLE_DETAILED_TELEMETRY");
    mutable std::ofstream trace_stream_;
    mutable bool trace_enabled_ = false;
    mutable std::string trace_file_path_;
    mutable long long trace_event_seq_ = 0;
    int trace_max_depth_ = 3;
        long long debr_refine_calls_ = 0;
        long long debr_refine_improved_ = 0;
        long long debr_total_moves_ = 0;
        long long debr_bridge_policy_calls_ = 0;
        long long debr_refine_windowed_calls_ = 0;
        long long debr_refine_unwindowed_calls_ = 0;
        long long debr_refine_overlap_segments_ = 0;
        long long debr_refine_calls_with_overlap_ = 0;
        long long debr_refine_calls_without_overlap_ = 0;
        long long debr_candidate_total_ = 0;
        long long debr_candidate_legal_ = 0;
        long long debr_candidate_source_size_rejects_ = 0;
        long long debr_candidate_target_size_rejects_ = 0;
        long long debr_candidate_descent_eligible_ = 0;
        long long debr_candidate_descent_rejected_ = 0;
        long long debr_candidate_bridge_eligible_ = 0;
        long long debr_candidate_bridge_window_blocked_ = 0;
        long long debr_candidate_bridge_used_blocked_ = 0;
        long long debr_candidate_bridge_guide_rejected_ = 0;
        long long debr_candidate_cleanup_eligible_ = 0;
        long long debr_candidate_cleanup_primary_rejected_ = 0;
        long long debr_candidate_cleanup_complexity_rejected_ = 0;
        long long debr_candidate_score_rejected_ = 0;
        long long debr_descent_moves_ = 0;
        long long debr_bridge_moves_ = 0;
        long long debr_simplify_moves_ = 0;
        std::vector<long long> debr_source_group_row_size_histogram_;
        std::vector<long long> debr_source_component_atom_size_histogram_;
        std::vector<long long> debr_source_component_row_size_histogram_;
    double debr_total_hard_gain_ = 0.0;
    double debr_total_soft_gain_ = 0.0;
    double debr_total_delta_j_ = 0.0;
    long long debr_total_component_delta_ = 0;
    long long debr_final_geo_wins_ = 0;
    long long debr_final_block_wins_ = 0;
    long long family_compare_total_ = 0;
    long long family_compare_equivalent_ = 0;
    long long family1_both_wins_ = 0;
    long long family2_hard_loss_wins_ = 0;
    long long family2_hard_impurity_wins_ = 0;
    long long family2_joint_impurity_wins_ = 0;
    long long family2_both_wins_ = 0;
    long long family_metric_disagreement_ = 0;
    double family1_hard_loss_sum_ = 0.0;
    double family2_hard_loss_sum_ = 0.0;
    double family_hard_loss_delta_sum_ = 0.0;
    double family1_hard_impurity_sum_ = 0.0;
    double family2_hard_impurity_sum_ = 0.0;
    double family_hard_impurity_delta_sum_ = 0.0;
    double family1_joint_impurity_sum_ = 0.0;
    double family2_joint_impurity_sum_ = 0.0;
    double family_joint_impurity_delta_sum_ = 0.0;
    double family1_soft_impurity_sum_ = 0.0;
    double family2_soft_impurity_sum_ = 0.0;
    double family_soft_impurity_delta_sum_ = 0.0;
    long long family_hard_loss_ties_ = 0;
    long long family_hard_impurity_ties_ = 0;
    long long family_joint_impurity_ties_ = 0;
    long long family_neither_both_wins_ = 0;
    long long family1_selected_by_equivalence_ = 0;
    long long family1_selected_by_dominance_ = 0;
    long long family2_selected_by_dominance_ = 0;
    long long family_sent_both_ = 0;
    std::vector<nlohmann::json> family1_hard_loss_inversion_traces_;
    AtomizedTelemetry atomized_telemetry_;
    long long atomized_features_prepared_ = 0;
    long long atomized_coarse_candidates_ = 0;
    long long atomized_final_candidates_ = 0;
    long long atomized_coarse_pruned_candidates_ = 0;
    long long atomized_compression_features_applied_ = 0;
    long long atomized_compression_features_collapsed_to_single_block_ = 0;
    long long atomized_compression_atoms_before_total_ = 0;
    long long atomized_compression_blocks_after_total_ = 0;
    long long atomized_compression_atoms_merged_total_ = 0;
    std::vector<long long> greedy_feature_survivor_histogram_;
    long long nominee_unique_total_ = 0;
    long long nominee_child_interval_lookups_ = 0;
    long long nominee_child_interval_unique_ = 0;
    long long nominee_exactified_total_ = 0;
    long long nominee_incumbent_updates_ = 0;
    long long nominee_threatening_samples_ = 0;
    double nominee_threatening_sum_ = 0.0;
    long long nominee_threatening_max_ = 0;
    long long nominee_certificate_nodes_ = 0;
    long long nominee_certificate_exhausted_nodes_ = 0;
    long long nominee_exactified_until_certificate_total_ = 0;
    long long nominee_exactified_until_certificate_max_ = 0;
    long long nominee_exactify_prefix_total_ = 0;
    long long nominee_exactify_prefix_max_ = 0;
    double nominee_certificate_min_remaining_lower_bound_sum_ = 0.0;
    double nominee_certificate_min_remaining_lower_bound_max_ = 0.0;
    double nominee_certificate_incumbent_exact_score_sum_ = 0.0;
    double nominee_certificate_incumbent_exact_score_max_ = 0.0;
    std::vector<long long> nominee_exactified_until_certificate_histogram_;
    std::vector<long long> nominee_certificate_stop_depth_histogram_;
    std::vector<long long> nominee_exactify_prefix_histogram_;
    std::vector<long long> atomized_feature_atom_count_histogram_;
    std::vector<long long> atomized_feature_block_atom_count_histogram_;
    std::vector<long long> atomized_feature_q_effective_histogram_;
    std::vector<long long> greedy_feature_preserved_histogram_;
    std::vector<long long> greedy_candidate_count_histogram_;
    std::vector<double> per_node_total_weight_;
    std::vector<double> per_node_mu_node_;
    std::vector<std::vector<double>> per_node_candidate_upper_bounds_;
    std::vector<std::vector<double>> per_node_candidate_lower_bounds_;
    std::vector<std::vector<double>> per_node_candidate_hard_loss_;
    std::vector<std::vector<double>> per_node_candidate_impurity_objective_;
    std::vector<std::vector<double>> per_node_candidate_hard_impurity_;
    std::vector<std::vector<double>> per_node_candidate_soft_impurity_;
    std::vector<std::vector<double>> per_node_candidate_boundary_penalty_;
    std::vector<std::vector<long long>> per_node_candidate_components_;

    std::vector<int> feature_bin_max_;
    mutable std::vector<int> ordered_bin_stamp_;
    mutable int ordered_bin_stamp_token_ = 0;
    mutable std::vector<std::vector<int>> ordered_bin_members_;
    mutable std::vector<int> ordered_bin_last_idx_;
    mutable std::vector<unsigned char> ordered_bin_needs_sort_;
    mutable std::vector<int> ordered_bin_touched_;

    mutable std::unordered_map<std::string, CanonicalSignatureSummary> canonical_signature_cache_;
    mutable std::unordered_map<std::string, GreedyCacheEntry> greedy_cache_;
    Clock::time_point start_time_;

    int x(int row, int feature) const { return x_flat_[(size_t)row * (size_t)n_features_ + (size_t)feature]; }

    static size_t estimate_cache_entry_bytes(const std::string &key, const GreedyResult &result) {
        return sizeof(GreedyCacheEntry) + key.capacity() + sizeof(result);
    }

    void cache_store(const std::string &key, const GreedyResult &result, int depth_remaining) const {
        if (key.empty() || depth_remaining > greedy_cache_max_depth_) {
            return;
        }
        auto [it, inserted] = greedy_cache_.emplace(
            key,
            GreedyCacheEntry{result, estimate_cache_entry_bytes(key, result)});
        if (!inserted) {
            const size_t old_bytes = it->second.bytes;
            it->second.result = result;
            it->second.bytes = estimate_cache_entry_bytes(key, result);
            if (it->second.bytes >= old_bytes) {
                greedy_cache_bytes_current_ += it->second.bytes - old_bytes;
            } else {
                greedy_cache_bytes_current_ -= old_bytes - it->second.bytes;
            }
            greedy_cache_bytes_peak_ = std::max(greedy_cache_bytes_peak_, greedy_cache_bytes_current_);
            return;
        }
        ++greedy_cache_states_;
        greedy_cache_entries_current_ += 1U;
        greedy_cache_bytes_current_ += it->second.bytes;
        greedy_cache_entries_peak_ = std::max(
            greedy_cache_entries_peak_,
            static_cast<long long>(greedy_cache_entries_current_));
        greedy_cache_bytes_peak_ = std::max(greedy_cache_bytes_peak_, greedy_cache_bytes_current_);
    }

    void record_greedy_complete_call(int depth_remaining) {
        const size_t bucket = depth_remaining < 0 ? 0U : static_cast<size_t>(depth_remaining);
        if (profiling_greedy_complete_calls_by_depth_.size() <= bucket) {
            profiling_greedy_complete_calls_by_depth_.resize(bucket + 1U, 0);
        }
        ++profiling_greedy_complete_calls_by_depth_[bucket];
    }

    void initialize_trace() {
        const char *greedy_cache_max_depth_env = std::getenv("MSPLIT_GREEDY_CACHE_MAX_DEPTH");
        if (greedy_cache_max_depth_env != nullptr && *greedy_cache_max_depth_env != '\0') {
            const int parsed = std::atoi(greedy_cache_max_depth_env);
            if (parsed >= 0) {
                greedy_cache_max_depth_ = parsed;
            }
        }
        const char *trace_path = std::getenv("MSPLIT_TRACE_FILE");
        if (trace_path == nullptr || *trace_path == '\0') {
            return;
        }
        trace_file_path_ = trace_path;
        const char *depth_limit_env = std::getenv("MSPLIT_TRACE_MAX_DEPTH");
        if (depth_limit_env != nullptr && *depth_limit_env != '\0') {
            const int parsed = std::atoi(depth_limit_env);
            if (parsed >= 0) {
                trace_max_depth_ = parsed;
            }
        }
        trace_stream_.open(trace_file_path_, std::ios::out | std::ios::app);
        trace_enabled_ = trace_stream_.is_open();
        if (trace_enabled_) {
            std::ostringstream oss;
            oss << "seq=" << trace_event_seq_++
                << " phase=trace_begin"
                << " pid=" << getpid()
                << " max_depth=" << trace_max_depth_
                << " greedy_cache_max_depth=" << greedy_cache_max_depth_
                << " rss_kb=" << current_rss_kb()
                << " cache_entries=" << greedy_cache_entries_current_
                << " cache_bytes=" << greedy_cache_bytes_current_;
            trace_line(oss.str());
        }
    }

    long long current_rss_kb() const {
        std::ifstream statm("/proc/self/statm");
        long long total_pages = 0;
        long long resident_pages = 0;
        if (!(statm >> total_pages >> resident_pages)) {
            return -1;
        }
        const long page_kb = std::max<long>(1L, sysconf(_SC_PAGESIZE) / 1024L);
        return resident_pages * page_kb;
    }

    void trace_line(const std::string &line) const {
        if (!trace_enabled_) {
            return;
        }
        trace_stream_ << line << '\n';
        trace_stream_.flush();
    }

    void trace_greedy_snapshot(
        const char *phase,
        int depth_remaining,
        size_t indices_size,
        size_t preserved_feature_count,
        size_t candidate_count,
        size_t promising_count,
        size_t processed_count,
        size_t recurse_attempts,
        double incumbent_objective,
        double best_lower_bound,
        size_t incumbent_updates
    ) const {
        if (!trace_enabled_ || depth_remaining > trace_max_depth_) {
            return;
        }
        std::ostringstream oss;
        oss << std::setprecision(17)
            << "seq=" << trace_event_seq_++
            << " phase=" << phase
            << " depth=" << depth_remaining
            << " rows=" << indices_size
            << " preserved=" << preserved_feature_count
            << " candidates=" << candidate_count
            << " promising=" << promising_count
            << " processed=" << processed_count
            << " recurse=" << recurse_attempts
            << " incumbent_updates=" << incumbent_updates
            << " incumbent=" << incumbent_objective
            << " best_lb=" << best_lower_bound
            << " cache_states=" << greedy_cache_states_
            << " cache_entries=" << greedy_cache_entries_current_
            << " cache_bytes=" << greedy_cache_bytes_current_
            << " cache_peak_entries=" << greedy_cache_entries_peak_
            << " cache_peak_bytes=" << greedy_cache_bytes_peak_
            << " rss_kb=" << current_rss_kb()
            << " greedy_calls=" << profiling_greedy_complete_calls_;
        trace_line(oss.str());
    }

    void initialize_weights() {
        sample_weight_.assign((size_t)n_rows_, 0.0);
        if (sample_weight_raw_.empty()) {
            const double uniform = 1.0 / static_cast<double>(n_rows_);
            std::fill(sample_weight_.begin(), sample_weight_.end(), uniform);
            sample_weight_uniform_ = true;
            return;
        }

        double sum = 0.0;
        for (double value : sample_weight_raw_) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument("MSPLIT sample weights must be finite and non-negative.");
            }
            sum += value;
        }
        if (!std::isfinite(sum) || sum <= 0.0) {
            throw std::invalid_argument("MSPLIT sample weights must have positive finite sum.");
        }

        const double inv_sum = 1.0 / sum;
        sample_weight_uniform_ = true;
        for (int row = 0; row < n_rows_; ++row) {
            sample_weight_[(size_t)row] = sample_weight_raw_[(size_t)row] * inv_sum;
            if (row > 0 && std::abs(sample_weight_[0] - sample_weight_[(size_t)row]) > kEpsUpdate) {
                sample_weight_uniform_ = false;
            }
        }
    }

    static double sigmoid(double value) {
        if (value >= 0.0) {
            const double z = std::exp(-value);
            return 1.0 / (1.0 + z);
        }
        const double z = std::exp(value);
        return z / (1.0 + z);
    }

    void initialize_class_info() {
        int max_label = -1;
        for (int label : y_) {
            if (label < 0) {
                throw std::invalid_argument("MSPLIT expects non-negative class ids.");
            }
            max_label = std::max(max_label, label);
        }
        n_classes_ = std::max(1, max_label + 1);
        binary_mode_ = (n_classes_ == 2);
        if (teacher_class_count_ < 0) {
            throw std::invalid_argument("MSPLIT teacher_class_count must be non-negative.");
        }
        if (teacher_logit_raw_.empty()) {
            teacher_class_count_ = 0;
            return;
        }
        if (teacher_class_count_ <= 0) {
            teacher_class_count_ = 1;
        }
        const size_t expected =
            static_cast<size_t>(n_rows_) * static_cast<size_t>(teacher_class_count_);
        if (teacher_class_count_ == 1) {
            if ((int)teacher_logit_raw_.size() != n_rows_) {
                throw std::invalid_argument(
                    "MSPLIT binary teacher_logit must have length n_rows when provided.");
            }
        } else {
            if ((int)teacher_logit_raw_.size() != (int)expected) {
                throw std::invalid_argument(
                    "MSPLIT multiclass teacher_logit must have shape (n_rows, n_classes).");
            }
            if (teacher_class_count_ != n_classes_) {
                throw std::invalid_argument(
                    "MSPLIT multiclass teacher_logit class dimension must match the number of labels.");
            }
        }
    }

    void initialize_teacher_prob() {
        teacher_prob_.assign((size_t)n_rows_, 0.5);
        teacher_prob_flat_.clear();
        teacher_prediction_.assign((size_t)n_rows_, 0);
        teacher_available_ = !teacher_logit_raw_.empty();
        if (!teacher_available_) {
            throw std::invalid_argument(
                "MSPLIT requires teacher_logit for the reference-guided atomized solver.");
        }
        if (teacher_class_count_ <= 1) {
            for (int row = 0; row < n_rows_; ++row) {
                const double logit = teacher_logit_raw_[(size_t)row];
                const double prob = std::isfinite(logit) ? sigmoid(logit) : 0.5;
                teacher_prob_[(size_t)row] = prob;
                teacher_prediction_[(size_t)row] = (prob >= 0.5) ? 1 : 0;
            }
            return;
        }
        if (binary_mode_ && teacher_class_count_ == 2) {
            teacher_prob_flat_.assign(static_cast<size_t>(n_rows_) * 2U, 0.0);
            for (int row = 0; row < n_rows_; ++row) {
                const size_t base = static_cast<size_t>(row) * 2U;
                const double logit0 = teacher_logit_raw_[base];
                const double logit1 = teacher_logit_raw_[base + 1U];
                if (!std::isfinite(logit0) && !std::isfinite(logit1)) {
                    teacher_prob_[(size_t)row] = 0.5;
                    teacher_prob_flat_[base] = 0.5;
                    teacher_prob_flat_[base + 1U] = 0.5;
                    teacher_prediction_[(size_t)row] = 1;
                    continue;
                }
                const double safe0 = std::isfinite(logit0) ? logit0 : -kInfinity;
                const double safe1 = std::isfinite(logit1) ? logit1 : -kInfinity;
                const double max_logit = std::max(safe0, safe1);
                const double exp0 = std::isfinite(safe0) ? std::exp(safe0 - max_logit) : 0.0;
                const double exp1 = std::isfinite(safe1) ? std::exp(safe1 - max_logit) : 0.0;
                const double sum_exp = exp0 + exp1;
                const double pos_prob = (sum_exp > kEpsUpdate) ? (exp1 / sum_exp) : 0.5;
                teacher_prob_[(size_t)row] = pos_prob;
                teacher_prob_flat_[base] = 1.0 - pos_prob;
                teacher_prob_flat_[base + 1U] = pos_prob;
                teacher_prediction_[(size_t)row] = (pos_prob >= 0.5) ? 1 : 0;
            }
            return;
        }
        teacher_prob_flat_.assign(static_cast<size_t>(n_rows_) * static_cast<size_t>(teacher_class_count_), 0.0);
        for (int row = 0; row < n_rows_; ++row) {
            const size_t base = static_cast<size_t>(row) * static_cast<size_t>(teacher_class_count_);
            double max_logit = -kInfinity;
            for (int cls = 0; cls < teacher_class_count_; ++cls) {
                const double logit = teacher_logit_raw_[base + static_cast<size_t>(cls)];
                if (std::isfinite(logit)) {
                    max_logit = std::max(max_logit, logit);
                }
            }
            if (!std::isfinite(max_logit)) {
                const double uniform = 1.0 / static_cast<double>(teacher_class_count_);
                for (int cls = 0; cls < teacher_class_count_; ++cls) {
                    teacher_prob_flat_[base + static_cast<size_t>(cls)] = uniform;
                }
                teacher_prediction_[(size_t)row] = 0;
                continue;
            }
            double sum_exp = 0.0;
            for (int cls = 0; cls < teacher_class_count_; ++cls) {
                const double logit = teacher_logit_raw_[base + static_cast<size_t>(cls)];
                const double weight = std::isfinite(logit) ? std::exp(logit - max_logit) : 0.0;
                teacher_prob_flat_[base + static_cast<size_t>(cls)] = weight;
                sum_exp += weight;
            }
            if (sum_exp <= kEpsUpdate) {
                const double uniform = 1.0 / static_cast<double>(teacher_class_count_);
                for (int cls = 0; cls < teacher_class_count_; ++cls) {
                    teacher_prob_flat_[base + static_cast<size_t>(cls)] = uniform;
                }
                teacher_prediction_[(size_t)row] = 0;
                continue;
            }
            const double inv_sum = 1.0 / sum_exp;
            int best_cls = 0;
            double best_prob = -1.0;
            for (int cls = 0; cls < teacher_class_count_; ++cls) {
                const double prob = teacher_prob_flat_[base + static_cast<size_t>(cls)] * inv_sum;
                teacher_prob_flat_[base + static_cast<size_t>(cls)] = prob;
                if (prob > best_prob) {
                    best_prob = prob;
                    best_cls = cls;
                }
            }
            teacher_prediction_[(size_t)row] = best_cls;
        }
    }

    void initialize_feature_bin_max() {
        feature_bin_max_.assign((size_t)n_features_, -1);
        for (int row = 0; row < n_rows_; ++row) {
            for (int feature = 0; feature < n_features_; ++feature) {
                const int value = x(row, feature);
                if (value < 0) {
                    throw std::invalid_argument("MSPLIT expects non-negative integer bins.");
                }
                feature_bin_max_[(size_t)feature] = std::max(feature_bin_max_[(size_t)feature], value);
            }
        }
    }

    void initialize_teacher_boundary_strength() {
        feature_boundary_prefix_.assign((size_t)n_features_, {});
        if (teacher_boundary_cols_ <= 0 || teacher_boundary_gain_raw_.empty() ||
            teacher_boundary_cover_raw_.empty() || teacher_boundary_value_jump_raw_.empty()) {
            return;
        }

        const size_t expected = static_cast<size_t>(n_features_) * static_cast<size_t>(teacher_boundary_cols_);
        if (teacher_boundary_gain_raw_.size() != expected ||
            teacher_boundary_cover_raw_.size() != expected ||
            teacher_boundary_value_jump_raw_.size() != expected) {
            throw std::invalid_argument("Teacher boundary prior arrays must have shape (n_features, n_boundaries).");
        }

        for (int feature = 0; feature < n_features_; ++feature) {
            const int max_bin = feature_bin_max_[(size_t)feature];
            const int n_boundaries = std::min(std::max(0, max_bin), teacher_boundary_cols_);
            std::vector<double> prefix((size_t)n_boundaries + 1U, 0.0);
            if (n_boundaries <= 0) {
                feature_boundary_prefix_[(size_t)feature] = std::move(prefix);
                continue;
            }

            double gain_max = 0.0;
            double cover_max = 0.0;
            double jump_max = 0.0;
            for (int b = 0; b < n_boundaries; ++b) {
                const size_t offset = static_cast<size_t>(feature) * static_cast<size_t>(teacher_boundary_cols_) + static_cast<size_t>(b);
                gain_max = std::max(gain_max, std::log1p(std::max(0.0, teacher_boundary_gain_raw_[offset])));
                cover_max = std::max(cover_max, std::log1p(std::max(0.0, teacher_boundary_cover_raw_[offset])));
                jump_max = std::max(jump_max, std::log1p(std::max(0.0, teacher_boundary_value_jump_raw_[offset])));
            }

            for (int b = 0; b < n_boundaries; ++b) {
                const size_t offset = static_cast<size_t>(feature) * static_cast<size_t>(teacher_boundary_cols_) + static_cast<size_t>(b);
                const double gain = std::log1p(std::max(0.0, teacher_boundary_gain_raw_[offset]));
                const double cover = std::log1p(std::max(0.0, teacher_boundary_cover_raw_[offset]));
                const double jump = std::log1p(std::max(0.0, teacher_boundary_value_jump_raw_[offset]));
                double strength = 0.0;
                strength += (gain_max > kEpsUpdate) ? (gain / gain_max) : 0.0;
                strength += (cover_max > kEpsUpdate) ? (cover / cover_max) : 0.0;
                strength += (jump_max > kEpsUpdate) ? (jump / jump_max) : 0.0;
                strength /= 3.0;
                prefix[(size_t)(b + 1)] = prefix[(size_t)b] + strength;
            }
            feature_boundary_prefix_[(size_t)feature] = std::move(prefix);
        }
    }

    double boundary_strength_between_bins(int feature, int left_bin, int right_bin_exclusive) const {
        if (feature < 0 || feature >= n_features_) {
            return 0.0;
        }
        const auto &prefix = feature_boundary_prefix_[(size_t)feature];
        if (prefix.empty()) {
            return 0.0;
        }
        const int lo = std::max(0, left_bin);
        const int hi = std::min<int>(right_bin_exclusive, static_cast<int>(prefix.size()) - 1);
        if (hi <= lo) {
            return 0.0;
        }
        return prefix[(size_t)hi] - prefix[(size_t)lo];
    }

    void check_timeout() const {
        if (time_limit_seconds_ <= 0.0) {
            return;
        }
        const double elapsed = std::chrono::duration<double>(Clock::now() - start_time_).count();
        if (elapsed > time_limit_seconds_) {
            throw std::runtime_error("MSPLIT exceeded time_limit during C++ solve.");
        }
    }

    int max_groups_for_bins(int n_bins) const {
        if (n_bins <= 0) {
            return 0;
        }
        return (max_branching_ <= 0) ? n_bins : std::min(n_bins, max_branching_);
    }

    static uint64_t hash_mix_u64(uint64_t seed, uint64_t value) {
        return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
    }

    static uint64_t state_hash(const std::vector<int> &indices, int depth_remaining) {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        h = hash_mix_u64(h, static_cast<uint64_t>(static_cast<uint32_t>(depth_remaining)));
        h = hash_mix_u64(h, static_cast<uint64_t>(static_cast<uint32_t>(indices.size())));
        for (int value : indices) {
            h = hash_mix_u64(h, static_cast<uint64_t>(static_cast<uint32_t>(value)));
        }
        return h;
    }

    struct SubproblemStats {
        int total_count = 0;
        int pos_count = 0;
        int neg_count = 0;
        double pos_weight = 0.0;
        double neg_weight = 0.0;
        std::vector<int> class_counts;
        std::vector<double> class_weight;
        double sum_weight = 0.0;
        double sum_weight_sq = 0.0;
        double reference_error_weight = 0.0;
        int prediction = 0;
        bool pure = true;
        double leaf_objective = 0.0;
    };

    SubproblemStats compute_subproblem_stats(const std::vector<int> &indices) const {
        SubproblemStats out;
        out.total_count = (int)indices.size();
        if (binary_mode_) {
            int first_label = -1;
            for (int idx : indices) {
                const int label = y_[(size_t)idx];
                const double weight = sample_weight_[(size_t)idx];
                out.sum_weight += weight;
                out.sum_weight_sq += weight * weight;
                if (teacher_prediction_[(size_t)idx] != label) {
                    out.reference_error_weight += weight;
                }
                if (label == 1) {
                    ++out.pos_count;
                    out.pos_weight += weight;
                } else {
                    ++out.neg_count;
                    out.neg_weight += weight;
                }
                if (first_label < 0) {
                    first_label = label;
                } else if (label != first_label) {
                    out.pure = false;
                }
            }
            out.prediction = (out.pos_weight >= out.neg_weight) ? 1 : 0;
            out.leaf_objective = split_leaf_loss(out.pos_weight, out.neg_weight) + regularization_;
            return out;
        }

        out.class_counts.assign((size_t)n_classes_, 0);
        out.class_weight.assign((size_t)n_classes_, 0.0);
        int first_label = -1;
        for (int idx : indices) {
            const int label = y_[(size_t)idx];
            const double weight = sample_weight_[(size_t)idx];
            out.sum_weight += weight;
            out.sum_weight_sq += weight * weight;
            if (teacher_prediction_[(size_t)idx] != label) {
                out.reference_error_weight += weight;
            }
            ++out.class_counts[(size_t)label];
            out.class_weight[(size_t)label] += weight;
            if (first_label < 0) {
                first_label = label;
            } else if (label != first_label) {
                out.pure = false;
            }
        }
        out.prediction = argmax_index(out.class_weight);
        out.leaf_objective = split_leaf_loss(out.class_weight) + regularization_;
        return out;
    }

    std::shared_ptr<Node> make_leaf_node(const SubproblemStats &stats) const {
        auto leaf = std::make_shared<Node>();
        leaf->prediction = stats.prediction;
        leaf->loss = stats.leaf_objective;
        leaf->n_samples = stats.total_count;
        if (binary_mode_) {
            leaf->neg_count = stats.neg_count;
            leaf->pos_count = stats.pos_count;
        } else {
            leaf->class_counts = stats.class_counts;
        }
        return leaf;
    }

    std::pair<double, std::shared_ptr<Node>> leaf_solution(const SubproblemStats &stats) const {
        return {stats.leaf_objective, make_leaf_node(stats)};
    }

    bool build_ordered_bins(const std::vector<int> &indices, int feature, OrderedBins &out) const {
        if (feature < 0 || feature >= n_features_) {
            return false;
        }
        const int max_bin = feature_bin_max_[(size_t)feature];
        if (max_bin < 0) {
            return false;
        }

        const size_t dense_size = (size_t)max_bin + 1U;
        if (ordered_bin_stamp_.size() < dense_size) {
            ordered_bin_stamp_.resize(dense_size, 0);
            ordered_bin_members_.resize(dense_size);
            ordered_bin_last_idx_.resize(dense_size, 0);
            ordered_bin_needs_sort_.resize(dense_size, 0);
        }

        ++ordered_bin_stamp_token_;
        if (ordered_bin_stamp_token_ == std::numeric_limits<int>::max()) {
            std::fill(ordered_bin_stamp_.begin(), ordered_bin_stamp_.end(), 0);
            ordered_bin_stamp_token_ = 1;
        }
        const int stamp = ordered_bin_stamp_token_;

        ordered_bin_touched_.clear();
        ordered_bin_touched_.reserve(std::min((size_t)indices.size(), dense_size));

        for (int idx : indices) {
            const int bin = x(idx, feature);
            if (ordered_bin_stamp_[(size_t)bin] != stamp) {
                ordered_bin_stamp_[(size_t)bin] = stamp;
                ordered_bin_members_[(size_t)bin].clear();
                ordered_bin_last_idx_[(size_t)bin] = idx;
                ordered_bin_needs_sort_[(size_t)bin] = 0;
                ordered_bin_touched_.push_back(bin);
            } else if (idx < ordered_bin_last_idx_[(size_t)bin]) {
                ordered_bin_needs_sort_[(size_t)bin] = 1;
            }
            ordered_bin_last_idx_[(size_t)bin] = idx;
            ordered_bin_members_[(size_t)bin].push_back(idx);
        }

        if (ordered_bin_touched_.size() <= 1U) {
            return false;
        }

        std::sort(ordered_bin_touched_.begin(), ordered_bin_touched_.end());
        out.values.clear();
        out.members.clear();
        out.values.reserve(ordered_bin_touched_.size());
        out.members.reserve(ordered_bin_touched_.size());
        for (int bin : ordered_bin_touched_) {
            auto &members = ordered_bin_members_[(size_t)bin];
            if (ordered_bin_needs_sort_[(size_t)bin]) {
                std::sort(members.begin(), members.end());
            }
            out.values.push_back(bin);
            out.members.push_back(std::move(members));
        }
        return true;
    }

    static void gather_group_members_sorted(
        const OrderedBins &bins,
        const std::vector<int> &group_positions,
        std::vector<int> &dst
    ) {
        size_t total = 0U;
        for (int atom_pos : group_positions) {
            total += bins.members[(size_t)atom_pos].size();
        }
        dst.clear();
        dst.reserve(total);
        for (int atom_pos : group_positions) {
            const auto &members = bins.members[(size_t)atom_pos];
            dst.insert(dst.end(), members.begin(), members.end());
        }
        std::sort(dst.begin(), dst.end());
    }

    std::string encode_signature_code(int row) const {
        std::string code;
        code.resize((size_t)n_features_ * sizeof(int));
        std::memcpy(code.data(), &x_flat_[(size_t)row * (size_t)n_features_], (size_t)n_features_ * sizeof(int));
        return code;
    }

    std::string canonical_signature_key(const std::vector<int> &indices) const {
        std::string key;
        key.reserve(sizeof(uint64_t) + indices.size() * sizeof(uint64_t));
        auto append_varint = [&](uint64_t value) {
            while (value >= 0x80U) {
                key.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
                value >>= 7U;
            }
            key.push_back(static_cast<char>(value));
        };
        append_varint((uint64_t)indices.size());
        uint64_t previous = 0U;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            const uint64_t row = static_cast<uint64_t>(indices[idx]);
            append_varint(idx == 0U ? row : (row - previous));
            previous = row;
        }
        return key;
    }

    CanonicalSignatureSummary get_canonical_signature_summary(const std::vector<int> &indices) const {
        const std::string key = canonical_signature_key(indices);
        auto cache_it = canonical_signature_cache_.find(key);
        if (cache_it != canonical_signature_cache_.end()) {
            return cache_it->second;
        }

        CanonicalSignatureSummary summary;
        summary.block_count = 0U;
        std::unordered_map<std::string, CanonicalSignatureBlock> blocks_by_code;
        blocks_by_code.reserve(indices.size());

        for (int row : indices) {
            const std::string code = encode_signature_code(row);
            CanonicalSignatureBlock &block = blocks_by_code[code];
            if (block.code_key.empty()) {
                block.code_key = code;
                if (!binary_mode_) {
                    block.class_weight.assign((size_t)n_classes_, 0.0);
                }
            }
            const double w = sample_weight_[(size_t)row];
            ++block.row_count;
            if (binary_mode_) {
                if (y_[(size_t)row] == 1) {
                    block.pos_weight += w;
                } else {
                    block.neg_weight += w;
                }
            } else {
                block.class_weight[(size_t)y_[(size_t)row]] += w;
            }
        }

        std::vector<CanonicalSignatureBlock> blocks;
        blocks.reserve(blocks_by_code.size());
        for (auto &entry : blocks_by_code) {
            blocks.push_back(std::move(entry.second));
        }
        std::sort(
            blocks.begin(),
            blocks.end(),
            [](const CanonicalSignatureBlock &lhs, const CanonicalSignatureBlock &rhs) {
                return lhs.code_key < rhs.code_key;
            });

        summary.signature_bound = regularization_;
        if (binary_mode_) {
            double total_pos = 0.0;
            double total_neg = 0.0;
            for (const auto &block : blocks) {
                summary.signature_bound += std::min(block.pos_weight, block.neg_weight);
                total_pos += block.pos_weight;
                total_neg += block.neg_weight;
            }
            summary.leaf_objective = regularization_ + std::min(total_pos, total_neg);
        } else {
            std::vector<double> total_class_weight((size_t)n_classes_, 0.0);
            for (const auto &block : blocks) {
                double total = 0.0;
                double best = 0.0;
                for (double value : block.class_weight) {
                    total += value;
                    best = std::max(best, value);
                }
                summary.signature_bound += (total > best) ? (total - best) : 0.0;
                for (int cls = 0; cls < n_classes_; ++cls) {
                    total_class_weight[(size_t)cls] += block.class_weight[(size_t)cls];
                }
            }
            double total = 0.0;
            double best = 0.0;
            for (double value : total_class_weight) {
                total += value;
                best = std::max(best, value);
            }
            summary.leaf_objective = regularization_ + ((total > best) ? (total - best) : 0.0);
        }
        summary.block_count = blocks.size();

        canonical_signature_cache_.emplace(key, summary);
        return summary;
    }

    double signature_bound_for_indices(const std::vector<int> &indices) const {
        if (indices.empty()) {
            return regularization_;
        }
        if (std::is_sorted(indices.begin(), indices.end())) {
            return get_canonical_signature_summary(indices).signature_bound;
        }
        std::vector<int> sorted_indices = indices;
        std::sort(sorted_indices.begin(), sorted_indices.end());
        return get_canonical_signature_summary(sorted_indices).signature_bound;
    }

    #include "msplit_debug.cpp"
#if defined(MSPLIT_USE_BACKUP_SELECTOR)
    #include "msplit_nonlinear.cpp"
#else
    #include "msplit_linear.cpp"
#endif
};

}  // namespace

FitResult fit(
    const std::vector<int> &x_flat,
    int n_rows,
    int n_features,
    const std::vector<int> &y,
    const std::vector<double> &sample_weight,
    const std::vector<double> &teacher_logit,
    int teacher_class_count,
    const std::vector<double> &teacher_boundary_gain,
    const std::vector<double> &teacher_boundary_cover,
    const std::vector<double> &teacher_boundary_value_jump,
    int teacher_boundary_cols,
    int full_depth_budget,
    int lookahead_depth,
    double regularization,
    int min_split_size,
    int min_child_size,
    double time_limit_seconds,
    int max_branching,
    int exactify_top_k
) {
    Solver solver(
        x_flat,
        n_rows,
        n_features,
        y,
        sample_weight,
        teacher_logit,
        teacher_class_count,
        teacher_boundary_gain,
        teacher_boundary_cover,
        teacher_boundary_value_jump,
        teacher_boundary_cols,
        full_depth_budget,
        lookahead_depth,
        regularization,
        min_split_size,
        min_child_size,
        time_limit_seconds,
        max_branching,
        exactify_top_k);
    return solver.fit();
}

static bool json_tree_has_noncontiguous_group(const nlohmann::json &node) {
    if (!node.is_object() || node.value("type", "") != "node") {
        return false;
    }
    if (node.contains("groups") && node["groups"].is_array()) {
        for (const auto &group : node["groups"]) {
            if (group.contains("spans") && group["spans"].is_array() && group["spans"].size() > 1U) {
                return true;
            }
            if (group.contains("child") && json_tree_has_noncontiguous_group(group["child"])) {
                return true;
            }
        }
    }
    return false;
}

nlohmann::json debug_run_atomized_smoke_cases() {
    auto run_case = [](
        const char *name,
        const std::vector<int> &x_flat,
        int n_rows,
        int n_features,
        const std::vector<int> &y,
        const std::vector<double> &teacher_logit,
        int depth,
        int min_split_size,
        int min_child_size,
        int max_branching
    ) {
        FitResult result = fit(
            x_flat,
            n_rows,
            n_features,
            y,
            {},
            teacher_logit,
            teacher_logit.empty() ? 0 : 1,
            {},
            {},
            {},
            0,
            depth,
            std::min(depth, 3),
            0.0,
            min_split_size,
            min_child_size,
            5.0,
            max_branching,
            0);
        return nlohmann::json{
            {"name", name},
            {"objective", result.objective},
            {"has_noncontiguous_group", json_tree_has_noncontiguous_group(result.tree)},
            {"tree", result.tree},
        };
    };

    nlohmann::json cases = nlohmann::json::array();
    cases.push_back(run_case(
        "contiguous_binary_root",
        {0, 0, 1, 1, 2, 2, 3, 3},
        8,
        1,
        {0, 0, 0, 0, 1, 1, 1, 1},
        {},
        2,
        2,
        1,
        2));
    cases.push_back(run_case(
        "noncontiguous_teacher_root",
        {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5},
        12,
        1,
        {1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0},
        {4.0, 4.0, -4.0, -4.0, -3.0, -3.0, 4.0, 4.0, 3.5, 3.5, -3.5, -3.5},
        2,
        4,
        2,
        3));
    cases.push_back(run_case(
        "support_gate_blocks_overfragmentation",
        {0, 0, 1, 1, 2, 2, 3, 3},
        8,
        1,
        {1, 0, 0, 0, 1, 1, 0, 1},
        {3.0, 2.0, -3.0, -2.5, 2.5, 3.0, -2.0, 2.0},
        2,
        6,
        3,
        4));
    return cases;
}

}  // namespace msplit
