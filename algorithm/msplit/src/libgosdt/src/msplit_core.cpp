#include "msplit.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

namespace msplit {

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kEpsUpdate = 1e-12;
std::atomic<std::uint64_t> g_solver_instance_counter{1};

static bool env_flag_enabled(const char *name) {
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }
    return !(raw[0] == '0' && raw[1] == '\0');
}

template <typename T>
static void update_atomic_max(std::atomic<T> &target, T candidate) {
    T observed = target.load(std::memory_order_relaxed);
    while (candidate > observed &&
           !target.compare_exchange_weak(
               observed,
               candidate,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
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
        int pattern_id = -1;
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

    struct SharedCanonicalSignatureCacheShard {
        std::mutex mutex;
        std::unordered_map<std::string, CanonicalSignatureSummary> entries;
    };

    struct SharedGreedyCacheShard {
        std::mutex mutex;
        std::unordered_map<std::string, GreedyCacheEntry> entries;
    };

    struct SharedCacheBundle {
        static constexpr size_t kShardCount = 64U;
        std::array<SharedCanonicalSignatureCacheShard, kShardCount> canonical_signature_shards;
        std::array<SharedGreedyCacheShard, kShardCount> greedy_shards;
        std::atomic<long long> greedy_unique_states{0};
        std::atomic<long long> greedy_entries_peak{0};
        std::atomic<long long> greedy_bytes_peak{0};
        std::atomic<size_t> greedy_entries_current{0};
        std::atomic<size_t> greedy_bytes_current{0};
    };

    struct AtomizedTelemetry {
        long long &partition_refinement_refine_calls;
        long long &partition_refinement_refine_improved;
        long long &partition_refinement_total_moves;
        long long &partition_refinement_bridge_policy_calls;
        long long &partition_refinement_refine_windowed_calls;
        long long &partition_refinement_refine_unwindowed_calls;
        long long &partition_refinement_refine_overlap_segments;
        long long &partition_refinement_refine_calls_with_overlap;
        long long &partition_refinement_refine_calls_without_overlap;
        long long &partition_refinement_candidate_total;
        long long &partition_refinement_candidate_legal;
        long long &partition_refinement_candidate_source_size_rejects;
        long long &partition_refinement_candidate_target_size_rejects;
        long long &partition_refinement_candidate_descent_eligible;
        long long &partition_refinement_candidate_descent_rejected;
        long long &partition_refinement_candidate_bridge_eligible;
        long long &partition_refinement_candidate_bridge_window_blocked;
        long long &partition_refinement_candidate_bridge_used_blocked;
        long long &partition_refinement_candidate_bridge_guide_rejected;
        long long &partition_refinement_candidate_cleanup_eligible;
        long long &partition_refinement_candidate_cleanup_primary_rejected;
        long long &partition_refinement_candidate_cleanup_complexity_rejected;
        long long &partition_refinement_candidate_score_rejected;
        long long &partition_refinement_descent_moves;
        long long &partition_refinement_bridge_moves;
        long long &partition_refinement_simplify_moves;
        std::vector<long long> &partition_refinement_source_group_row_size_histogram;
        std::vector<long long> &partition_refinement_source_component_atom_size_histogram;
        std::vector<long long> &partition_refinement_source_component_row_size_histogram;
        double &partition_refinement_total_hard_gain;
        double &partition_refinement_total_soft_gain;
        double &partition_refinement_total_delta_j;
        long long &partition_refinement_total_component_delta;
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
        long long &partition_refinement_final_geo_wins;
        long long &partition_refinement_final_block_wins;
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
        std::span<const int> x_flat,
        int n_rows,
        int n_features,
        std::span<const int> y,
        std::span<const double> sample_weight,
        std::span<const double> teacher_logit,
        int teacher_class_count,
        std::span<const double> teacher_boundary_gain,
        std::span<const double> teacher_boundary_cover,
        std::span<const double> teacher_boundary_value_jump,
        int teacher_boundary_cols,
        int full_depth_budget,
        int lookahead_depth,
        double regularization,
        int min_split_size,
        int min_child_size,
        double time_limit_seconds,
        int max_branching,
        int exactify_top_k,
        int worker_limit
    )
        : solver_instance_id_(g_solver_instance_counter.fetch_add(1U, std::memory_order_relaxed)),
          x_flat_(x_flat),
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
          worker_limit_(worker_limit),
          atomized_telemetry_(
              partition_refinement_refine_calls_,
              partition_refinement_refine_improved_,
              partition_refinement_total_moves_,
              partition_refinement_bridge_policy_calls_,
              partition_refinement_refine_windowed_calls_,
              partition_refinement_refine_unwindowed_calls_,
              partition_refinement_refine_overlap_segments_,
              partition_refinement_refine_calls_with_overlap_,
              partition_refinement_refine_calls_without_overlap_,
              partition_refinement_candidate_total_,
              partition_refinement_candidate_legal_,
              partition_refinement_candidate_source_size_rejects_,
              partition_refinement_candidate_target_size_rejects_,
              partition_refinement_candidate_descent_eligible_,
              partition_refinement_candidate_descent_rejected_,
              partition_refinement_candidate_bridge_eligible_,
              partition_refinement_candidate_bridge_window_blocked_,
              partition_refinement_candidate_bridge_used_blocked_,
              partition_refinement_candidate_bridge_guide_rejected_,
              partition_refinement_candidate_cleanup_eligible_,
              partition_refinement_candidate_cleanup_primary_rejected_,
              partition_refinement_candidate_cleanup_complexity_rejected_,
              partition_refinement_candidate_score_rejected_,
              partition_refinement_descent_moves_,
              partition_refinement_bridge_moves_,
              partition_refinement_simplify_moves_,
              partition_refinement_source_group_row_size_histogram_,
              partition_refinement_source_component_atom_size_histogram_,
              partition_refinement_source_component_row_size_histogram_,
              partition_refinement_total_hard_gain_,
              partition_refinement_total_soft_gain_,
              partition_refinement_total_delta_j_,
              partition_refinement_total_component_delta_,
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
              partition_refinement_final_geo_wins_,
              partition_refinement_final_block_wins_,
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
        if (worker_limit_ < 0) {
            throw std::invalid_argument("MSPLIT worker_limit must be >= 0.");
        }
        if (worker_limit_ == 0) {
            const unsigned int hardware = std::thread::hardware_concurrency();
            worker_limit_ = static_cast<int>(hardware == 0U ? 1U : hardware);
        }
        if (worker_limit_ > 1) {
            shared_cache_bundle_ = std::make_shared<SharedCacheBundle>();
        }
        if (!sample_weight_raw_.empty() && (int)sample_weight_raw_.size() != n_rows_) {
            throw std::invalid_argument("MSPLIT sample_weight must have length n_rows when provided.");
        }
        initialize_row_patterns();
        initialize_runtime_overrides();
        initialize_class_info();
        initialize_weights();
        initialize_teacher_prob();
        initialize_feature_bin_max();
        initialize_teacher_boundary_strength();
        if (profiling_enabled_) {
            profiling_greedy_complete_calls_by_depth_.assign(
                static_cast<size_t>(full_depth_budget_) + 1U,
                0);
        }
        if (diagnostics_enabled_) {
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
    }

    FitResult fit() {
        std::vector<int> root_indices((size_t)n_rows_);
        std::iota(root_indices.begin(), root_indices.end(), 0);

        GreedyResult solved = solve_subproblem(std::move(root_indices), full_depth_budget_);
        absorb_registered_parallel_worker_metrics();
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
        out.partition_refinement_refine_calls = partition_refinement_refine_calls_;
        out.partition_refinement_refine_improved = partition_refinement_refine_improved_;
        out.partition_refinement_total_moves = partition_refinement_total_moves_;
        out.partition_refinement_bridge_policy_calls = partition_refinement_bridge_policy_calls_;
        out.partition_refinement_refine_windowed_calls = partition_refinement_refine_windowed_calls_;
        out.partition_refinement_refine_unwindowed_calls = partition_refinement_refine_unwindowed_calls_;
        out.partition_refinement_refine_overlap_segments = partition_refinement_refine_overlap_segments_;
        out.partition_refinement_refine_calls_with_overlap = partition_refinement_refine_calls_with_overlap_;
        out.partition_refinement_refine_calls_without_overlap = partition_refinement_refine_calls_without_overlap_;
        out.partition_refinement_candidate_total = partition_refinement_candidate_total_;
        out.partition_refinement_candidate_legal = partition_refinement_candidate_legal_;
        out.partition_refinement_candidate_source_size_rejects = partition_refinement_candidate_source_size_rejects_;
        out.partition_refinement_candidate_target_size_rejects = partition_refinement_candidate_target_size_rejects_;
        out.partition_refinement_candidate_descent_eligible = partition_refinement_candidate_descent_eligible_;
        out.partition_refinement_candidate_descent_rejected = partition_refinement_candidate_descent_rejected_;
        out.partition_refinement_candidate_bridge_eligible = partition_refinement_candidate_bridge_eligible_;
        out.partition_refinement_candidate_bridge_window_blocked = partition_refinement_candidate_bridge_window_blocked_;
        out.partition_refinement_candidate_bridge_used_blocked = partition_refinement_candidate_bridge_used_blocked_;
        out.partition_refinement_candidate_bridge_guide_rejected = partition_refinement_candidate_bridge_guide_rejected_;
        out.partition_refinement_candidate_cleanup_eligible = partition_refinement_candidate_cleanup_eligible_;
        out.partition_refinement_candidate_cleanup_primary_rejected = partition_refinement_candidate_cleanup_primary_rejected_;
        out.partition_refinement_candidate_cleanup_complexity_rejected = partition_refinement_candidate_cleanup_complexity_rejected_;
        out.partition_refinement_candidate_score_rejected = partition_refinement_candidate_score_rejected_;
        out.partition_refinement_descent_moves = partition_refinement_descent_moves_;
        out.partition_refinement_bridge_moves = partition_refinement_bridge_moves_;
        out.partition_refinement_simplify_moves = partition_refinement_simplify_moves_;
        out.partition_refinement_source_group_row_size_histogram = partition_refinement_source_group_row_size_histogram_;
        out.partition_refinement_source_component_atom_size_histogram = partition_refinement_source_component_atom_size_histogram_;
        out.partition_refinement_source_component_row_size_histogram = partition_refinement_source_component_row_size_histogram_;
        out.partition_refinement_total_hard_gain = partition_refinement_total_hard_gain_;
        out.partition_refinement_total_soft_gain = partition_refinement_total_soft_gain_;
        out.partition_refinement_total_delta_j = partition_refinement_total_delta_j_;
        out.partition_refinement_total_component_delta = partition_refinement_total_component_delta_;
        out.partition_refinement_final_geo_wins = partition_refinement_final_geo_wins_;
        out.partition_refinement_final_block_wins = partition_refinement_final_block_wins_;
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
        out.teacher_available = true;
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

    bool diagnostics_enabled() const noexcept {
        return diagnostics_enabled_;
    }

   private:
    std::uint64_t solver_instance_id_ = 0;
    std::shared_ptr<SharedCacheBundle> shared_cache_bundle_;
    std::mutex parallel_worker_clones_mutex_;
    std::vector<const Solver *> parallel_worker_clones_;
    bool parallel_worker_metrics_absorbed_ = false;
    std::span<const int> x_flat_;
    int n_rows_;
    int n_features_;
    std::span<const int> y_;
    std::span<const double> sample_weight_raw_;
    std::span<const double> teacher_logit_raw_;
    int teacher_class_count_ = 0;
    std::span<const double> teacher_boundary_gain_raw_;
    std::span<const double> teacher_boundary_cover_raw_;
    std::span<const double> teacher_boundary_value_jump_raw_;
    std::vector<size_t> row_pattern_hash_;
    std::vector<int> row_pattern_id_;
    std::vector<double> sample_weight_;
    bool sample_weight_uniform_ = false;
    double uniform_sample_weight_ = 0.0;
    std::vector<double> teacher_prob_;
    std::vector<double> teacher_prob_flat_;
    std::vector<int> teacher_prediction_;
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
    int worker_limit_ = 1;
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
    const bool diagnostics_enabled_ =
        detailed_selector_telemetry_enabled_ || env_flag_enabled("MSPLIT_ENABLE_DIAGNOSTICS");
    long long partition_refinement_refine_calls_ = 0;
    long long partition_refinement_refine_improved_ = 0;
    long long partition_refinement_total_moves_ = 0;
    long long partition_refinement_bridge_policy_calls_ = 0;
    long long partition_refinement_refine_windowed_calls_ = 0;
    long long partition_refinement_refine_unwindowed_calls_ = 0;
    long long partition_refinement_refine_overlap_segments_ = 0;
    long long partition_refinement_refine_calls_with_overlap_ = 0;
    long long partition_refinement_refine_calls_without_overlap_ = 0;
    long long partition_refinement_candidate_total_ = 0;
    long long partition_refinement_candidate_legal_ = 0;
    long long partition_refinement_candidate_source_size_rejects_ = 0;
    long long partition_refinement_candidate_target_size_rejects_ = 0;
    long long partition_refinement_candidate_descent_eligible_ = 0;
    long long partition_refinement_candidate_descent_rejected_ = 0;
    long long partition_refinement_candidate_bridge_eligible_ = 0;
    long long partition_refinement_candidate_bridge_window_blocked_ = 0;
    long long partition_refinement_candidate_bridge_used_blocked_ = 0;
    long long partition_refinement_candidate_bridge_guide_rejected_ = 0;
    long long partition_refinement_candidate_cleanup_eligible_ = 0;
    long long partition_refinement_candidate_cleanup_primary_rejected_ = 0;
    long long partition_refinement_candidate_cleanup_complexity_rejected_ = 0;
    long long partition_refinement_candidate_score_rejected_ = 0;
    long long partition_refinement_descent_moves_ = 0;
    long long partition_refinement_bridge_moves_ = 0;
    long long partition_refinement_simplify_moves_ = 0;
    std::vector<long long> partition_refinement_source_group_row_size_histogram_;
    std::vector<long long> partition_refinement_source_component_atom_size_histogram_;
    std::vector<long long> partition_refinement_source_component_row_size_histogram_;
    double partition_refinement_total_hard_gain_ = 0.0;
    double partition_refinement_total_soft_gain_ = 0.0;
    double partition_refinement_total_delta_j_ = 0.0;
    long long partition_refinement_total_component_delta_ = 0;
    long long partition_refinement_final_geo_wins_ = 0;
    long long partition_refinement_final_block_wins_ = 0;
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

    mutable std::unordered_map<std::string, CanonicalSignatureSummary> canonical_signature_cache_;
    mutable std::unordered_map<std::string, GreedyCacheEntry> greedy_cache_;
    Clock::time_point start_time_;
    int x(int row, int feature) const { return x_flat_[(size_t)row * (size_t)n_features_ + (size_t)feature]; }

    const int *row_ptr(int row) const {
        return x_flat_.data() + (static_cast<size_t>(row) * static_cast<size_t>(n_features_));
    }

    void initialize_row_patterns() {
        row_pattern_hash_.assign(static_cast<size_t>(n_rows_), 0U);
        row_pattern_id_.assign(static_cast<size_t>(n_rows_), -1);
        for (int row = 0; row < n_rows_; ++row) {
            const int *row_key = row_ptr(row);
            size_t seed = 1469598103934665603ULL;
            for (int feature = 0; feature < n_features_; ++feature) {
                const size_t value_hash = std::hash<int>{}(row_key[(size_t)feature]);
                seed ^= value_hash + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
            }
            row_pattern_hash_[(size_t)row] = seed;
        }
        std::vector<int> row_order(static_cast<size_t>(n_rows_), 0);
        std::iota(row_order.begin(), row_order.end(), 0);
        std::sort(
            row_order.begin(),
            row_order.end(),
            [&](int lhs_row, int rhs_row) {
                return row_pattern_less(row_ptr(lhs_row), row_ptr(rhs_row));
            });
        int next_pattern_id = -1;
        const int *previous_pattern = nullptr;
        for (int row : row_order) {
            const int *current_pattern = row_ptr(row);
            if (previous_pattern == nullptr ||
                std::memcmp(
                    previous_pattern,
                    current_pattern,
                    static_cast<size_t>(n_features_) * sizeof(int)) != 0) {
                ++next_pattern_id;
                previous_pattern = current_pattern;
            }
            row_pattern_id_[(size_t)row] = next_pattern_id;
        }
    }

    bool row_pattern_less(const int *lhs, const int *rhs) const noexcept {
        return std::lexicographical_compare(lhs, lhs + n_features_, rhs, rhs + n_features_);
    }

    static size_t estimate_cache_entry_bytes(const std::string &key, const GreedyResult &result) {
        return sizeof(GreedyCacheEntry) + key.capacity() + sizeof(result);
    }

    size_t shared_cache_shard_index(const std::string &key) const {
        return std::hash<std::string>{}(key) % SharedCacheBundle::kShardCount;
    }

    bool greedy_cache_lookup(const std::string &key, GreedyResult &result) const {
        if (key.empty()) {
            return false;
        }
        if (shared_cache_bundle_) {
            SharedGreedyCacheShard &shard =
                shared_cache_bundle_->greedy_shards[shared_cache_shard_index(key)];
            std::scoped_lock<std::mutex> guard(shard.mutex);
            auto it = shard.entries.find(key);
            if (it == shard.entries.end()) {
                return false;
            }
            ++greedy_cache_hits_;
            result = it->second.result;
            return true;
        }
        auto it = greedy_cache_.find(key);
        if (it == greedy_cache_.end()) {
            return false;
        }
        ++greedy_cache_hits_;
        result = it->second.result;
        return true;
    }

    void cache_store(const std::string &key, const GreedyResult &result, int depth_remaining) const {
        if (key.empty() || depth_remaining > greedy_cache_max_depth_) {
            return;
        }
        if (shared_cache_bundle_) {
            const size_t bytes = estimate_cache_entry_bytes(key, result);
            SharedGreedyCacheShard &shard =
                shared_cache_bundle_->greedy_shards[shared_cache_shard_index(key)];
            std::scoped_lock<std::mutex> guard(shard.mutex);
            auto [it, inserted] = shard.entries.emplace(key, GreedyCacheEntry{result, bytes});
            if (!inserted) {
                const size_t old_bytes = it->second.bytes;
                it->second.result = result;
                it->second.bytes = bytes;
                if (bytes >= old_bytes) {
                    shared_cache_bundle_->greedy_bytes_current.fetch_add(
                        bytes - old_bytes,
                        std::memory_order_relaxed);
                } else {
                    shared_cache_bundle_->greedy_bytes_current.fetch_sub(
                        old_bytes - bytes,
                        std::memory_order_relaxed);
                }
                update_atomic_max(
                    shared_cache_bundle_->greedy_bytes_peak,
                    static_cast<long long>(
                        shared_cache_bundle_->greedy_bytes_current.load(std::memory_order_relaxed)));
                return;
            }
            const size_t entries_current = shared_cache_bundle_->greedy_entries_current.fetch_add(
                                              1U,
                                              std::memory_order_relaxed) +
                1U;
            const size_t bytes_current = shared_cache_bundle_->greedy_bytes_current.fetch_add(
                                             bytes,
                                             std::memory_order_relaxed) +
                bytes;
            shared_cache_bundle_->greedy_unique_states.fetch_add(1, std::memory_order_relaxed);
            update_atomic_max(
                shared_cache_bundle_->greedy_entries_peak,
                static_cast<long long>(entries_current));
            update_atomic_max(
                shared_cache_bundle_->greedy_bytes_peak,
                static_cast<long long>(bytes_current));
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

    void register_parallel_clone(const Solver *clone) {
        if (clone == nullptr || clone == this) {
            return;
        }
        std::scoped_lock<std::mutex> guard(parallel_worker_clones_mutex_);
        parallel_worker_clones_.push_back(clone);
    }

    static void accumulate_count_vector(
        std::vector<long long> &dst,
        const std::vector<long long> &src) {
        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); ++i) {
            dst[i] += src[i];
        }
    }

    static void accumulate_sum_vector(
        std::vector<double> &dst,
        const std::vector<double> &src) {
        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0.0);
        }
        for (size_t i = 0; i < src.size(); ++i) {
            dst[i] += src[i];
        }
    }

    static void accumulate_max_vector(
        std::vector<double> &dst,
        const std::vector<double> &src) {
        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0.0);
        }
        for (size_t i = 0; i < src.size(); ++i) {
            dst[i] = std::max(dst[i], src[i]);
        }
    }

    void absorb_worker_metrics(const Solver &worker) {
        greedy_internal_nodes_ += worker.greedy_internal_nodes_;
        greedy_subproblem_calls_ += worker.greedy_subproblem_calls_;
        exact_dp_subproblem_calls_above_lookahead_ +=
            worker.exact_dp_subproblem_calls_above_lookahead_;
        greedy_cache_hits_ += worker.greedy_cache_hits_;
        greedy_cache_states_ += worker.greedy_cache_states_;
        greedy_interval_evals_ += worker.greedy_interval_evals_;
        profiling_lp_solve_calls_ += worker.profiling_lp_solve_calls_;
        profiling_lp_solve_sec_ += worker.profiling_lp_solve_sec_;
        profiling_pricing_calls_ += worker.profiling_pricing_calls_;
        profiling_pricing_sec_ += worker.profiling_pricing_sec_;
        profiling_greedy_complete_calls_ += worker.profiling_greedy_complete_calls_;
        profiling_greedy_complete_sec_ += worker.profiling_greedy_complete_sec_;
        profiling_feature_prepare_sec_ += worker.profiling_feature_prepare_sec_;
        profiling_candidate_nomination_sec_ += worker.profiling_candidate_nomination_sec_;
        profiling_candidate_shortlist_sec_ += worker.profiling_candidate_shortlist_sec_;
        profiling_candidate_generation_sec_ += worker.profiling_candidate_generation_sec_;
        profiling_recursive_child_eval_sec_ += worker.profiling_recursive_child_eval_sec_;
        heuristic_selector_nodes_ += worker.heuristic_selector_nodes_;
        heuristic_selector_candidate_total_ += worker.heuristic_selector_candidate_total_;
        heuristic_selector_candidate_pruned_total_ += worker.heuristic_selector_candidate_pruned_total_;
        heuristic_selector_survivor_total_ += worker.heuristic_selector_survivor_total_;
        heuristic_selector_leaf_optimal_nodes_ += worker.heuristic_selector_leaf_optimal_nodes_;
        heuristic_selector_improving_split_nodes_ += worker.heuristic_selector_improving_split_nodes_;
        heuristic_selector_improving_split_retained_nodes_ += worker.heuristic_selector_improving_split_retained_nodes_;
        heuristic_selector_improving_split_margin_sum_ +=
            worker.heuristic_selector_improving_split_margin_sum_;
        heuristic_selector_improving_split_margin_max_ = std::max(
            heuristic_selector_improving_split_margin_max_,
            worker.heuristic_selector_improving_split_margin_max_);
        above_lookahead_impurity_pairs_total_ += worker.above_lookahead_impurity_pairs_total_;
        above_lookahead_hardloss_pairs_total_ += worker.above_lookahead_hardloss_pairs_total_;
        above_lookahead_impurity_bucket_before_prune_total_ +=
            worker.above_lookahead_impurity_bucket_before_prune_total_;
        above_lookahead_impurity_bucket_after_prune_total_ +=
            worker.above_lookahead_impurity_bucket_after_prune_total_;
        above_lookahead_hardloss_bucket_before_prune_total_ +=
            worker.above_lookahead_hardloss_bucket_before_prune_total_;
        above_lookahead_hardloss_bucket_after_prune_total_ +=
            worker.above_lookahead_hardloss_bucket_after_prune_total_;
        profiling_refine_calls_ += worker.profiling_refine_calls_;
        profiling_refine_sec_ += worker.profiling_refine_sec_;
        partition_refinement_refine_calls_ += worker.partition_refinement_refine_calls_;
        partition_refinement_refine_improved_ += worker.partition_refinement_refine_improved_;
        partition_refinement_total_moves_ += worker.partition_refinement_total_moves_;
        partition_refinement_bridge_policy_calls_ += worker.partition_refinement_bridge_policy_calls_;
        partition_refinement_refine_windowed_calls_ += worker.partition_refinement_refine_windowed_calls_;
        partition_refinement_refine_unwindowed_calls_ += worker.partition_refinement_refine_unwindowed_calls_;
        partition_refinement_refine_overlap_segments_ += worker.partition_refinement_refine_overlap_segments_;
        partition_refinement_refine_calls_with_overlap_ += worker.partition_refinement_refine_calls_with_overlap_;
        partition_refinement_refine_calls_without_overlap_ += worker.partition_refinement_refine_calls_without_overlap_;
        partition_refinement_candidate_total_ += worker.partition_refinement_candidate_total_;
        partition_refinement_candidate_legal_ += worker.partition_refinement_candidate_legal_;
        partition_refinement_candidate_source_size_rejects_ +=
            worker.partition_refinement_candidate_source_size_rejects_;
        partition_refinement_candidate_target_size_rejects_ +=
            worker.partition_refinement_candidate_target_size_rejects_;
        partition_refinement_candidate_descent_eligible_ +=
            worker.partition_refinement_candidate_descent_eligible_;
        partition_refinement_candidate_descent_rejected_ +=
            worker.partition_refinement_candidate_descent_rejected_;
        partition_refinement_candidate_bridge_eligible_ +=
            worker.partition_refinement_candidate_bridge_eligible_;
        partition_refinement_candidate_bridge_window_blocked_ +=
            worker.partition_refinement_candidate_bridge_window_blocked_;
        partition_refinement_candidate_bridge_used_blocked_ +=
            worker.partition_refinement_candidate_bridge_used_blocked_;
        partition_refinement_candidate_bridge_guide_rejected_ +=
            worker.partition_refinement_candidate_bridge_guide_rejected_;
        partition_refinement_candidate_cleanup_eligible_ +=
            worker.partition_refinement_candidate_cleanup_eligible_;
        partition_refinement_candidate_cleanup_primary_rejected_ +=
            worker.partition_refinement_candidate_cleanup_primary_rejected_;
        partition_refinement_candidate_cleanup_complexity_rejected_ +=
            worker.partition_refinement_candidate_cleanup_complexity_rejected_;
        partition_refinement_candidate_score_rejected_ +=
            worker.partition_refinement_candidate_score_rejected_;
        partition_refinement_descent_moves_ += worker.partition_refinement_descent_moves_;
        partition_refinement_bridge_moves_ += worker.partition_refinement_bridge_moves_;
        partition_refinement_simplify_moves_ += worker.partition_refinement_simplify_moves_;
        partition_refinement_total_hard_gain_ += worker.partition_refinement_total_hard_gain_;
        partition_refinement_total_soft_gain_ += worker.partition_refinement_total_soft_gain_;
        partition_refinement_total_delta_j_ += worker.partition_refinement_total_delta_j_;
        partition_refinement_total_component_delta_ += worker.partition_refinement_total_component_delta_;
        partition_refinement_final_geo_wins_ += worker.partition_refinement_final_geo_wins_;
        partition_refinement_final_block_wins_ += worker.partition_refinement_final_block_wins_;
        family_compare_total_ += worker.family_compare_total_;
        family_compare_equivalent_ += worker.family_compare_equivalent_;
        family1_both_wins_ += worker.family1_both_wins_;
        family2_hard_loss_wins_ += worker.family2_hard_loss_wins_;
        family2_hard_impurity_wins_ += worker.family2_hard_impurity_wins_;
        family2_joint_impurity_wins_ += worker.family2_joint_impurity_wins_;
        family2_both_wins_ += worker.family2_both_wins_;
        family_metric_disagreement_ += worker.family_metric_disagreement_;
        family_hard_loss_ties_ += worker.family_hard_loss_ties_;
        family_hard_impurity_ties_ += worker.family_hard_impurity_ties_;
        family_joint_impurity_ties_ += worker.family_joint_impurity_ties_;
        family_neither_both_wins_ += worker.family_neither_both_wins_;
        family1_selected_by_equivalence_ += worker.family1_selected_by_equivalence_;
        family1_selected_by_dominance_ += worker.family1_selected_by_dominance_;
        family2_selected_by_dominance_ += worker.family2_selected_by_dominance_;
        family_sent_both_ += worker.family_sent_both_;
        family1_hard_loss_sum_ += worker.family1_hard_loss_sum_;
        family2_hard_loss_sum_ += worker.family2_hard_loss_sum_;
        family_hard_loss_delta_sum_ += worker.family_hard_loss_delta_sum_;
        family1_hard_impurity_sum_ += worker.family1_hard_impurity_sum_;
        family2_hard_impurity_sum_ += worker.family2_hard_impurity_sum_;
        family_hard_impurity_delta_sum_ += worker.family_hard_impurity_delta_sum_;
        family1_joint_impurity_sum_ += worker.family1_joint_impurity_sum_;
        family2_joint_impurity_sum_ += worker.family2_joint_impurity_sum_;
        family_joint_impurity_delta_sum_ += worker.family_joint_impurity_delta_sum_;
        family1_soft_impurity_sum_ += worker.family1_soft_impurity_sum_;
        family2_soft_impurity_sum_ += worker.family2_soft_impurity_sum_;
        family_soft_impurity_delta_sum_ += worker.family_soft_impurity_delta_sum_;
        atomized_features_prepared_ += worker.atomized_features_prepared_;
        atomized_coarse_candidates_ += worker.atomized_coarse_candidates_;
        atomized_final_candidates_ += worker.atomized_final_candidates_;
        atomized_coarse_pruned_candidates_ += worker.atomized_coarse_pruned_candidates_;
        atomized_compression_features_applied_ += worker.atomized_compression_features_applied_;
        atomized_compression_features_collapsed_to_single_block_ +=
            worker.atomized_compression_features_collapsed_to_single_block_;
        atomized_compression_atoms_before_total_ += worker.atomized_compression_atoms_before_total_;
        atomized_compression_blocks_after_total_ += worker.atomized_compression_blocks_after_total_;
        atomized_compression_atoms_merged_total_ += worker.atomized_compression_atoms_merged_total_;
        nominee_unique_total_ += worker.nominee_unique_total_;
        nominee_child_interval_lookups_ += worker.nominee_child_interval_lookups_;
        nominee_child_interval_unique_ += worker.nominee_child_interval_unique_;
        nominee_exactified_total_ += worker.nominee_exactified_total_;
        nominee_incumbent_updates_ += worker.nominee_incumbent_updates_;
        nominee_threatening_samples_ += worker.nominee_threatening_samples_;
        nominee_threatening_sum_ += worker.nominee_threatening_sum_;
        nominee_threatening_max_ = std::max(
            nominee_threatening_max_,
            worker.nominee_threatening_max_);
        nominee_certificate_nodes_ += worker.nominee_certificate_nodes_;
        nominee_certificate_exhausted_nodes_ += worker.nominee_certificate_exhausted_nodes_;
        nominee_exactified_until_certificate_total_ +=
            worker.nominee_exactified_until_certificate_total_;
        nominee_exactified_until_certificate_max_ = std::max(
            nominee_exactified_until_certificate_max_,
            worker.nominee_exactified_until_certificate_max_);
        nominee_exactify_prefix_total_ += worker.nominee_exactify_prefix_total_;
        nominee_exactify_prefix_max_ = std::max(
            nominee_exactify_prefix_max_,
            worker.nominee_exactify_prefix_max_);
        nominee_certificate_min_remaining_lower_bound_sum_ +=
            worker.nominee_certificate_min_remaining_lower_bound_sum_;
        nominee_certificate_min_remaining_lower_bound_max_ = std::max(
            nominee_certificate_min_remaining_lower_bound_max_,
            worker.nominee_certificate_min_remaining_lower_bound_max_);
        nominee_certificate_incumbent_exact_score_sum_ +=
            worker.nominee_certificate_incumbent_exact_score_sum_;
        nominee_certificate_incumbent_exact_score_max_ = std::max(
            nominee_certificate_incumbent_exact_score_max_,
            worker.nominee_certificate_incumbent_exact_score_max_);
        accumulate_count_vector(
            profiling_greedy_complete_calls_by_depth_,
            worker.profiling_greedy_complete_calls_by_depth_);
        accumulate_count_vector(
            heuristic_selector_nodes_by_depth_,
            worker.heuristic_selector_nodes_by_depth_);
        accumulate_count_vector(
            heuristic_selector_candidate_total_by_depth_,
            worker.heuristic_selector_candidate_total_by_depth_);
        accumulate_count_vector(
            heuristic_selector_candidate_pruned_total_by_depth_,
            worker.heuristic_selector_candidate_pruned_total_by_depth_);
        accumulate_count_vector(
            heuristic_selector_survivor_total_by_depth_,
            worker.heuristic_selector_survivor_total_by_depth_);
        accumulate_count_vector(
            heuristic_selector_leaf_optimal_nodes_by_depth_,
            worker.heuristic_selector_leaf_optimal_nodes_by_depth_);
        accumulate_count_vector(
            heuristic_selector_improving_split_nodes_by_depth_,
            worker.heuristic_selector_improving_split_nodes_by_depth_);
        accumulate_count_vector(
            heuristic_selector_improving_split_retained_nodes_by_depth_,
            worker.heuristic_selector_improving_split_retained_nodes_by_depth_);
        accumulate_sum_vector(
            heuristic_selector_improving_split_margin_sum_by_depth_,
            worker.heuristic_selector_improving_split_margin_sum_by_depth_);
        accumulate_max_vector(
            heuristic_selector_improving_split_margin_max_by_depth_,
            worker.heuristic_selector_improving_split_margin_max_by_depth_);
        accumulate_count_vector(
            partition_refinement_source_group_row_size_histogram_,
            worker.partition_refinement_source_group_row_size_histogram_);
        accumulate_count_vector(
            partition_refinement_source_component_atom_size_histogram_,
            worker.partition_refinement_source_component_atom_size_histogram_);
        accumulate_count_vector(
            partition_refinement_source_component_row_size_histogram_,
            worker.partition_refinement_source_component_row_size_histogram_);
        accumulate_count_vector(
            greedy_feature_survivor_histogram_,
            worker.greedy_feature_survivor_histogram_);
        accumulate_count_vector(
            nominee_exactified_until_certificate_histogram_,
            worker.nominee_exactified_until_certificate_histogram_);
        accumulate_count_vector(
            nominee_certificate_stop_depth_histogram_,
            worker.nominee_certificate_stop_depth_histogram_);
        accumulate_count_vector(
            nominee_exactify_prefix_histogram_,
            worker.nominee_exactify_prefix_histogram_);
        accumulate_count_vector(
            atomized_feature_atom_count_histogram_,
            worker.atomized_feature_atom_count_histogram_);
        accumulate_count_vector(
            atomized_feature_block_atom_count_histogram_,
            worker.atomized_feature_block_atom_count_histogram_);
        accumulate_count_vector(
            atomized_feature_q_effective_histogram_,
            worker.atomized_feature_q_effective_histogram_);
        accumulate_count_vector(
            greedy_feature_preserved_histogram_,
            worker.greedy_feature_preserved_histogram_);
        accumulate_count_vector(
            greedy_candidate_count_histogram_,
            worker.greedy_candidate_count_histogram_);
    }

    void absorb_registered_parallel_worker_metrics() {
        if (parallel_worker_metrics_absorbed_) {
            return;
        }
        parallel_worker_metrics_absorbed_ = true;
        std::vector<const Solver *> clones;
        {
            std::scoped_lock<std::mutex> guard(parallel_worker_clones_mutex_);
            clones = parallel_worker_clones_;
        }
        for (const Solver *clone : clones) {
            if (clone != nullptr) {
                absorb_worker_metrics(*clone);
            }
        }
        if (shared_cache_bundle_) {
            greedy_cache_states_ =
                shared_cache_bundle_->greedy_unique_states.load(std::memory_order_relaxed);
            greedy_cache_entries_peak_ = std::max(
                greedy_cache_entries_peak_,
                shared_cache_bundle_->greedy_entries_peak.load(std::memory_order_relaxed));
            greedy_cache_bytes_peak_ = std::max(
                greedy_cache_bytes_peak_,
                static_cast<size_t>(
                    shared_cache_bundle_->greedy_bytes_peak.load(std::memory_order_relaxed)));
        }
    }

    void record_greedy_complete_call(int depth_remaining) {
        if (!profiling_enabled_) {
            return;
        }
        const size_t bucket = depth_remaining < 0 ? 0U : static_cast<size_t>(depth_remaining);
        if (profiling_greedy_complete_calls_by_depth_.size() <= bucket) {
            profiling_greedy_complete_calls_by_depth_.resize(bucket + 1U, 0);
        }
        ++profiling_greedy_complete_calls_by_depth_[bucket];
    }

    void initialize_runtime_overrides() {
        const char *greedy_cache_max_depth_env = std::getenv("MSPLIT_GREEDY_CACHE_MAX_DEPTH");
        if (greedy_cache_max_depth_env != nullptr && *greedy_cache_max_depth_env != '\0') {
            const int parsed = std::atoi(greedy_cache_max_depth_env);
            if (parsed >= 0) {
                greedy_cache_max_depth_ = parsed;
            }
        }
    }

    void initialize_weights() {
        sample_weight_.assign((size_t)n_rows_, 0.0);
        uniform_sample_weight_ = 0.0;
        if (sample_weight_raw_.empty()) {
            const double uniform = 1.0 / static_cast<double>(n_rows_);
            std::fill(sample_weight_.begin(), sample_weight_.end(), uniform);
            sample_weight_uniform_ = true;
            uniform_sample_weight_ = uniform;
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
        if (sample_weight_uniform_ && !sample_weight_.empty()) {
            uniform_sample_weight_ = sample_weight_[0];
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
        if (teacher_logit_raw_.empty()) {
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
            if (sample_weight_uniform_) {
                int reference_error_count = 0;
                for (int idx : indices) {
                    const int label = y_[(size_t)idx];
                    if (teacher_prediction_[(size_t)idx] != label) {
                        ++reference_error_count;
                    }
                    if (label == 1) {
                        ++out.pos_count;
                    } else {
                        ++out.neg_count;
                    }
                }
                out.pos_weight = static_cast<double>(out.pos_count) * uniform_sample_weight_;
                out.neg_weight = static_cast<double>(out.neg_count) * uniform_sample_weight_;
                out.sum_weight = static_cast<double>(out.total_count) * uniform_sample_weight_;
                out.sum_weight_sq =
                    static_cast<double>(out.total_count) * uniform_sample_weight_ * uniform_sample_weight_;
                out.reference_error_weight =
                    static_cast<double>(reference_error_count) * uniform_sample_weight_;
                out.pure = (out.pos_count == 0 || out.neg_count == 0);
                out.prediction = (out.pos_weight >= out.neg_weight) ? 1 : 0;
                out.leaf_objective = split_leaf_loss(out.pos_weight, out.neg_weight) + regularization_;
                return out;
            }
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
        struct OrderedBinScratch {
            std::vector<int> stamp;
            int stamp_token = 0;
            std::vector<std::vector<int>> members;
            std::vector<int> last_idx;
            std::vector<unsigned char> needs_sort;
            std::vector<int> touched;
        };
        static thread_local OrderedBinScratch scratch;
        if (feature < 0 || feature >= n_features_) {
            return false;
        }
        const int max_bin = feature_bin_max_[(size_t)feature];
        if (max_bin < 0) {
            return false;
        }

        const size_t dense_size = (size_t)max_bin + 1U;
        if (scratch.stamp.size() < dense_size) {
            scratch.stamp.resize(dense_size, 0);
            scratch.members.resize(dense_size);
            scratch.last_idx.resize(dense_size, 0);
            scratch.needs_sort.resize(dense_size, 0);
        }

        ++scratch.stamp_token;
        if (scratch.stamp_token == std::numeric_limits<int>::max()) {
            std::fill(scratch.stamp.begin(), scratch.stamp.end(), 0);
            scratch.stamp_token = 1;
        }
        const int stamp = scratch.stamp_token;

        scratch.touched.clear();
        scratch.touched.reserve(std::min((size_t)indices.size(), dense_size));

        for (int idx : indices) {
            const int bin = x(idx, feature);
            if (scratch.stamp[(size_t)bin] != stamp) {
                scratch.stamp[(size_t)bin] = stamp;
                scratch.members[(size_t)bin].clear();
                scratch.last_idx[(size_t)bin] = idx;
                scratch.needs_sort[(size_t)bin] = 0;
                scratch.touched.push_back(bin);
            } else if (idx < scratch.last_idx[(size_t)bin]) {
                scratch.needs_sort[(size_t)bin] = 1;
            }
            scratch.last_idx[(size_t)bin] = idx;
            scratch.members[(size_t)bin].push_back(idx);
        }

        if (scratch.touched.size() <= 1U) {
            return false;
        }

        std::sort(scratch.touched.begin(), scratch.touched.end());
        out.values.clear();
        out.members.clear();
        out.values.reserve(scratch.touched.size());
        out.members.reserve(scratch.touched.size());
        for (int bin : scratch.touched) {
            auto &members = scratch.members[(size_t)bin];
            if (scratch.needs_sort[(size_t)bin]) {
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
        if (group_positions.empty()) {
            return;
        }
        if (group_positions.size() == 1U) {
            dst = bins.members[(size_t)group_positions.front()];
            return;
        }
        dst.reserve(total);
        if (group_positions.size() == 2U) {
            const auto &lhs = bins.members[(size_t)group_positions[0]];
            const auto &rhs = bins.members[(size_t)group_positions[1]];
            dst.resize(total);
            std::merge(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), dst.begin());
            return;
        }
        struct MergeCursor {
            int value = 0;
            size_t group_idx = 0U;
            size_t member_idx = 0U;
        };
        struct MergeCursorGreater {
            bool operator()(const MergeCursor &lhs, const MergeCursor &rhs) const noexcept {
                if (lhs.value != rhs.value) {
                    return lhs.value > rhs.value;
                }
                if (lhs.group_idx != rhs.group_idx) {
                    return lhs.group_idx > rhs.group_idx;
                }
                return lhs.member_idx > rhs.member_idx;
            }
        };
        std::priority_queue<MergeCursor, std::vector<MergeCursor>, MergeCursorGreater> heap;
        for (size_t group_idx = 0; group_idx < group_positions.size(); ++group_idx) {
            const auto &members = bins.members[(size_t)group_positions[group_idx]];
            if (!members.empty()) {
                heap.push(MergeCursor{members[0], group_idx, 0U});
            }
        }
        while (!heap.empty()) {
            const MergeCursor cursor = heap.top();
            heap.pop();
            dst.push_back(cursor.value);
            const auto &members = bins.members[(size_t)group_positions[cursor.group_idx]];
            const size_t next_idx = cursor.member_idx + 1U;
            if (next_idx < members.size()) {
                heap.push(MergeCursor{members[next_idx], cursor.group_idx, next_idx});
            }
        }
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
        if (shared_cache_bundle_) {
            SharedCanonicalSignatureCacheShard &shard =
                shared_cache_bundle_->canonical_signature_shards[shared_cache_shard_index(key)];
            std::scoped_lock<std::mutex> guard(shard.mutex);
            auto cache_it = shard.entries.find(key);
            if (cache_it != shard.entries.end()) {
                return cache_it->second;
            }
        } else {
            auto cache_it = canonical_signature_cache_.find(key);
            if (cache_it != canonical_signature_cache_.end()) {
                return cache_it->second;
            }
        }

        CanonicalSignatureSummary summary;
        summary.block_count = 0U;
        std::unordered_map<int, CanonicalSignatureBlock> blocks_by_pattern;
        blocks_by_pattern.reserve(indices.size());

        for (int row : indices) {
            const int pattern_id = row_pattern_id_[(size_t)row];
            CanonicalSignatureBlock &block = blocks_by_pattern[pattern_id];
            if (block.pattern_id < 0) {
                block.pattern_id = pattern_id;
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
        blocks.reserve(blocks_by_pattern.size());
        for (auto &entry : blocks_by_pattern) {
            blocks.push_back(std::move(entry.second));
        }
        std::sort(
            blocks.begin(),
            blocks.end(),
            [](const CanonicalSignatureBlock &lhs, const CanonicalSignatureBlock &rhs) {
                return lhs.pattern_id < rhs.pattern_id;
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

        if (shared_cache_bundle_) {
            SharedCanonicalSignatureCacheShard &shard =
                shared_cache_bundle_->canonical_signature_shards[shared_cache_shard_index(key)];
            std::scoped_lock<std::mutex> guard(shard.mutex);
            shard.entries.emplace(key, summary);
        } else {
            canonical_signature_cache_.emplace(key, summary);
        }
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
    // Keep one explicit active selector path instead of hiding it behind a
    // build-time switch. The current production solver lives in msplit_nonlinear.cpp.
    #include "msplit_nonlinear.cpp"
};

}  // namespace

FitResult fit(
    std::span<const int> x_flat,
    int n_rows,
    int n_features,
    std::span<const int> y,
    std::span<const double> sample_weight,
    std::span<const double> teacher_logit,
    int teacher_class_count,
    std::span<const double> teacher_boundary_gain,
    std::span<const double> teacher_boundary_cover,
    std::span<const double> teacher_boundary_value_jump,
    int teacher_boundary_cols,
    int full_depth_budget,
    int lookahead_depth,
    double regularization,
    int min_split_size,
    int min_child_size,
    double time_limit_seconds,
    int max_branching,
    int exactify_top_k,
    int worker_limit
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
        exactify_top_k,
        worker_limit);
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
            0,
            1);
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
