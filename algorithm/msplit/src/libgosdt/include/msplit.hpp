#ifndef MSPLIT_H
#define MSPLIT_H

#include <span>
#include <vector>

#include "nlohmann/json.hpp"

namespace msplit {

struct FitResult {
    nlohmann::json tree;
    double objective = 0.0;
    int greedy_internal_nodes = 0;
    long long greedy_subproblem_calls = 0;
    long long exact_dp_subproblem_calls_above_lookahead = 0;
    long long greedy_cache_hits = 0;
    long long greedy_unique_states = 0;
    long long greedy_cache_entries_peak = 0;
    long long greedy_cache_bytes_peak = 0;
    long long greedy_interval_evals = 0;
    double elapsed_time_sec = 0.0;
    long long profiling_lp_solve_calls = 0;
    double profiling_lp_solve_sec = 0.0;
    long long profiling_pricing_calls = 0;
    double profiling_pricing_sec = 0.0;
    long long profiling_greedy_complete_calls = 0;
    double profiling_greedy_complete_sec = 0.0;
    std::vector<long long> profiling_greedy_complete_calls_by_depth;
    double profiling_feature_prepare_sec = 0.0;
    double profiling_candidate_nomination_sec = 0.0;
    double profiling_candidate_shortlist_sec = 0.0;
    double profiling_candidate_generation_sec = 0.0;
    double profiling_recursive_child_eval_sec = 0.0;
    long long heuristic_selector_nodes = 0;
    long long heuristic_selector_candidate_total = 0;
    long long heuristic_selector_candidate_pruned_total = 0;
    long long heuristic_selector_survivor_total = 0;
    long long heuristic_selector_leaf_optimal_nodes = 0;
    long long heuristic_selector_improving_split_nodes = 0;
    long long heuristic_selector_improving_split_retained_nodes = 0;
    double heuristic_selector_improving_split_margin_sum = 0.0;
    double heuristic_selector_improving_split_margin_max = 0.0;
    long long above_lookahead_impurity_pairs_total = 0;
    long long above_lookahead_hardloss_pairs_total = 0;
    long long above_lookahead_impurity_bucket_before_prune_total = 0;
    long long above_lookahead_impurity_bucket_after_prune_total = 0;
    long long above_lookahead_hardloss_bucket_before_prune_total = 0;
    long long above_lookahead_hardloss_bucket_after_prune_total = 0;
    std::vector<long long> heuristic_selector_nodes_by_depth;
    std::vector<long long> heuristic_selector_candidate_total_by_depth;
    std::vector<long long> heuristic_selector_candidate_pruned_total_by_depth;
    std::vector<long long> heuristic_selector_survivor_total_by_depth;
    std::vector<long long> heuristic_selector_leaf_optimal_nodes_by_depth;
    std::vector<long long> heuristic_selector_improving_split_nodes_by_depth;
    std::vector<long long> heuristic_selector_improving_split_retained_nodes_by_depth;
    std::vector<double> heuristic_selector_improving_split_margin_sum_by_depth;
    std::vector<double> heuristic_selector_improving_split_margin_max_by_depth;
    long long profiling_refine_calls = 0;
    double profiling_refine_sec = 0.0;
    long long partition_refinement_refine_calls = 0;
    long long partition_refinement_refine_improved = 0;
    long long partition_refinement_total_moves = 0;
    long long partition_refinement_bridge_policy_calls = 0;
    long long partition_refinement_refine_windowed_calls = 0;
    long long partition_refinement_refine_unwindowed_calls = 0;
    long long partition_refinement_refine_overlap_segments = 0;
    long long partition_refinement_refine_calls_with_overlap = 0;
    long long partition_refinement_refine_calls_without_overlap = 0;
    long long partition_refinement_candidate_total = 0;
    long long partition_refinement_candidate_legal = 0;
    long long partition_refinement_candidate_source_size_rejects = 0;
    long long partition_refinement_candidate_target_size_rejects = 0;
    long long partition_refinement_candidate_descent_eligible = 0;
    long long partition_refinement_candidate_descent_rejected = 0;
    long long partition_refinement_candidate_bridge_eligible = 0;
    long long partition_refinement_candidate_bridge_window_blocked = 0;
    long long partition_refinement_candidate_bridge_used_blocked = 0;
    long long partition_refinement_candidate_bridge_guide_rejected = 0;
    long long partition_refinement_candidate_cleanup_eligible = 0;
    long long partition_refinement_candidate_cleanup_primary_rejected = 0;
    long long partition_refinement_candidate_cleanup_complexity_rejected = 0;
    long long partition_refinement_candidate_score_rejected = 0;
    long long partition_refinement_descent_moves = 0;
    long long partition_refinement_bridge_moves = 0;
    long long partition_refinement_simplify_moves = 0;
    std::vector<long long> partition_refinement_source_group_row_size_histogram;
    std::vector<long long> partition_refinement_source_component_atom_size_histogram;
    std::vector<long long> partition_refinement_source_component_row_size_histogram;
    double partition_refinement_total_hard_gain = 0.0;
    double partition_refinement_total_soft_gain = 0.0;
    double partition_refinement_total_delta_j = 0.0;
    long long partition_refinement_total_component_delta = 0;
    long long partition_refinement_final_geo_wins = 0;
    long long partition_refinement_final_block_wins = 0;
    long long family_compare_total = 0;
    long long family_compare_equivalent = 0;
    long long family1_both_wins = 0;
    long long family2_hard_loss_wins = 0;
    long long family2_hard_impurity_wins = 0;
    long long family2_both_wins = 0;
    long long family_metric_disagreement = 0;
    long long family_hard_loss_ties = 0;
    long long family_hard_impurity_ties = 0;
    long long family_joint_impurity_ties = 0;
    long long family_neither_both_wins = 0;
    long long family1_selected_by_equivalence = 0;
    long long family1_selected_by_dominance = 0;
    long long family2_selected_by_dominance = 0;
    long long family_sent_both = 0;
    double family1_hard_loss_sum = 0.0;
    double family2_hard_loss_sum = 0.0;
    double family_hard_loss_delta_sum = 0.0;
    double family1_hard_impurity_sum = 0.0;
    double family2_hard_impurity_sum = 0.0;
    double family_hard_impurity_delta_sum = 0.0;
    double family1_joint_impurity_sum = 0.0;
    double family2_joint_impurity_sum = 0.0;
    double family_joint_impurity_delta_sum = 0.0;
    double family1_soft_impurity_sum = 0.0;
    double family2_soft_impurity_sum = 0.0;
    double family_soft_impurity_delta_sum = 0.0;
    long long family2_joint_impurity_wins = 0;
    bool teacher_available = false;
    int n_classes = 0;
    int teacher_class_count = 0;
    bool binary_mode = true;
    long long atomized_features_prepared = 0;
    long long atomized_coarse_candidates = 0;
    long long atomized_final_candidates = 0;
    long long atomized_coarse_pruned_candidates = 0;
    long long atomized_compression_features_applied = 0;
    long long atomized_compression_features_collapsed_to_single_block = 0;
    long long atomized_compression_atoms_before_total = 0;
    long long atomized_compression_blocks_after_total = 0;
    long long atomized_compression_atoms_merged_total = 0;
    std::vector<long long> greedy_feature_survivor_histogram;
    long long nominee_unique_total = 0;
    long long nominee_child_interval_lookups = 0;
    long long nominee_child_interval_unique = 0;
    long long nominee_exactified_total = 0;
    long long nominee_incumbent_updates = 0;
    long long nominee_threatening_samples = 0;
    double nominee_threatening_sum = 0.0;
    long long nominee_threatening_max = 0;
    long long nominee_certificate_nodes = 0;
    long long nominee_certificate_exhausted_nodes = 0;
    long long nominee_exactified_until_certificate_total = 0;
    long long nominee_exactified_until_certificate_max = 0;
    long long nominee_exactify_prefix_total = 0;
    long long nominee_exactify_prefix_max = 0;
    double nominee_certificate_min_remaining_lower_bound_sum = 0.0;
    double nominee_certificate_min_remaining_lower_bound_max = 0.0;
    double nominee_certificate_incumbent_exact_score_sum = 0.0;
    double nominee_certificate_incumbent_exact_score_max = 0.0;
    std::vector<long long> nominee_exactified_until_certificate_histogram;
    std::vector<long long> nominee_certificate_stop_depth_histogram;
    std::vector<long long> nominee_exactify_prefix_histogram;
    std::vector<long long> atomized_feature_atom_count_histogram;
    std::vector<long long> atomized_feature_block_atom_count_histogram;
    std::vector<long long> atomized_feature_q_effective_histogram;
    std::vector<long long> greedy_feature_preserved_histogram;
    std::vector<long long> greedy_candidate_count_histogram;
    std::vector<double> per_node_total_weight;
    std::vector<double> per_node_mu_node;
    std::vector<std::vector<double>> per_node_candidate_upper_bounds;
    std::vector<std::vector<double>> per_node_candidate_lower_bounds;
    std::vector<std::vector<double>> per_node_candidate_hard_loss;
    std::vector<std::vector<double>> per_node_candidate_impurity_objective;
    std::vector<std::vector<double>> per_node_candidate_hard_impurity;
    std::vector<std::vector<double>> per_node_candidate_soft_impurity;
    std::vector<std::vector<double>> per_node_candidate_boundary_penalty;
    std::vector<std::vector<long long>> per_node_candidate_components;
};

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
);

nlohmann::json debug_run_atomized_smoke_cases();

}  // namespace msplit

#endif
