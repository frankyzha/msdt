    #include "msplit_atomized_support.cpp"

    AtomizedCandidate refine_atomized_candidate_partition_locally(
        const std::vector<AtomizedBin> &atoms,
        const AtomizedCandidate &seed,
        double mu_node,
        const std::vector<std::pair<int, int>> *active_windows = nullptr,
        AtomizedRefinementSummary *summary = nullptr,
        const std::vector<double> *adjacency_bonus = nullptr,
        double adjacency_bonus_total = 0.0,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        if (!seed.feasible) {
            return seed;
        }
        ++profiling_refine_calls_;
        ScopedTimer refine_timer(profiling_refine_sec_, profiling_enabled_);

        AtomizedRefinementSummary local_summary;
        const bool collect_summary = summary != nullptr;
        int move_count = 0;

        const int bin_count = (int)atoms.size();
        const int groups = seed.groups;
        if (bin_count <= 1 || groups <= 1) {
            return seed;
        }
        const bool hard_loss_mode = (mode == AtomizedObjectiveMode::kHardLoss);
        if (collect_summary) {
            ++local_summary.bridge_policy_calls;
        }
        auto bump_histogram = [&](std::vector<long long> &hist, int bucket) {
            if (!collect_summary || bucket < 0) {
                return;
            }
            const size_t idx = static_cast<size_t>(bucket);
            if (hist.size() <= idx) {
                hist.resize(idx + 1U, 0);
            }
            ++hist[idx];
        };

        struct RankedMove {
            AtomizedRefinementMove move;
            double primary_gain = 0.0;
            double guide_gain = 0.0;
            double complexity_gain = 0.0;
            double bridge_gain = 0.0;
            bool valid = false;
        };

        auto ranked_move_better_metadata = [](const RankedMove &lhs, const RankedMove &rhs) {
            if (lhs.move.length != rhs.move.length) {
                return lhs.move.length < rhs.move.length;
            }
            if (lhs.move.source_group != rhs.move.source_group) {
                return lhs.move.source_group < rhs.move.source_group;
            }
            if (lhs.move.target_group != rhs.move.target_group) {
                return lhs.move.target_group < rhs.move.target_group;
            }
            if (lhs.move.start != rhs.move.start) {
                return lhs.move.start < rhs.move.start;
            }
            return lhs.move.end < rhs.move.end;
        };

        auto ranked_move_better_descent = [&](const RankedMove &lhs, const RankedMove &rhs) {
            if (!rhs.valid) {
                return lhs.valid;
            }
            if (!lhs.valid) {
                return false;
            }
            if (lhs.primary_gain > rhs.primary_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.primary_gain > lhs.primary_gain + kEpsUpdate) {
                return false;
            }
            if (lhs.guide_gain > rhs.guide_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.guide_gain > lhs.guide_gain + kEpsUpdate) {
                return false;
            }
            if (lhs.complexity_gain > rhs.complexity_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.complexity_gain > lhs.complexity_gain + kEpsUpdate) {
                return false;
            }
            return ranked_move_better_metadata(lhs, rhs);
        };

        auto ranked_move_better_bridge = [&](const RankedMove &lhs, const RankedMove &rhs) {
            if (!rhs.valid) {
                return lhs.valid;
            }
            if (!lhs.valid) {
                return false;
            }
            if (lhs.bridge_gain > rhs.bridge_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.bridge_gain > lhs.bridge_gain + kEpsUpdate) {
                return false;
            }
            if (lhs.guide_gain > rhs.guide_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.guide_gain > lhs.guide_gain + kEpsUpdate) {
                return false;
            }
            if (lhs.complexity_gain > rhs.complexity_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.complexity_gain > lhs.complexity_gain + kEpsUpdate) {
                return false;
            }
            return ranked_move_better_metadata(lhs, rhs);
        };

        auto ranked_move_better_cleanup = [&](const RankedMove &lhs, const RankedMove &rhs) {
            if (!rhs.valid) {
                return lhs.valid;
            }
            if (!lhs.valid) {
                return false;
            }
            if (lhs.complexity_gain > rhs.complexity_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.complexity_gain > lhs.complexity_gain + kEpsUpdate) {
                return false;
            }
            if (lhs.guide_gain > rhs.guide_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.guide_gain > lhs.guide_gain + kEpsUpdate) {
                return false;
            }
            return ranked_move_better_metadata(lhs, rhs);
        };

        const bool bridge_allowed = (active_windows == nullptr || active_windows->empty());
        bool bridge_used = false;
        const bool has_active_windows = active_windows != nullptr && !active_windows->empty();
        if (collect_summary) {
            if (has_active_windows) {
                ++local_summary.refine_windowed_calls;
            } else {
                ++local_summary.refine_unwindowed_calls;
            }
        }
        bool saw_overlap_segment = false;

        auto make_ranked_move = [&](AtomizedRefinementMove &&move,
                                    double primary_gain,
                                    double guide_gain) {
            RankedMove ranked;
            ranked.move = std::move(move);
            ranked.primary_gain = primary_gain;
            ranked.guide_gain = guide_gain;
            ranked.complexity_gain = -static_cast<double>(ranked.move.delta_components);
            ranked.bridge_gain = ranked.guide_gain + mu_node * ranked.complexity_gain;
            ranked.valid = true;
            return ranked;
        };

        auto consider_ranked_move = [&](AtomizedRefinementMove &&move,
                                        double primary_gain,
                                        double guide_gain,
                                        RankedMove &best_descent,
                                        RankedMove &best_bridge) {
            if (collect_summary) {
                ++local_summary.candidate_legal;
            }
            const bool is_descent = primary_gain > kEpsUpdate;
            RankedMove ranked = make_ranked_move(
                std::move(move),
                primary_gain,
                guide_gain);
            if (is_descent) {
                if (collect_summary) {
                    ++local_summary.candidate_descent_eligible;
                }
                if (!best_descent.valid || ranked_move_better_descent(ranked, best_descent)) {
                    best_descent = std::move(ranked);
                }
                return;
            }

            if (collect_summary) {
                ++local_summary.candidate_descent_rejected;
            }
            const bool primary_near_zero = std::abs(primary_gain) <= kEpsUpdate;
            if (primary_near_zero) {
                if (active_windows != nullptr && !active_windows->empty()) {
                    if (collect_summary) {
                        ++local_summary.candidate_bridge_window_blocked;
                    }
                } else if (bridge_used) {
                    if (collect_summary) {
                        ++local_summary.candidate_bridge_used_blocked;
                    }
                } else if (ranked.guide_gain > kEpsUpdate) {
                    if (collect_summary) {
                        ++local_summary.candidate_bridge_eligible;
                    }
                    if (!best_bridge.valid || ranked_move_better_bridge(ranked, best_bridge)) {
                        best_bridge = std::move(ranked);
                    }
                    return;
                } else {
                    if (collect_summary) {
                        ++local_summary.candidate_bridge_guide_rejected;
                    }
                }
            }
            if (collect_summary) {
                ++local_summary.candidate_score_rejected;
            }
        };

        auto atomized_move_is_improving = [&](const AtomizedRefinementMove &move) {
            return move.valid &&
                (move.delta_j > kEpsUpdate ||
                 (std::abs(move.delta_j) <= kEpsUpdate && move.delta_soft > kEpsUpdate));
        };

        auto atomized_move_better = [&](const AtomizedRefinementMove &lhs,
                                        const AtomizedRefinementMove &rhs) {
            if (!rhs.valid) {
                return lhs.valid;
            }
            if (!lhs.valid) {
                return false;
            }
            if (lhs.delta_j > rhs.delta_j + kEpsUpdate) {
                return true;
            }
            if (rhs.delta_j > lhs.delta_j + kEpsUpdate) {
                return false;
            }
            if (lhs.delta_hard > rhs.delta_hard + kEpsUpdate) {
                return true;
            }
            if (rhs.delta_hard > lhs.delta_hard + kEpsUpdate) {
                return false;
            }
            if (lhs.delta_soft > rhs.delta_soft + kEpsUpdate) {
                return true;
            }
            if (rhs.delta_soft > lhs.delta_soft + kEpsUpdate) {
                return false;
            }
            if (lhs.delta_components != rhs.delta_components) {
                return lhs.delta_components < rhs.delta_components;
            }
            if (lhs.length != rhs.length) {
                return lhs.length < rhs.length;
            }
            if (lhs.source_group != rhs.source_group) {
                return lhs.source_group < rhs.source_group;
            }
            if (lhs.target_group != rhs.target_group) {
                return lhs.target_group < rhs.target_group;
            }
            if (lhs.start != rhs.start) {
                return lhs.start < rhs.start;
            }
            return lhs.end < rhs.end;
        };

        if (!binary_mode_) {
            std::vector<int> assign = seed.partition;
            std::vector<int> branch_rows((size_t)groups, 0);
            std::vector<double> branch_class_weight(
                static_cast<size_t>(groups) * static_cast<size_t>(n_classes_), 0.0);
            for (int bin_pos = 0; bin_pos < bin_count; ++bin_pos) {
                const int group_idx = assign[(size_t)bin_pos];
                if (group_idx < 0 || group_idx >= groups) {
                    return seed;
                }
                branch_rows[(size_t)group_idx] += atoms[(size_t)bin_pos].row_count;
                const size_t base = static_cast<size_t>(group_idx) * static_cast<size_t>(n_classes_);
                for (int cls = 0; cls < n_classes_; ++cls) {
                    branch_class_weight[base + static_cast<size_t>(cls)] +=
                        atoms[(size_t)bin_pos].class_weight[(size_t)cls];
                }
            }

            std::vector<double> branch_loss((size_t)groups, 0.0);
            std::vector<double> branch_hard_impurity((size_t)groups, 0.0);
            for (int group_idx = 0; group_idx < groups; ++group_idx) {
                const size_t group_base =
                    static_cast<size_t>(group_idx) * static_cast<size_t>(n_classes_);
                branch_loss[(size_t)group_idx] =
                    split_leaf_loss_flat(branch_class_weight, n_classes_, group_base);
                branch_hard_impurity[(size_t)group_idx] =
                    hard_label_impurity_flat(branch_class_weight, n_classes_, group_base);
            }

            while (true) {
                check_timeout();
                std::vector<unsigned char> seen_source_groups((size_t)groups, 0);

                RankedMove best_descent;
                RankedMove best_bridge;
                int interval_start = 0;
                while (interval_start < bin_count) {
                    const int source_group = assign[(size_t)interval_start];
                    int interval_end = interval_start + 1;
                    while (interval_end < bin_count && assign[(size_t)interval_end] == source_group) {
                        ++interval_end;
                    }
                    if (collect_summary && !seen_source_groups[(size_t)source_group]) {
                        seen_source_groups[(size_t)source_group] = 1U;
                        bump_histogram(local_summary.source_group_row_size_histogram, branch_rows[(size_t)source_group]);
                    }
                    bump_histogram(local_summary.source_component_atom_size_histogram, interval_end - interval_start);
                    int interval_rows_total = 0;
                    for (int pos = interval_start; pos < interval_end; ++pos) {
                        interval_rows_total += atoms[(size_t)pos].row_count;
                    }
                    bump_histogram(local_summary.source_component_row_size_histogram, interval_rows_total);

                    for (int target_group = 0; target_group < groups; ++target_group) {
                        if (target_group == source_group) {
                            continue;
                        }
                        const int source_rows_total = branch_rows[(size_t)source_group];
                        const int target_rows_total = branch_rows[(size_t)target_group];
                        const double source_loss_before = branch_loss[(size_t)source_group];
                        const double target_loss_before = branch_loss[(size_t)target_group];
                        const double source_hard_before = branch_hard_impurity[(size_t)source_group];
                        const double target_hard_before = branch_hard_impurity[(size_t)target_group];
                        auto scan_interval_overlap = [&](int lo, int hi) {
                            if (lo > hi) {
                                return;
                            }
                            for (int start = lo; start <= hi; ++start) {
                                int move_rows = 0;
                                std::vector<double> move_class_weight((size_t)n_classes_, 0.0);
                                for (int end = start; end <= hi; ++end) {
                                    if (collect_summary) {
                                        ++local_summary.candidate_total;
                                    }
                                    const AtomizedBin &atom = atoms[(size_t)end];
                                    move_rows += atom.row_count;
                                    for (int cls = 0; cls < n_classes_; ++cls) {
                                        move_class_weight[(size_t)cls] += atom.class_weight[(size_t)cls];
                                    }

                                    const int source_rows_after = source_rows_total - move_rows;
                                    if (source_rows_after < min_child_size_) {
                                        if (collect_summary) {
                                            ++local_summary.candidate_source_size_rejects;
                                        }
                                        break;
                                    }
                                    const int target_rows_after = target_rows_total + move_rows;
                                    if (target_rows_after < min_child_size_) {
                                        if (collect_summary) {
                                            ++local_summary.candidate_target_size_rejects;
                                        }
                                        continue;
                                    }

                                    const size_t source_base =
                                        static_cast<size_t>(source_group) * static_cast<size_t>(n_classes_);
                                    const size_t target_base =
                                        static_cast<size_t>(target_group) * static_cast<size_t>(n_classes_);
                                    const double source_loss_after = split_leaf_loss_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_class_weight,
                                        false);
                                    const double target_loss_after = split_leaf_loss_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_class_weight,
                                        true);
                                    const double source_hard_after = hard_label_impurity_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_class_weight,
                                        false);
                                    const double target_hard_after = hard_label_impurity_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_class_weight,
                                        true);
                                    const double primary_before = hard_loss_mode
                                        ? (source_loss_before + target_loss_before)
                                        : (source_hard_before + target_hard_before);
                                    const double primary_after = hard_loss_mode
                                        ? (source_loss_after + target_loss_after)
                                        : (source_hard_after + target_hard_after);
                                    const double guide_before = hard_loss_mode
                                        ? (source_hard_before + target_hard_before)
                                        : (source_loss_before + target_loss_before);
                                    const double guide_after = hard_loss_mode
                                        ? (source_hard_after + target_hard_after)
                                        : (source_loss_after + target_loss_after);
                                    const double delta_primary = primary_before - primary_after;
                                    const double delta_guide = guide_before - guide_after;
                                    const int source_delta_components =
                                        (start == interval_start && end == interval_end - 1)
                                            ? -1
                                            : ((start == interval_start || end == interval_end - 1) ? 0 : 1);
                                    const bool left_target =
                                        (start > 0 && assign[(size_t)(start - 1)] == target_group);
                                    const bool right_target =
                                        (end + 1 < bin_count && assign[(size_t)(end + 1)] == target_group);
                                    int target_delta_components = 0;
                                    if (!left_target && !right_target) {
                                        target_delta_components = 1;
                                    } else if (left_target && right_target) {
                                        target_delta_components = -1;
                                    }

                                    AtomizedRefinementMove move;
                                    move.valid = true;
                                    move.source_group = source_group;
                                    move.target_group = target_group;
                                    move.start = start;
                                    move.end = end;
                                    move.length = end - start + 1;
                                    move.delta_components = source_delta_components + target_delta_components;
                                    move.row_count = move_rows;
                                    move.class_weight = move_class_weight;
                                    move.source_loss_after = source_loss_after;
                                    move.target_loss_after = target_loss_after;
                                    move.source_hard_impurity_after = source_hard_after;
                                    move.target_hard_impurity_after = target_hard_after;
                                    move.delta_hard = delta_primary;
                                    move.delta_soft = delta_guide;
                                    move.delta_j = delta_primary - mu_node * (double)move.delta_components;
                                    consider_ranked_move(
                                        std::move(move),
                                        delta_primary,
                                        delta_guide,
                                        best_descent,
                                        best_bridge);
                                }
                            }
                        };

                        if (active_windows == nullptr || active_windows->empty()) {
                            scan_interval_overlap(interval_start, interval_end - 1);
                            saw_overlap_segment = true;
                            if (collect_summary) {
                                ++local_summary.refine_overlap_segments;
                            }
                        } else {
                            for (const auto &window : *active_windows) {
                                const int overlap_start = std::max(interval_start, window.first);
                                const int overlap_end = std::min(interval_end - 1, window.second);
                                if (overlap_start <= overlap_end) {
                                    saw_overlap_segment = true;
                                    if (collect_summary) {
                                        ++local_summary.refine_overlap_segments;
                                    }
                                    scan_interval_overlap(overlap_start, overlap_end);
                                }
                            }
                        }
                    }

                interval_start = interval_end;
            }

            if (collect_summary) {
                if (saw_overlap_segment) {
                    ++local_summary.refine_calls_with_overlap;
                } else {
                    ++local_summary.refine_calls_without_overlap;
                }
            }

            const RankedMove *chosen_ranked = nullptr;
            const AtomizedRefinementMove *best_move_ptr = nullptr;
                if (best_descent.valid) {
                    chosen_ranked = &best_descent;
                } else if (bridge_allowed && !bridge_used && best_bridge.valid) {
                    chosen_ranked = &best_bridge;
                    bridge_used = true;
                } else {
                    break;
                }
                best_move_ptr = &chosen_ranked->move;
                const AtomizedRefinementMove &best_move = *best_move_ptr;

                for (int bin_pos = best_move.start; bin_pos <= best_move.end; ++bin_pos) {
                    assign[(size_t)bin_pos] = best_move.target_group;
                }

                branch_rows[(size_t)best_move.source_group] -= best_move.row_count;
                branch_rows[(size_t)best_move.target_group] += best_move.row_count;
                const size_t source_base =
                    static_cast<size_t>(best_move.source_group) * static_cast<size_t>(n_classes_);
                const size_t target_base =
                    static_cast<size_t>(best_move.target_group) * static_cast<size_t>(n_classes_);
                for (int cls = 0; cls < n_classes_; ++cls) {
                    branch_class_weight[source_base + static_cast<size_t>(cls)] -=
                        best_move.class_weight[(size_t)cls];
                    branch_class_weight[target_base + static_cast<size_t>(cls)] +=
                        best_move.class_weight[(size_t)cls];
                }
                branch_loss[(size_t)best_move.source_group] = best_move.source_loss_after;
                branch_loss[(size_t)best_move.target_group] = best_move.target_loss_after;
                branch_hard_impurity[(size_t)best_move.source_group] = best_move.source_hard_impurity_after;
                branch_hard_impurity[(size_t)best_move.target_group] = best_move.target_hard_impurity_after;

                ++move_count;
                if (collect_summary) {
                    local_summary.moves += 1;
                    local_summary.hard_gain += chosen_ranked->primary_gain;
                    local_summary.soft_gain += chosen_ranked->guide_gain;
                    if (chosen_ranked == &best_descent) {
                        ++local_summary.descent_moves;
                    } else {
                        ++local_summary.bridge_moves;
                    }
                    local_summary.delta_j += best_move.delta_j;
                    local_summary.component_delta += best_move.delta_components;
                }
            }

            while (true) {
                check_timeout();
                std::vector<unsigned char> seen_source_groups((size_t)groups, 0);

                RankedMove best_cleanup;
                int interval_start = 0;
                while (interval_start < bin_count) {
                    const int source_group = assign[(size_t)interval_start];
                    int interval_end = interval_start + 1;
                    while (interval_end < bin_count && assign[(size_t)interval_end] == source_group) {
                        ++interval_end;
                    }
                    if (collect_summary && !seen_source_groups[(size_t)source_group]) {
                        seen_source_groups[(size_t)source_group] = 1U;
                        bump_histogram(local_summary.source_group_row_size_histogram, branch_rows[(size_t)source_group]);
                    }
                    bump_histogram(local_summary.source_component_atom_size_histogram, interval_end - interval_start);

                    for (int target_group = 0; target_group < groups; ++target_group) {
                        if (target_group == source_group) {
                            continue;
                        }
                        const int source_rows_total = branch_rows[(size_t)source_group];
                        const int target_rows_total = branch_rows[(size_t)target_group];
                        const double source_loss_before = branch_loss[(size_t)source_group];
                        const double target_loss_before = branch_loss[(size_t)target_group];
                        const double source_hard_before = branch_hard_impurity[(size_t)source_group];
                        const double target_hard_before = branch_hard_impurity[(size_t)target_group];
                        auto scan_interval_overlap = [&](int lo, int hi) {
                            if (lo > hi) {
                                return;
                            }
                            for (int start = lo; start <= hi; ++start) {
                                int move_rows = 0;
                                std::vector<double> move_class_weight((size_t)n_classes_, 0.0);
                                for (int end = start; end <= hi; ++end) {
                                    if (collect_summary) {
                                        ++local_summary.candidate_total;
                                    }
                                    const AtomizedBin &atom = atoms[(size_t)end];
                                    move_rows += atom.row_count;
                                    for (int cls = 0; cls < n_classes_; ++cls) {
                                        move_class_weight[(size_t)cls] += atom.class_weight[(size_t)cls];
                                    }

                                    const int source_rows_after = source_rows_total - move_rows;
                                    if (source_rows_after < min_child_size_) {
                                        if (collect_summary) {
                                            ++local_summary.candidate_source_size_rejects;
                                        }
                                        break;
                                    }
                                    const int target_rows_after = target_rows_total + move_rows;
                                    if (target_rows_after < min_child_size_) {
                                        if (collect_summary) {
                                            ++local_summary.candidate_target_size_rejects;
                                        }
                                        continue;
                                    }

                                    const size_t source_base =
                                        static_cast<size_t>(source_group) * static_cast<size_t>(n_classes_);
                                    const size_t target_base =
                                        static_cast<size_t>(target_group) * static_cast<size_t>(n_classes_);
                                    const double source_loss_after = split_leaf_loss_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_class_weight,
                                        false);
                                    const double target_loss_after = split_leaf_loss_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_class_weight,
                                        true);
                                    const double source_hard_after = hard_label_impurity_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_class_weight,
                                        false);
                                    const double target_hard_after = hard_label_impurity_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_class_weight,
                                        true);
                                    const double source_soft_after = 0.0;
                                    const double target_soft_after = 0.0;
                                    const double primary_gain = hard_loss_mode
                                        ? ((source_loss_before + target_loss_before) -
                                           (source_loss_after + target_loss_after))
                                        : ((source_hard_before + target_hard_before) -
                                           (source_hard_after + target_hard_after));
                                    const double guide_gain = hard_loss_mode
                                        ? ((source_hard_before + target_hard_before) -
                                           (source_hard_after + target_hard_after))
                                        : ((source_loss_before + target_loss_before) -
                                           (source_loss_after + target_loss_after));
                                    const int source_delta_components =
                                        (start == interval_start && end == interval_end - 1)
                                            ? -1
                                            : ((start == interval_start || end == interval_end - 1) ? 0 : 1);
                                    const bool left_target =
                                        (start > 0 && assign[(size_t)(start - 1)] == target_group);
                                    const bool right_target =
                                        (end + 1 < bin_count && assign[(size_t)(end + 1)] == target_group);
                                    int target_delta_components = 0;
                                    if (!left_target && !right_target) {
                                        target_delta_components = 1;
                                    } else if (left_target && right_target) {
                                        target_delta_components = -1;
                                    }

                                    AtomizedRefinementMove move;
                                    move.valid = true;
                                    move.source_group = source_group;
                                    move.target_group = target_group;
                                    move.start = start;
                                    move.end = end;
                                    move.length = end - start + 1;
                                    move.delta_components = source_delta_components + target_delta_components;
                                    move.row_count = move_rows;
                                    move.class_weight = move_class_weight;
                                    move.source_loss_after = source_loss_after;
                                    move.target_loss_after = target_loss_after;
                                    move.source_hard_impurity_after = source_hard_after;
                                    move.target_hard_impurity_after = target_hard_after;
                                    move.delta_hard = primary_gain;
                                    move.delta_soft = guide_gain;
                                    move.delta_j = primary_gain - mu_node * (double)move.delta_components;
                                    const double complexity_gain = -static_cast<double>(move.delta_components);
                                    if (collect_summary) {
                                        ++local_summary.candidate_legal;
                                    }
                                    if (std::abs(primary_gain) > kEpsUpdate) {
                                        if (collect_summary) {
                                            ++local_summary.candidate_cleanup_primary_rejected;
                                            ++local_summary.candidate_score_rejected;
                                        }
                                        continue;
                                    }
                                    if (complexity_gain <= kEpsUpdate) {
                                        if (collect_summary) {
                                            ++local_summary.candidate_cleanup_complexity_rejected;
                                            ++local_summary.candidate_score_rejected;
                                        }
                                        continue;
                                    }
                                    if (collect_summary) {
                                        ++local_summary.candidate_cleanup_eligible;
                                    }
                                    RankedMove ranked = make_ranked_move(
                                        std::move(move),
                                        primary_gain,
                                        guide_gain);
                                    if (!best_cleanup.valid || ranked_move_better_cleanup(ranked, best_cleanup)) {
                                        best_cleanup = std::move(ranked);
                                    }
                                }
                            }
                        };

                        if (active_windows == nullptr || active_windows->empty()) {
                            scan_interval_overlap(interval_start, interval_end - 1);
                        } else {
                            for (const auto &window : *active_windows) {
                                const int overlap_start = std::max(interval_start, window.first);
                                const int overlap_end = std::min(interval_end - 1, window.second);
                                if (overlap_start <= overlap_end) {
                                    scan_interval_overlap(overlap_start, overlap_end);
                                }
                            }
                        }
                    }

                    interval_start = interval_end;
                }

                if (!best_cleanup.valid) {
                    break;
                }

                const AtomizedRefinementMove &best_cleanup_move = best_cleanup.move;
                for (int bin_pos = best_cleanup_move.start; bin_pos <= best_cleanup_move.end; ++bin_pos) {
                    assign[(size_t)bin_pos] = best_cleanup_move.target_group;
                }

                branch_rows[(size_t)best_cleanup_move.source_group] -= best_cleanup_move.row_count;
                branch_rows[(size_t)best_cleanup_move.target_group] += best_cleanup_move.row_count;
                const size_t source_base =
                    static_cast<size_t>(best_cleanup_move.source_group) * static_cast<size_t>(n_classes_);
                const size_t target_base =
                    static_cast<size_t>(best_cleanup_move.target_group) * static_cast<size_t>(n_classes_);
                for (int cls = 0; cls < n_classes_; ++cls) {
                    branch_class_weight[source_base + static_cast<size_t>(cls)] -=
                        best_cleanup_move.class_weight[(size_t)cls];
                    branch_class_weight[target_base + static_cast<size_t>(cls)] +=
                        best_cleanup_move.class_weight[(size_t)cls];
                }
                branch_loss[(size_t)best_cleanup_move.source_group] = best_cleanup_move.source_loss_after;
                branch_loss[(size_t)best_cleanup_move.target_group] = best_cleanup_move.target_loss_after;
                branch_hard_impurity[(size_t)best_cleanup_move.source_group] = best_cleanup_move.source_hard_impurity_after;
                branch_hard_impurity[(size_t)best_cleanup_move.target_group] = best_cleanup_move.target_hard_impurity_after;

                ++move_count;
                if (collect_summary) {
                    local_summary.moves += 1;
                    ++local_summary.simplify_moves;
                    local_summary.hard_gain += best_cleanup.primary_gain;
                    local_summary.soft_gain += best_cleanup.guide_gain;
                    local_summary.delta_j += best_cleanup_move.delta_j;
                    local_summary.component_delta += best_cleanup_move.delta_components;
                }
            }

            if (move_count == 0) {
                if (collect_summary) {
                    *summary = local_summary;
                }
                return seed;
            }

            AtomizedCandidate refined = candidate_from_partition(
                seed.feature,
                atoms,
                assign,
                groups,
                adjacency_bonus,
                adjacency_bonus_total,
                seed.hard_loss_mode ? AtomizedObjectiveMode::kHardLoss : AtomizedObjectiveMode::kImpurity);
            if (!refined.feasible) {
                return seed;
            }
            const bool improved =
                atomized_score_better_for_refinement(refined.score, seed.score, mu_node, mode);
            if (collect_summary) {
                local_summary.improved = improved;
                *summary = local_summary;
            }
            return improved ? refined : seed;
        }

        std::vector<int> assign = seed.partition;
        std::vector<int> branch_rows((size_t)groups, 0);
        std::vector<double> branch_pos((size_t)groups, 0.0);
        std::vector<double> branch_neg((size_t)groups, 0.0);
        for (int bin_pos = 0; bin_pos < bin_count; ++bin_pos) {
            const int group_idx = assign[(size_t)bin_pos];
            if (group_idx < 0 || group_idx >= groups) {
                return seed;
            }
            const AtomizedBin &atom = atoms[(size_t)bin_pos];
            branch_rows[(size_t)group_idx] += atom.row_count;
            branch_pos[(size_t)group_idx] += atom.pos_weight;
            branch_neg[(size_t)group_idx] += atom.neg_weight;
        }
        std::vector<double> branch_loss((size_t)groups, 0.0);
        std::vector<double> branch_hard_impurity((size_t)groups, 0.0);
        for (int group_idx = 0; group_idx < groups; ++group_idx) {
            branch_loss[(size_t)group_idx] = split_leaf_loss(
                branch_pos[(size_t)group_idx],
                branch_neg[(size_t)group_idx]);
            branch_hard_impurity[(size_t)group_idx] = hard_label_impurity(
                branch_pos[(size_t)group_idx],
                branch_neg[(size_t)group_idx]);
        }
        while (true) {
            check_timeout();
            std::vector<unsigned char> seen_source_groups((size_t)groups, 0);

            std::vector<double> branch_primary((size_t)groups, 0.0);
            std::vector<double> branch_secondary((size_t)groups, 0.0);
            for (int group_idx = 0; group_idx < groups; ++group_idx) {
                if (hard_loss_mode) {
                    branch_primary[(size_t)group_idx] = split_leaf_loss(
                        branch_pos[(size_t)group_idx],
                        branch_neg[(size_t)group_idx]);
                    branch_secondary[(size_t)group_idx] = branch_hard_impurity[(size_t)group_idx];
                } else {
                    branch_primary[(size_t)group_idx] = branch_hard_impurity[(size_t)group_idx];
                    branch_secondary[(size_t)group_idx] = branch_loss[(size_t)group_idx];
                }
            }

            AtomizedRefinementMove best_move;
            int interval_start = 0;
            while (interval_start < bin_count) {
                const int source_group = assign[(size_t)interval_start];
                int interval_end = interval_start + 1;
                while (interval_end < bin_count && assign[(size_t)interval_end] == source_group) {
                    ++interval_end;
                }
                if (collect_summary && !seen_source_groups[(size_t)source_group]) {
                    seen_source_groups[(size_t)source_group] = 1U;
                    bump_histogram(local_summary.source_group_row_size_histogram, branch_rows[(size_t)source_group]);
                }
                bump_histogram(local_summary.source_component_atom_size_histogram, interval_end - interval_start);
                int interval_rows_total = 0;
                for (int pos = interval_start; pos < interval_end; ++pos) {
                    interval_rows_total += atoms[(size_t)pos].row_count;
                }
                bump_histogram(local_summary.source_component_row_size_histogram, interval_rows_total);

                for (int target_group = 0; target_group < groups; ++target_group) {
                    if (target_group == source_group) {
                        continue;
                    }
                    const int source_rows_total = branch_rows[(size_t)source_group];
                    const int target_rows_total = branch_rows[(size_t)target_group];
                    const double source_pos_total = branch_pos[(size_t)source_group];
                    const double source_neg_total = branch_neg[(size_t)source_group];
                    const double target_pos_total = branch_pos[(size_t)target_group];
                    const double target_neg_total = branch_neg[(size_t)target_group];
                    const double source_primary_before = branch_primary[(size_t)source_group];
                    const double target_primary_before = branch_primary[(size_t)target_group];
                    const double source_secondary_before = branch_secondary[(size_t)source_group];
                    const double target_secondary_before = branch_secondary[(size_t)target_group];
                    auto scan_interval_overlap = [&](int lo, int hi) {
                        if (lo > hi) {
                            return;
                        }
                            for (int start = lo; start <= hi; ++start) {
                                int move_rows = 0;
                                double move_pos = 0.0;
                                double move_neg = 0.0;
                                for (int end = start; end <= hi; ++end) {
                                    if (collect_summary) {
                                        ++local_summary.candidate_total;
                                    }
                                    const AtomizedBin &atom = atoms[(size_t)end];
                                    move_rows += atom.row_count;
                                    move_pos += atom.pos_weight;
                                    move_neg += atom.neg_weight;

                                    const int source_rows_after = source_rows_total - move_rows;
                                    if (source_rows_after < min_child_size_) {
                                        if (collect_summary) {
                                            ++local_summary.candidate_source_size_rejects;
                                        }
                                        break;
                                    }
                                    const int target_rows_after = target_rows_total + move_rows;
                                    if (target_rows_after < min_child_size_) {
                                        if (collect_summary) {
                                            ++local_summary.candidate_target_size_rejects;
                                        }
                                        continue;
                                    }

                                double source_primary_after = 0.0;
                                double target_primary_after = 0.0;
                                double source_secondary_after = 0.0;
                                double target_secondary_after = 0.0;
                                if (hard_loss_mode) {
                                    source_primary_after = split_leaf_loss(
                                        source_pos_total - move_pos,
                                        source_neg_total - move_neg);
                                    target_primary_after = split_leaf_loss(
                                        target_pos_total + move_pos,
                                        target_neg_total + move_neg);
                                    source_secondary_after = hard_label_impurity(
                                        source_pos_total - move_pos,
                                        source_neg_total - move_neg);
                                    target_secondary_after = hard_label_impurity(
                                        target_pos_total + move_pos,
                                        target_neg_total + move_neg);
                                } else {
                                    source_primary_after = hard_label_impurity(
                                        source_pos_total - move_pos,
                                        source_neg_total - move_neg);
                                    target_primary_after = hard_label_impurity(
                                        target_pos_total + move_pos,
                                        target_neg_total + move_neg);
                                    source_secondary_after = split_leaf_loss(
                                        source_pos_total - move_pos,
                                        source_neg_total - move_neg);
                                    target_secondary_after = split_leaf_loss(
                                        target_pos_total + move_pos,
                                        target_neg_total + move_neg);
                                }
                                const double delta_primary =
                                    (source_primary_before + target_primary_before) -
                                    (source_primary_after + target_primary_after);
                                const double delta_secondary =
                                    (source_secondary_before + target_secondary_before) -
                                    (source_secondary_after + target_secondary_after);
                                const int source_delta_components =
                                    (start == interval_start && end == interval_end - 1)
                                        ? -1
                                        : ((start == interval_start || end == interval_end - 1) ? 0 : 1);
                                const bool left_target = (start > 0 && assign[(size_t)(start - 1)] == target_group);
                                const bool right_target = (end + 1 < bin_count && assign[(size_t)(end + 1)] == target_group);
                                int target_delta_components = 0;
                                if (!left_target && !right_target) {
                                    target_delta_components = 1;
                                } else if (left_target && right_target) {
                                    target_delta_components = -1;
                                }

                                AtomizedRefinementMove move;
                                move.valid = true;
                                move.source_group = source_group;
                                move.target_group = target_group;
                                move.start = start;
                                move.end = end;
                                move.length = end - start + 1;
                                move.delta_components = source_delta_components + target_delta_components;
                                move.row_count = move_rows;
                                move.pos_weight = move_pos;
                                move.neg_weight = move_neg;
                                move.source_loss_after = hard_loss_mode ? source_primary_after : source_secondary_after;
                                move.target_loss_after = hard_loss_mode ? target_primary_after : target_secondary_after;
                                move.source_hard_impurity_after = hard_loss_mode ? source_secondary_after : source_primary_after;
                                move.target_hard_impurity_after = hard_loss_mode ? target_secondary_after : target_primary_after;
                                move.delta_hard = delta_primary;
                                move.delta_soft = delta_secondary;
                                move.delta_j = delta_primary - mu_node * (double)move.delta_components;
                                if (collect_summary) {
                                    ++local_summary.candidate_legal;
                                }
                                if (!atomized_move_is_improving(move)) {
                                    if (collect_summary) {
                                        ++local_summary.candidate_score_rejected;
                                    }
                                    continue;
                                }
                                if (collect_summary) {
                                    ++local_summary.candidate_descent_eligible;
                                }
                                if (!best_move.valid || atomized_move_better(move, best_move)) {
                                    best_move = std::move(move);
                                }
                                }
                            }
                        };

                    if (active_windows == nullptr || active_windows->empty()) {
                        scan_interval_overlap(interval_start, interval_end - 1);
                    } else {
                        for (const auto &window : *active_windows) {
                            const int overlap_start = std::max(interval_start, window.first);
                            const int overlap_end = std::min(interval_end - 1, window.second);
                            if (overlap_start <= overlap_end) {
                                scan_interval_overlap(overlap_start, overlap_end);
                            }
                        }
                    }
                }

                interval_start = interval_end;
            }

            if (!best_move.valid) {
                break;
            }

            for (int bin_pos = best_move.start; bin_pos <= best_move.end; ++bin_pos) {
                assign[(size_t)bin_pos] = best_move.target_group;
            }

            branch_rows[(size_t)best_move.source_group] -= best_move.row_count;
            branch_rows[(size_t)best_move.target_group] += best_move.row_count;
            branch_pos[(size_t)best_move.source_group] -= best_move.pos_weight;
            branch_pos[(size_t)best_move.target_group] += best_move.pos_weight;
            branch_neg[(size_t)best_move.source_group] -= best_move.neg_weight;
            branch_neg[(size_t)best_move.target_group] += best_move.neg_weight;
            branch_loss[(size_t)best_move.source_group] = best_move.source_loss_after;
            branch_loss[(size_t)best_move.target_group] = best_move.target_loss_after;
            branch_hard_impurity[(size_t)best_move.source_group] = best_move.source_hard_impurity_after;
            branch_hard_impurity[(size_t)best_move.target_group] = best_move.target_hard_impurity_after;

            ++move_count;
            if (collect_summary) {
                local_summary.moves += 1;
                local_summary.hard_gain += best_move.delta_j + mu_node * (double)best_move.delta_components;
                local_summary.soft_gain += best_move.delta_soft;
                ++local_summary.descent_moves;
                local_summary.delta_j += best_move.delta_j;
                local_summary.component_delta += best_move.delta_components;
            }
        }

        if (move_count == 0) {
            if (collect_summary) {
                *summary = local_summary;
            }
            return seed;
        }

        AtomizedCandidate refined = candidate_from_partition(
            seed.feature,
            atoms,
            assign,
            groups,
            adjacency_bonus,
            adjacency_bonus_total,
            seed.hard_loss_mode ? AtomizedObjectiveMode::kHardLoss : AtomizedObjectiveMode::kImpurity);
        if (!refined.feasible) {
            return seed;
        }
        const bool improved = atomized_score_better_for_refinement(refined.score, seed.score, mu_node, mode);
        if (collect_summary) {
            local_summary.improved = improved;
            *summary = local_summary;
        }
        return improved ? refined : seed;
    }

    std::vector<AtomizedCandidatePair> solve_atomized_geometry_family_pairs(
        const std::vector<AtomizedBin> &atoms,
        const AtomizedPrefixes &prefix,
        int feature,
        int max_groups
    ) const {
        std::vector<AtomizedCandidatePair> out;
        const int m = (int)atoms.size();
        if (max_groups < 2 || max_groups > m) {
            return out;
        }
        out.resize((size_t)max_groups + 1U);

        const int stride = m + 1;
        auto at = [stride](int g, int t) -> size_t {
            return static_cast<size_t>(g) * static_cast<size_t>(stride) + static_cast<size_t>(t);
        };

        std::vector<AtomizedScore> dp_impurity(
            static_cast<size_t>(max_groups + 1) * static_cast<size_t>(stride));
        std::vector<AtomizedScore> dp_hardloss(
            static_cast<size_t>(max_groups + 1) * static_cast<size_t>(stride));
        std::vector<int> parent_impurity(
            static_cast<size_t>(max_groups + 1) * static_cast<size_t>(stride),
            -1);
        std::vector<int> parent_hardloss(
            static_cast<size_t>(max_groups + 1) * static_cast<size_t>(stride),
            -1);
        dp_impurity[at(0, 0)] = AtomizedScore{0.0, 0.0, 0.0, 0.0, 0.0, 0};
        dp_hardloss[at(0, 0)] = AtomizedScore{0.0, 0.0, 0.0, 0.0, 0.0, 0};

        auto state_feasible = [&](const std::vector<AtomizedScore> &dp,
                                  int groups,
                                  int end_pos,
                                  AtomizedObjectiveMode mode) {
            return std::isfinite(atomized_primary_objective(dp[at(groups, end_pos)], mode));
        };

        auto cut_boundary_penalty = [&](int p) {
            return (p > 0)
                ? -contiguous_boundary_bonus(feature, atoms[(size_t)(p - 1)], atoms[(size_t)p])
                : 0.0;
        };

        auto segment_hard_loss = [&](int p, int t) {
            if (binary_mode_) {
                const double seg_pos = prefix.pos[(size_t)t] - prefix.pos[(size_t)p];
                const double seg_neg = prefix.neg[(size_t)t] - prefix.neg[(size_t)p];
                return split_leaf_loss(seg_pos, seg_neg);
            }
            return split_leaf_loss_prefix_segment(
                prefix.class_weight_prefix,
                n_classes_,
                p,
                t);
        };

        auto segment_hard_impurity = [&](int p, int t) {
            if (binary_mode_) {
                const double seg_pos = prefix.pos[(size_t)t] - prefix.pos[(size_t)p];
                const double seg_neg = prefix.neg[(size_t)t] - prefix.neg[(size_t)p];
                return hard_label_impurity(seg_pos, seg_neg);
            }
            return hard_label_impurity_prefix_segment(
                prefix.class_weight_prefix,
                n_classes_,
                p,
                t);
        };

        std::vector<int> impurity_prune_activation((size_t)m + 1U, m + 1);
        int activation_ptr = 1;
        for (int t = 0; t <= m; ++t) {
            if (activation_ptr < t + 1) {
                activation_ptr = t + 1;
            }
            while (activation_ptr <= m &&
                   prefix.rows[(size_t)activation_ptr] - prefix.rows[(size_t)t] < min_child_size_) {
                ++activation_ptr;
            }
            impurity_prune_activation[(size_t)t] = activation_ptr;
        }

        auto extend_score = [&](const AtomizedScore &prev, int p, int t) {
            AtomizedScore cand = prev;
            const double seg_hard_loss = segment_hard_loss(p, t);
            cand.hard_loss += seg_hard_loss;
            cand.soft_loss += seg_hard_loss;
            cand.hard_impurity += segment_hard_impurity(p, t);
            cand.boundary_penalty += cut_boundary_penalty(p);
            cand.components += 1;
            return cand;
        };

        struct RunningMinimum {
            bool valid = false;
            double value = kInfinity;
            double boundary = kInfinity;
            int parent = -1;
        };

        auto running_minimum_better = [&](double value, double boundary, int parent, const RunningMinimum &best) {
            if (!best.valid) {
                return true;
            }
            if (value < best.value - kEpsUpdate) {
                return true;
            }
            if (best.value < value - kEpsUpdate) {
                return false;
            }
            if (boundary < best.boundary - kEpsUpdate) {
                return true;
            }
            if (best.boundary < boundary - kEpsUpdate) {
                return false;
            }
            return parent < best.parent;
        };

        auto candidate_better = [&](double lhs_primary,
                                    double lhs_boundary,
                                    int lhs_parent,
                                    double rhs_primary,
                                    double rhs_boundary,
                                    int rhs_parent) {
            if (lhs_parent < 0) {
                return false;
            }
            if (rhs_parent < 0) {
                return true;
            }
            if (lhs_primary < rhs_primary - kEpsUpdate) {
                return true;
            }
            if (rhs_primary < lhs_primary - kEpsUpdate) {
                return false;
            }
            if (lhs_boundary < rhs_boundary - kEpsUpdate) {
                return true;
            }
            if (rhs_boundary < lhs_boundary - kEpsUpdate) {
                return false;
            }
            return lhs_parent < rhs_parent;
        };

        for (int g = 1; g <= max_groups; ++g) {
            if (binary_mode_) {
                RunningMinimum plus_best;
                RunningMinimum minus_best;
                int next_insert = g - 1;
                for (int t = g; t <= m; ++t) {
                    while (next_insert < t &&
                           prefix.rows[(size_t)next_insert] <= prefix.rows[(size_t)t] - min_child_size_) {
                        if (state_feasible(dp_hardloss, g - 1, next_insert, AtomizedObjectiveMode::kHardLoss)) {
                            const AtomizedScore &prev = dp_hardloss[at(g - 1, next_insert)];
                            const double total_prefix = prefix.total_weight[(size_t)next_insert];
                            const double margin_prefix =
                                prefix.pos[(size_t)next_insert] - prefix.neg[(size_t)next_insert];
                            const double boundary = prev.boundary_penalty + cut_boundary_penalty(next_insert);
                            const double plus_value =
                                prev.hard_loss - 0.5 * total_prefix + 0.5 * margin_prefix;
                            const double minus_value =
                                prev.hard_loss - 0.5 * total_prefix - 0.5 * margin_prefix;
                            if (running_minimum_better(
                                    plus_value,
                                    boundary,
                                    next_insert,
                                    plus_best)) {
                                plus_best = RunningMinimum{
                                    true,
                                    plus_value,
                                    boundary,
                                    next_insert};
                            }
                            if (running_minimum_better(
                                    minus_value,
                                    boundary,
                                    next_insert,
                                    minus_best)) {
                                minus_best = RunningMinimum{
                                    true,
                                    minus_value,
                                    boundary,
                                    next_insert};
                            }
                        }
                        ++next_insert;
                    }

                    const double total_t = prefix.total_weight[(size_t)t];
                    const double margin_t = prefix.pos[(size_t)t] - prefix.neg[(size_t)t];
                    double best_primary = kInfinity;
                    double best_boundary = kInfinity;
                    int best_parent = -1;
                    if (plus_best.valid) {
                        const double cand_primary = 0.5 * total_t + plus_best.value - 0.5 * margin_t;
                        if (candidate_better(
                                cand_primary,
                                plus_best.boundary,
                                plus_best.parent,
                                best_primary,
                                best_boundary,
                                best_parent)) {
                            best_primary = cand_primary;
                            best_boundary = plus_best.boundary;
                            best_parent = plus_best.parent;
                        }
                    }
                    if (minus_best.valid) {
                        const double cand_primary = 0.5 * total_t + minus_best.value + 0.5 * margin_t;
                        if (candidate_better(
                                cand_primary,
                                minus_best.boundary,
                                minus_best.parent,
                                best_primary,
                                best_boundary,
                                best_parent)) {
                            best_primary = cand_primary;
                            best_boundary = minus_best.boundary;
                            best_parent = minus_best.parent;
                        }
                    }
                    if (best_parent >= 0) {
                        dp_hardloss[at(g, t)] = extend_score(
                            dp_hardloss[at(g - 1, best_parent)],
                            best_parent,
                            t);
                        parent_hardloss[at(g, t)] = best_parent;
                    }
                }
            } else {
                std::vector<RunningMinimum> best_by_class((size_t)n_classes_);
                int next_insert = g - 1;
                for (int t = g; t <= m; ++t) {
                    while (next_insert < t &&
                           prefix.rows[(size_t)next_insert] <= prefix.rows[(size_t)t] - min_child_size_) {
                        if (state_feasible(dp_hardloss, g - 1, next_insert, AtomizedObjectiveMode::kHardLoss)) {
                            const AtomizedScore &prev = dp_hardloss[at(g - 1, next_insert)];
                            const double total_prefix = prefix.total_weight[(size_t)next_insert];
                            const double boundary = prev.boundary_penalty + cut_boundary_penalty(next_insert);
                            const size_t base =
                                static_cast<size_t>(next_insert) * static_cast<size_t>(n_classes_);
                            for (int cls = 0; cls < n_classes_; ++cls) {
                                const double value =
                                    prev.hard_loss - total_prefix +
                                    prefix.class_weight_prefix[base + static_cast<size_t>(cls)];
                                RunningMinimum &best = best_by_class[(size_t)cls];
                                if (running_minimum_better(value, boundary, next_insert, best)) {
                                    best = RunningMinimum{true, value, boundary, next_insert};
                                }
                            }
                        }
                        ++next_insert;
                    }

                    const double total_t = prefix.total_weight[(size_t)t];
                    const size_t base_t =
                        static_cast<size_t>(t) * static_cast<size_t>(n_classes_);
                    double best_primary = kInfinity;
                    double best_boundary = kInfinity;
                    int best_parent = -1;
                    for (int cls = 0; cls < n_classes_; ++cls) {
                        const RunningMinimum &best = best_by_class[(size_t)cls];
                        if (!best.valid) {
                            continue;
                        }
                        const double cand_primary =
                            total_t + best.value -
                            prefix.class_weight_prefix[base_t + static_cast<size_t>(cls)];
                        if (candidate_better(
                                cand_primary,
                                best.boundary,
                                best.parent,
                                best_primary,
                                best_boundary,
                                best_parent)) {
                            best_primary = cand_primary;
                            best_boundary = best.boundary;
                            best_parent = best.parent;
                        }
                    }
                    if (best_parent >= 0) {
                        dp_hardloss[at(g, t)] = extend_score(
                            dp_hardloss[at(g - 1, best_parent)],
                            best_parent,
                            t);
                        parent_hardloss[at(g, t)] = best_parent;
                    }
                }
            }

            std::vector<int> active_impurity;
            std::vector<unsigned char> active_impurity_flag((size_t)m + 1U, 0U);
            std::vector<int> scheduled_prune_at((size_t)m + 1U, m + 1);
            std::vector<int> prune_event_head((size_t)m + 2U, -1);
            std::vector<int> prune_event_next((size_t)m + 1U, -1);
            size_t live_impurity_count = 0U;
            if (state_feasible(dp_impurity, g - 1, g - 1, AtomizedObjectiveMode::kImpurity)) {
                active_impurity.push_back(g - 1);
                active_impurity_flag[(size_t)(g - 1)] = 1U;
                live_impurity_count = 1U;
            }
            for (int t = g; t <= m; ++t) {
                for (int p = prune_event_head[(size_t)t]; p >= 0; p = prune_event_next[(size_t)p]) {
                    if (scheduled_prune_at[(size_t)p] == t &&
                        active_impurity_flag[(size_t)p]) {
                        active_impurity_flag[(size_t)p] = 0U;
                        if (live_impurity_count > 0U) {
                            --live_impurity_count;
                        }
                    }
                }
                const bool prev_t_feasible =
                    state_feasible(dp_impurity, g - 1, t, AtomizedObjectiveMode::kImpurity);
                const double prev_t_primary =
                    prev_t_feasible ? dp_impurity[at(g - 1, t)].hard_impurity : kInfinity;
                const int activation =
                    prev_t_feasible ? impurity_prune_activation[(size_t)t] : (m + 1);
                double best_primary = kInfinity;
                double best_boundary = kInfinity;
                int best_parent = -1;
                for (int p : active_impurity) {
                    if (!active_impurity_flag[(size_t)p]) {
                        continue;
                    }
                    const double cand_primary =
                        dp_impurity[at(g - 1, p)].hard_impurity +
                        segment_hard_impurity(p, t);
                    if (prev_t_feasible &&
                        activation <= m &&
                        cand_primary > prev_t_primary + kEpsUpdate &&
                        activation < scheduled_prune_at[(size_t)p]) {
                        scheduled_prune_at[(size_t)p] = activation;
                        prune_event_next[(size_t)p] = prune_event_head[(size_t)activation];
                        prune_event_head[(size_t)activation] = p;
                    }
                    if (prefix.rows[(size_t)t] - prefix.rows[(size_t)p] < min_child_size_) {
                        continue;
                    }
                    const double cand_boundary =
                        dp_impurity[at(g - 1, p)].boundary_penalty +
                        cut_boundary_penalty(p);
                    if (candidate_better(
                            cand_primary,
                            cand_boundary,
                            p,
                            best_primary,
                            best_boundary,
                            best_parent)) {
                        best_primary = cand_primary;
                        best_boundary = cand_boundary;
                        best_parent = p;
                    }
                }
                if (best_parent >= 0) {
                    dp_impurity[at(g, t)] = extend_score(
                        dp_impurity[at(g - 1, best_parent)],
                        best_parent,
                        t);
                    parent_impurity[at(g, t)] = best_parent;
                }
                if (prev_t_feasible) {
                    active_impurity_flag[(size_t)t] = 1U;
                    active_impurity.push_back(t);
                    ++live_impurity_count;
                }
                if (active_impurity.size() > 64U &&
                    live_impurity_count * 2U < active_impurity.size()) {
                    std::vector<int> compacted;
                    compacted.reserve(live_impurity_count + 1U);
                    for (int p : active_impurity) {
                        if (active_impurity_flag[(size_t)p]) {
                            compacted.push_back(p);
                        }
                    }
                    active_impurity.swap(compacted);
                }
            }
        }

        auto build_candidate = [&](int groups,
                                   const std::vector<AtomizedScore> &dp,
                                   const std::vector<int> &parent,
                                   AtomizedObjectiveMode mode) {
            AtomizedCandidate candidate;
            if (!std::isfinite(atomized_primary_objective(dp[at(groups, m)], mode))) {
                return candidate;
            }
            std::vector<int> partition((size_t)m, -1);
            int t = m;
            int g = groups;
            int group_idx = groups - 1;
            while (g > 0) {
                const int p = parent[at(g, t)];
                if (p < 0) {
                    return AtomizedCandidate{};
                }
                for (int pos = p; pos < t; ++pos) {
                    partition[(size_t)pos] = group_idx;
                }
                t = p;
                --g;
                --group_idx;
            }
            candidate.feasible = true;
            candidate.score = dp[at(groups, m)];
            candidate.groups = groups;
            candidate.partition = std::move(partition);
            candidate.feature = feature;
            return candidate;
        };

        for (int groups = 2; groups <= max_groups; ++groups) {
            out[(size_t)groups].impurity = build_candidate(
                groups,
                dp_impurity,
                parent_impurity,
                AtomizedObjectiveMode::kImpurity);
            out[(size_t)groups].misclassification = build_candidate(
                groups,
                dp_hardloss,
                parent_hardloss,
                AtomizedObjectiveMode::kHardLoss);
        }
        return out;
    }

    void record_refinement_summary(const AtomizedRefinementSummary &summary) const {
        if (!diagnostics_enabled()) {
            return;
        }
        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        auto accumulate_histogram = [](const std::vector<long long> &src, std::vector<long long> &dst) {
            if (src.size() > dst.size()) {
                dst.resize(src.size(), 0);
            }
            for (size_t i = 0; i < src.size(); ++i) {
                dst[i] += src[i];
            }
        };
        ++telemetry.partition_refinement_refine_calls;
        telemetry.partition_refinement_total_moves += summary.moves;
        telemetry.partition_refinement_bridge_policy_calls += summary.bridge_policy_calls;
        telemetry.partition_refinement_refine_windowed_calls += summary.refine_windowed_calls;
        telemetry.partition_refinement_refine_unwindowed_calls += summary.refine_unwindowed_calls;
        telemetry.partition_refinement_refine_overlap_segments += summary.refine_overlap_segments;
        telemetry.partition_refinement_refine_calls_with_overlap += summary.refine_calls_with_overlap;
        telemetry.partition_refinement_refine_calls_without_overlap += summary.refine_calls_without_overlap;
        telemetry.partition_refinement_candidate_total += summary.candidate_total;
        telemetry.partition_refinement_candidate_legal += summary.candidate_legal;
        telemetry.partition_refinement_candidate_source_size_rejects += summary.candidate_source_size_rejects;
        telemetry.partition_refinement_candidate_target_size_rejects += summary.candidate_target_size_rejects;
        telemetry.partition_refinement_candidate_descent_eligible += summary.candidate_descent_eligible;
        telemetry.partition_refinement_candidate_descent_rejected += summary.candidate_descent_rejected;
        telemetry.partition_refinement_candidate_bridge_eligible += summary.candidate_bridge_eligible;
        telemetry.partition_refinement_candidate_bridge_window_blocked += summary.candidate_bridge_window_blocked;
        telemetry.partition_refinement_candidate_bridge_used_blocked += summary.candidate_bridge_used_blocked;
        telemetry.partition_refinement_candidate_bridge_guide_rejected += summary.candidate_bridge_guide_rejected;
        telemetry.partition_refinement_candidate_cleanup_eligible += summary.candidate_cleanup_eligible;
        telemetry.partition_refinement_candidate_cleanup_primary_rejected += summary.candidate_cleanup_primary_rejected;
        telemetry.partition_refinement_candidate_cleanup_complexity_rejected += summary.candidate_cleanup_complexity_rejected;
        telemetry.partition_refinement_candidate_score_rejected += summary.candidate_score_rejected;
        telemetry.partition_refinement_descent_moves += summary.descent_moves;
        telemetry.partition_refinement_bridge_moves += summary.bridge_moves;
        telemetry.partition_refinement_simplify_moves += summary.simplify_moves;
        accumulate_histogram(summary.source_group_row_size_histogram, telemetry.partition_refinement_source_group_row_size_histogram);
        accumulate_histogram(
            summary.source_component_atom_size_histogram,
            telemetry.partition_refinement_source_component_atom_size_histogram);
        accumulate_histogram(
            summary.source_component_row_size_histogram,
            telemetry.partition_refinement_source_component_row_size_histogram);
        telemetry.partition_refinement_total_hard_gain += summary.hard_gain;
        telemetry.partition_refinement_total_soft_gain += summary.soft_gain;
        telemetry.partition_refinement_total_delta_j += summary.delta_j;
        telemetry.partition_refinement_total_component_delta += summary.component_delta;
        if (summary.improved) {
            ++telemetry.partition_refinement_refine_improved;
        }
    }

    AtomizedCoarseCandidate prepare_folded_family_coarse(
        int feature,
        const PreparedFeatureAtomized &prepared,
        int groups,
        double mu_node,
        const AtomizedCandidate &raw_seed,
        AtomizedObjectiveMode mode
    ) const {
        const bool diagnostics = diagnostics_enabled();
        AtomizedCoarseCandidate coarse;
        if (!raw_seed.feasible || groups < 2 || groups > prepared.q_effective) {
            return coarse;
        }
        coarse.geometry_seed_candidate = raw_seed;
        coarse.candidate = raw_seed;
        if (!prepared.has_block_compression) {
            return coarse;
        }
        if (groups > (int)prepared.block_atoms.size()) {
            return coarse;
        }

        std::vector<int> projected_block_partition;
        std::vector<unsigned char> mixed_block;
        if (!project_bin_partition_to_blocks(
                prepared.blocks,
                prepared.atoms,
                raw_seed.partition,
                projected_block_partition,
                mixed_block)) {
            return coarse;
        }
        if (std::any_of(mixed_block.begin(), mixed_block.end(), [](unsigned char flag) { return flag != 0U; })) {
            return coarse;
        }
        coarse.initial_block_partition = projected_block_partition;

        AtomizedCandidate projected_block_seed = candidate_from_partition(
            feature,
            prepared.block_atoms,
            projected_block_partition,
            raw_seed.groups,
            nullptr,
            0.0,
            mode);
        if (!projected_block_seed.feasible) {
            return coarse;
        }

        AtomizedCandidate lifted_projected_block = lift_block_candidate_to_atoms(
            feature,
            prepared.blocks,
            prepared.atoms,
            projected_block_seed,
            &prepared.atom_adjacency_bonus,
            prepared.atom_adjacency_bonus_total,
            mode);
        if (lifted_projected_block.feasible) {
            coarse.candidate = lifted_projected_block;
        }

        AtomizedRefinementSummary block_summary;
        AtomizedCandidate refined_block = refine_atomized_candidate_partition_locally(
            prepared.block_atoms,
            projected_block_seed,
            mu_node,
            nullptr,
            diagnostics ? &block_summary : nullptr,
            nullptr,
            0.0,
            mode);
        if (diagnostics) {
            record_refinement_summary(block_summary);
        }
        if (!refined_block.feasible) {
            refined_block = projected_block_seed;
        }

        coarse.refined_block_partition = refined_block.partition;
        coarse.block_candidate = lift_block_candidate_to_atoms(
            feature,
            prepared.blocks,
            prepared.atoms,
            refined_block,
            &prepared.atom_adjacency_bonus,
            prepared.atom_adjacency_bonus_total,
            mode);
        if (coarse.block_candidate.feasible &&
            atomized_candidate_better_for_objective(
                coarse.block_candidate,
                coarse.candidate,
                feature,
                feature,
                mode)) {
            coarse.candidate = coarse.block_candidate;
        }
        return coarse;
    }

    bool prepare_feature_atomized_local(
        const std::vector<int> &indices,
        int feature,
        double mu_node,
        PreparedFeatureAtomized &prepared
    ) const {
        prepared = PreparedFeatureAtomized{};
        if (!build_ordered_bins(indices, feature, prepared.bins)) {
            return false;
        }
        if (!build_atomized_bins(
                prepared.bins,
                prepared.atoms,
                &prepared.atom_hard_floor,
                &prepared.atom_imp_floor)) {
            return false;
        }

        const int q_support = std::max(0, (int)indices.size() / std::max(1, min_child_size_));
        prepared.q_effective = std::min(max_groups_for_bins((int)prepared.atoms.size()), q_support);
        if (prepared.q_effective < 2) {
            return false;
        }
        prepared.atom_prefix = build_atomized_prefixes(prepared.atoms);
        prepared.atom_adjacency_bonus.clear();
        prepared.atom_adjacency_bonus_total = 0.0;
        if (prepared.atoms.size() > 1U) {
            prepared.atom_adjacency_bonus.resize(prepared.atoms.size() - 1U, 0.0);
            for (size_t bin_pos = 0; bin_pos + 1U < prepared.atoms.size(); ++bin_pos) {
                const double bonus = contiguous_boundary_bonus(
                    feature,
                    prepared.atoms[bin_pos],
                    prepared.atoms[bin_pos + 1U]);
                prepared.atom_adjacency_bonus[bin_pos] = bonus;
                prepared.atom_adjacency_bonus_total += bonus;
            }
        }

        const AtomizedCompressionRule compression_rule = atomized_compression_rule();
        prepared.has_block_compression = has_pure_same_class_block_compression(prepared.atoms);
        if (prepared.has_block_compression) {
            build_atomized_blocks_and_bins(
                prepared.atoms,
                prepared.blocks,
                prepared.block_atoms,
                compression_rule);
        }
        if (prepared.has_block_compression) {
            prepared.has_block_compression = prepared.block_atoms.size() < prepared.atoms.size();
        }
        if (prepared.has_block_compression) {
            prepared.block_prefix = build_atomized_prefixes(prepared.block_atoms);
        } else {
            prepared.blocks.clear();
            prepared.block_atoms.clear();
            prepared.block_prefix = AtomizedPrefixes{};
        }
        prepared.coarse_by_groups.assign((size_t)prepared.q_effective + 1, AtomizedCoarseCandidate{});
        prepared.coarse_by_groups_hardloss.assign((size_t)prepared.q_effective + 1, AtomizedCoarseCandidate{});

        const std::vector<AtomizedCandidatePair> raw_seed_pairs =
            solve_atomized_geometry_family_pairs(
                prepared.atoms,
                prepared.atom_prefix,
                feature,
                prepared.q_effective);

        bool any_feasible = false;
        for (int groups = 2; groups <= prepared.q_effective; ++groups) {
            const AtomizedCandidatePair &raw_seed_pair = raw_seed_pairs[(size_t)groups];
            AtomizedCandidate impurity_raw_seed = raw_seed_pair.impurity;
            AtomizedCandidate hardloss_raw_seed = raw_seed_pair.misclassification;
            impurity_raw_seed.feature = feature;
            impurity_raw_seed.hard_loss_mode = false;
            hardloss_raw_seed.feature = feature;
            hardloss_raw_seed.hard_loss_mode = true;

            AtomizedCoarseCandidate coarse = prepare_folded_family_coarse(
                feature,
                prepared,
                groups,
                mu_node,
                impurity_raw_seed,
                AtomizedObjectiveMode::kImpurity);
            AtomizedCoarseCandidate hardloss_coarse = prepare_folded_family_coarse(
                feature,
                prepared,
                groups,
                mu_node,
                hardloss_raw_seed,
                AtomizedObjectiveMode::kHardLoss);
            if (!coarse.candidate.feasible && !hardloss_coarse.candidate.feasible) {
                continue;
            }
            prepared.coarse_by_groups[(size_t)groups] = std::move(coarse);
            prepared.coarse_by_groups_hardloss[(size_t)groups] = std::move(hardloss_coarse);
            any_feasible = true;
        }

        prepared.valid = any_feasible;
        return any_feasible;
    }

    AtomizedCandidate nominate_folded_family_candidate(
        const PreparedFeatureAtomized &prepared,
        double mu_node,
        const AtomizedCoarseCandidate &coarse,
        AtomizedCandidate raw_seed,
        AtomizedObjectiveMode mode
    ) const {
        const bool diagnostics = diagnostics_enabled();
        if (!raw_seed.feasible) {
            return raw_seed;
        }
        AtomizedCandidate atom_seed = raw_seed;
        std::vector<std::pair<int, int>> atom_windows;
        const std::vector<std::pair<int, int>> *atom_windows_ptr = nullptr;
        bool used_block_refinement = false;
        if (prepared.has_block_compression &&
            !prepared.block_atoms.empty() &&
            raw_seed.groups <= (int)prepared.block_atoms.size()) {
            std::vector<int> projected_block_partition;
            std::vector<unsigned char> mixed_block;
            if (project_bin_partition_to_blocks(
                    prepared.blocks,
                    prepared.atoms,
                    raw_seed.partition,
                    projected_block_partition,
                    mixed_block)) {
                if (!std::any_of(
                        mixed_block.begin(),
                        mixed_block.end(),
                        [](unsigned char flag) { return flag != 0U; })) {
                    const bool can_reuse_refined =
                        !coarse.initial_block_partition.empty() &&
                        !coarse.refined_block_partition.empty() &&
                        coarse.initial_block_partition.size() == projected_block_partition.size() &&
                        coarse.refined_block_partition.size() == projected_block_partition.size() &&
                        projected_block_partition == coarse.initial_block_partition;
                    AtomizedCandidate projected_block_seed = candidate_from_partition(
                        raw_seed.feature,
                        prepared.block_atoms,
                        projected_block_partition,
                        raw_seed.groups,
                        nullptr,
                        0.0,
                        mode);
                    if (projected_block_seed.feasible) {
                        AtomizedCandidate refined_block = projected_block_seed;
                        if (can_reuse_refined) {
                            refined_block.partition = coarse.refined_block_partition;
                        } else {
                            AtomizedRefinementSummary block_summary;
                            refined_block = refine_atomized_candidate_partition_locally(
                                prepared.block_atoms,
                                projected_block_seed,
                                mu_node,
                                nullptr,
                                diagnostics ? &block_summary : nullptr,
                                nullptr,
                                0.0,
                                mode);
                            if (diagnostics) {
                                record_refinement_summary(block_summary);
                            }
                            if (!refined_block.feasible) {
                                refined_block = projected_block_seed;
                            }
                        }

                        AtomizedCandidate candidate_from_blocks = lift_block_candidate_to_atoms(
                            raw_seed.feature,
                            prepared.blocks,
                            prepared.atoms,
                            refined_block,
                            &prepared.atom_adjacency_bonus,
                            prepared.atom_adjacency_bonus_total,
                            mode);
                        if (!candidate_from_blocks.feasible &&
                            can_reuse_refined &&
                            coarse.block_candidate.feasible) {
                            candidate_from_blocks = coarse.block_candidate;
                        }

                        if (candidate_from_blocks.feasible &&
                            atomized_score_better_for_refinement(
                                candidate_from_blocks.score,
                                atom_seed.score,
                                mu_node,
                                mode)) {
                            used_block_refinement = true;
                            std::vector<std::pair<int, int>> block_windows =
                                build_active_block_windows(
                                    projected_block_partition,
                                    refined_block.partition,
                                    &mixed_block);
                            if (!block_windows.empty()) {
                                atom_windows = block_windows_to_atom_windows(
                                    prepared.blocks,
                                    block_windows);
                                if (!atom_windows.empty()) {
                                    atom_windows_ptr = &atom_windows;
                                }
                            }
                            atom_seed = std::move(candidate_from_blocks);
                        }
                    }
                }
            }
        }

        if (used_block_refinement && atom_windows_ptr == nullptr) {
            return atom_seed;
        }

        AtomizedCandidate best = atom_seed;
        AtomizedRefinementSummary raw_summary;
        const AtomizedCandidate refined = refine_atomized_candidate_partition_locally(
            prepared.atoms,
            atom_seed,
            mu_node,
            atom_windows_ptr,
            diagnostics ? &raw_summary : nullptr,
            &prepared.atom_adjacency_bonus,
            prepared.atom_adjacency_bonus_total,
            mode);
        if (diagnostics) {
            record_refinement_summary(raw_summary);
        }

        if (refined.feasible &&
            atomized_score_better_for_refinement(refined.score, atom_seed.score, mu_node, mode)) {
            best = refined;
        }
        return best;
    }

    #define trace_greedy_snapshot(...) ((void)0)
    #include "msplit_atomized_heuristic.cpp"
    #undef trace_greedy_snapshot
