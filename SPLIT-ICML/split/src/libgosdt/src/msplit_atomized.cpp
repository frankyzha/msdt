    #include "msplit_atomized_support.cpp"

    AtomizedCandidate refine_atomized_candidate_debr(
        const std::vector<AtomizedAtom> &atoms,
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

        const int atom_count = (int)atoms.size();
        const int groups = seed.groups;
        if (atom_count <= 1 || groups <= 1) {
            return seed;
        }
        const bool hard_loss_mode = (mode == AtomizedObjectiveMode::kHardLoss);
        ++local_summary.bridge_policy_calls;
        auto bump_histogram = [](std::vector<long long> &hist, int bucket) {
            if (bucket < 0) {
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
        if (has_active_windows) {
            ++local_summary.refine_windowed_calls;
        } else {
            ++local_summary.refine_unwindowed_calls;
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
            ++local_summary.candidate_legal;
            const bool is_descent = primary_gain > kEpsUpdate;
            RankedMove ranked = make_ranked_move(
                std::move(move),
                primary_gain,
                guide_gain);
            if (is_descent) {
                ++local_summary.candidate_descent_eligible;
                if (!best_descent.valid || ranked_move_better_descent(ranked, best_descent)) {
                    best_descent = std::move(ranked);
                }
                return;
            }

            ++local_summary.candidate_descent_rejected;
            const bool primary_near_zero = std::abs(primary_gain) <= kEpsUpdate;
            if (primary_near_zero) {
                if (active_windows != nullptr && !active_windows->empty()) {
                    ++local_summary.candidate_bridge_window_blocked;
                } else if (bridge_used) {
                    ++local_summary.candidate_bridge_used_blocked;
                } else if (ranked.guide_gain > kEpsUpdate) {
                    ++local_summary.candidate_bridge_eligible;
                    if (!best_bridge.valid || ranked_move_better_bridge(ranked, best_bridge)) {
                        best_bridge = std::move(ranked);
                    }
                    return;
                } else {
                    ++local_summary.candidate_bridge_guide_rejected;
                }
            }
            ++local_summary.candidate_score_rejected;
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
            std::vector<int> assign = seed.assignment;
            std::vector<int> branch_rows((size_t)groups, 0);
            std::vector<double> branch_class_weight(
                static_cast<size_t>(groups) * static_cast<size_t>(n_classes_), 0.0);
            std::vector<double> branch_teacher_class_weight(
                static_cast<size_t>(groups) * static_cast<size_t>(n_classes_), 0.0);
            for (int atom_pos = 0; atom_pos < atom_count; ++atom_pos) {
                const int group_idx = assign[(size_t)atom_pos];
                if (group_idx < 0 || group_idx >= groups) {
                    return seed;
                }
                branch_rows[(size_t)group_idx] += atoms[(size_t)atom_pos].row_count;
                const size_t base = static_cast<size_t>(group_idx) * static_cast<size_t>(n_classes_);
                for (int cls = 0; cls < n_classes_; ++cls) {
                    branch_class_weight[base + static_cast<size_t>(cls)] +=
                        atoms[(size_t)atom_pos].class_weight[(size_t)cls];
                    branch_teacher_class_weight[base + static_cast<size_t>(cls)] +=
                        atoms[(size_t)atom_pos].teacher_class_weight[(size_t)cls];
                }
            }

            std::vector<double> branch_loss((size_t)groups, 0.0);
            std::vector<double> branch_hard_impurity((size_t)groups, 0.0);
            std::vector<double> branch_soft_impurity((size_t)groups, 0.0);
            for (int group_idx = 0; group_idx < groups; ++group_idx) {
                const size_t group_base =
                    static_cast<size_t>(group_idx) * static_cast<size_t>(n_classes_);
                branch_loss[(size_t)group_idx] =
                    split_leaf_loss_flat(branch_class_weight, n_classes_, group_base);
                branch_hard_impurity[(size_t)group_idx] =
                    hard_label_impurity_flat(branch_class_weight, n_classes_, group_base);
                branch_soft_impurity[(size_t)group_idx] =
                    hard_label_impurity_flat(branch_teacher_class_weight, n_classes_, group_base);
            }

            while (true) {
                check_timeout();
                std::vector<unsigned char> seen_source_groups((size_t)groups, 0);

                RankedMove best_descent;
                RankedMove best_bridge;
                int component_start = 0;
                while (component_start < atom_count) {
                    const int source_group = assign[(size_t)component_start];
                    int component_end = component_start + 1;
                    while (component_end < atom_count && assign[(size_t)component_end] == source_group) {
                        ++component_end;
                    }
                    if (!seen_source_groups[(size_t)source_group]) {
                        seen_source_groups[(size_t)source_group] = 1U;
                        bump_histogram(local_summary.source_group_row_size_histogram, branch_rows[(size_t)source_group]);
                    }
                    bump_histogram(
                        local_summary.source_component_atom_size_histogram,
                        component_end - component_start);
                    int component_rows_total = 0;
                    for (int pos = component_start; pos < component_end; ++pos) {
                        component_rows_total += atoms[(size_t)pos].row_count;
                    }
                    bump_histogram(local_summary.source_component_row_size_histogram, component_rows_total);

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
                        const double source_soft_before = branch_soft_impurity[(size_t)source_group];
                        const double target_soft_before = branch_soft_impurity[(size_t)target_group];
                        auto scan_component_overlap = [&](int lo, int hi) {
                            if (lo > hi) {
                                return;
                            }
                            for (int start = lo; start <= hi; ++start) {
                                int move_rows = 0;
                                std::vector<double> move_class_weight((size_t)n_classes_, 0.0);
                                std::vector<double> move_teacher_class_weight((size_t)n_classes_, 0.0);
                                for (int end = start; end <= hi; ++end) {
                                    ++local_summary.candidate_total;
                                    const AtomizedAtom &atom = atoms[(size_t)end];
                                    move_rows += atom.row_count;
                                    for (int cls = 0; cls < n_classes_; ++cls) {
                                        move_class_weight[(size_t)cls] += atom.class_weight[(size_t)cls];
                                        move_teacher_class_weight[(size_t)cls] += atom.teacher_class_weight[(size_t)cls];
                                    }

                                    const int source_rows_after = source_rows_total - move_rows;
                                    if (source_rows_after < min_child_size_) {
                                        ++local_summary.candidate_source_size_rejects;
                                        break;
                                    }
                                    const int target_rows_after = target_rows_total + move_rows;
                                    if (target_rows_after < min_child_size_) {
                                        ++local_summary.candidate_target_size_rejects;
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
                                    const double source_soft_after = hard_label_impurity_flat_delta(
                                        branch_teacher_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_teacher_class_weight,
                                        false);
                                    const double target_soft_after = hard_label_impurity_flat_delta(
                                        branch_teacher_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_teacher_class_weight,
                                        true);
                                    const double primary_before = hard_loss_mode
                                        ? (source_loss_before + target_loss_before)
                                        : (source_hard_before + target_hard_before +
                                           source_soft_before + target_soft_before);
                                    const double primary_after = hard_loss_mode
                                        ? (source_loss_after + target_loss_after)
                                        : (source_hard_after + target_hard_after +
                                           source_soft_after + target_soft_after);
                                    const double guide_before = hard_loss_mode
                                        ? (source_hard_before + target_hard_before)
                                        : (source_loss_before + target_loss_before);
                                    const double guide_after = hard_loss_mode
                                        ? (source_hard_after + target_hard_after)
                                        : (source_loss_after + target_loss_after);
                                    const double delta_primary = primary_before - primary_after;
                                    const double delta_guide = guide_before - guide_after;
                                    const int source_delta_components =
                                        (start == component_start && end == component_end - 1)
                                            ? -1
                                            : ((start == component_start || end == component_end - 1) ? 0 : 1);
                                    const bool left_target =
                                        (start > 0 && assign[(size_t)(start - 1)] == target_group);
                                    const bool right_target =
                                        (end + 1 < atom_count && assign[(size_t)(end + 1)] == target_group);
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
                                    move.teacher_class_weight = move_teacher_class_weight;
                                    move.source_loss_after = source_loss_after;
                                    move.target_loss_after = target_loss_after;
                                    move.source_hard_impurity_after = source_hard_after;
                                    move.target_hard_impurity_after = target_hard_after;
                                    move.source_soft_impurity_after = source_soft_after;
                                move.target_soft_impurity_after = target_soft_after;
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
                            scan_component_overlap(component_start, component_end - 1);
                            saw_overlap_segment = true;
                            ++local_summary.refine_overlap_segments;
                        } else {
                            for (const auto &window : *active_windows) {
                                const int overlap_start = std::max(component_start, window.first);
                                const int overlap_end = std::min(component_end - 1, window.second);
                                if (overlap_start <= overlap_end) {
                                    saw_overlap_segment = true;
                                    ++local_summary.refine_overlap_segments;
                                    scan_component_overlap(overlap_start, overlap_end);
                                }
                            }
                        }
                    }

                component_start = component_end;
            }

            if (saw_overlap_segment) {
                ++local_summary.refine_calls_with_overlap;
            } else {
                ++local_summary.refine_calls_without_overlap;
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

                for (int atom_pos = best_move.start; atom_pos <= best_move.end; ++atom_pos) {
                    assign[(size_t)atom_pos] = best_move.target_group;
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
                    branch_teacher_class_weight[source_base + static_cast<size_t>(cls)] -=
                        best_move.teacher_class_weight[(size_t)cls];
                    branch_teacher_class_weight[target_base + static_cast<size_t>(cls)] +=
                        best_move.teacher_class_weight[(size_t)cls];
                }
                branch_loss[(size_t)best_move.source_group] = best_move.source_loss_after;
                branch_loss[(size_t)best_move.target_group] = best_move.target_loss_after;
                branch_hard_impurity[(size_t)best_move.source_group] = best_move.source_hard_impurity_after;
                branch_hard_impurity[(size_t)best_move.target_group] = best_move.target_hard_impurity_after;
                branch_soft_impurity[(size_t)best_move.source_group] = best_move.source_soft_impurity_after;
                branch_soft_impurity[(size_t)best_move.target_group] = best_move.target_soft_impurity_after;

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

            while (true) {
                check_timeout();
                std::vector<unsigned char> seen_source_groups((size_t)groups, 0);

                RankedMove best_cleanup;
                int component_start = 0;
                while (component_start < atom_count) {
                    const int source_group = assign[(size_t)component_start];
                    int component_end = component_start + 1;
                    while (component_end < atom_count && assign[(size_t)component_end] == source_group) {
                        ++component_end;
                    }
                    if (!seen_source_groups[(size_t)source_group]) {
                        seen_source_groups[(size_t)source_group] = 1U;
                        bump_histogram(local_summary.source_group_row_size_histogram, branch_rows[(size_t)source_group]);
                    }
                    bump_histogram(
                        local_summary.source_component_atom_size_histogram,
                        component_end - component_start);

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
                        const double source_soft_before = branch_soft_impurity[(size_t)source_group];
                        const double target_soft_before = branch_soft_impurity[(size_t)target_group];
                        auto scan_component_overlap = [&](int lo, int hi) {
                            if (lo > hi) {
                                return;
                            }
                            for (int start = lo; start <= hi; ++start) {
                                int move_rows = 0;
                                std::vector<double> move_class_weight((size_t)n_classes_, 0.0);
                                std::vector<double> move_teacher_class_weight((size_t)n_classes_, 0.0);
                                for (int end = start; end <= hi; ++end) {
                                    ++local_summary.candidate_total;
                                    const AtomizedAtom &atom = atoms[(size_t)end];
                                    move_rows += atom.row_count;
                                    for (int cls = 0; cls < n_classes_; ++cls) {
                                        move_class_weight[(size_t)cls] += atom.class_weight[(size_t)cls];
                                        move_teacher_class_weight[(size_t)cls] += atom.teacher_class_weight[(size_t)cls];
                                    }

                                    const int source_rows_after = source_rows_total - move_rows;
                                    if (source_rows_after < min_child_size_) {
                                        ++local_summary.candidate_source_size_rejects;
                                        break;
                                    }
                                    const int target_rows_after = target_rows_total + move_rows;
                                    if (target_rows_after < min_child_size_) {
                                        ++local_summary.candidate_target_size_rejects;
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
                                    const double source_soft_after = hard_label_impurity_flat_delta(
                                        branch_teacher_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_teacher_class_weight,
                                        false);
                                    const double target_soft_after = hard_label_impurity_flat_delta(
                                        branch_teacher_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_teacher_class_weight,
                                        true);
                                    const double primary_gain = hard_loss_mode
                                        ? ((source_loss_before + target_loss_before) -
                                           (source_loss_after + target_loss_after))
                                        : ((source_hard_before + target_hard_before +
                                            source_soft_before + target_soft_before) -
                                           (source_hard_after + target_hard_after +
                                            source_soft_after + target_soft_after));
                                    const double guide_gain = hard_loss_mode
                                        ? ((source_hard_before + target_hard_before +
                                            source_soft_before + target_soft_before) -
                                           (source_hard_after + target_hard_after +
                                            source_soft_after + target_soft_after))
                                        : ((source_loss_before + target_loss_before) -
                                           (source_loss_after + target_loss_after));
                                    const int source_delta_components =
                                        (start == component_start && end == component_end - 1)
                                            ? -1
                                            : ((start == component_start || end == component_end - 1) ? 0 : 1);
                                    const bool left_target =
                                        (start > 0 && assign[(size_t)(start - 1)] == target_group);
                                    const bool right_target =
                                        (end + 1 < atom_count && assign[(size_t)(end + 1)] == target_group);
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
                                    move.teacher_class_weight = move_teacher_class_weight;
                                    move.source_loss_after = source_loss_after;
                                    move.target_loss_after = target_loss_after;
                                    move.source_hard_impurity_after = source_hard_after;
                                    move.target_hard_impurity_after = target_hard_after;
                                    move.source_soft_impurity_after = source_soft_after;
                                    move.target_soft_impurity_after = target_soft_after;
                                    move.delta_hard = primary_gain;
                                    move.delta_soft = guide_gain;
                                    move.delta_j = primary_gain - mu_node * (double)move.delta_components;
                                    const double complexity_gain = -static_cast<double>(move.delta_components);
                                    ++local_summary.candidate_legal;
                                    if (std::abs(primary_gain) > kEpsUpdate) {
                                        ++local_summary.candidate_cleanup_primary_rejected;
                                        ++local_summary.candidate_score_rejected;
                                        continue;
                                    }
                                    if (complexity_gain <= kEpsUpdate) {
                                        ++local_summary.candidate_cleanup_complexity_rejected;
                                        ++local_summary.candidate_score_rejected;
                                        continue;
                                    }
                                    ++local_summary.candidate_cleanup_eligible;
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
                            scan_component_overlap(component_start, component_end - 1);
                        } else {
                            for (const auto &window : *active_windows) {
                                const int overlap_start = std::max(component_start, window.first);
                                const int overlap_end = std::min(component_end - 1, window.second);
                                if (overlap_start <= overlap_end) {
                                    scan_component_overlap(overlap_start, overlap_end);
                                }
                            }
                        }
                    }

                    component_start = component_end;
                }

                if (!best_cleanup.valid) {
                    break;
                }

                const AtomizedRefinementMove &best_cleanup_move = best_cleanup.move;
                for (int atom_pos = best_cleanup_move.start; atom_pos <= best_cleanup_move.end; ++atom_pos) {
                    assign[(size_t)atom_pos] = best_cleanup_move.target_group;
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
                    branch_teacher_class_weight[source_base + static_cast<size_t>(cls)] -=
                        best_cleanup_move.teacher_class_weight[(size_t)cls];
                    branch_teacher_class_weight[target_base + static_cast<size_t>(cls)] +=
                        best_cleanup_move.teacher_class_weight[(size_t)cls];
                }
                branch_loss[(size_t)best_cleanup_move.source_group] = best_cleanup_move.source_loss_after;
                branch_loss[(size_t)best_cleanup_move.target_group] = best_cleanup_move.target_loss_after;
                branch_hard_impurity[(size_t)best_cleanup_move.source_group] = best_cleanup_move.source_hard_impurity_after;
                branch_hard_impurity[(size_t)best_cleanup_move.target_group] = best_cleanup_move.target_hard_impurity_after;
                branch_soft_impurity[(size_t)best_cleanup_move.source_group] = best_cleanup_move.source_soft_impurity_after;
                branch_soft_impurity[(size_t)best_cleanup_move.target_group] = best_cleanup_move.target_soft_impurity_after;

                local_summary.moves += 1;
                ++local_summary.simplify_moves;
                local_summary.hard_gain += best_cleanup.primary_gain;
                local_summary.soft_gain += best_cleanup.guide_gain;
                local_summary.delta_j += best_cleanup_move.delta_j;
                local_summary.component_delta += best_cleanup_move.delta_components;
            }

            if (local_summary.moves == 0) {
                if (summary != nullptr) {
                    *summary = local_summary;
                }
                return seed;
            }

            AtomizedCandidate refined = candidate_from_assignment(
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
            local_summary.improved =
                atomized_score_better_for_refinement(refined.score, seed.score, mu_node, mode);
            if (summary != nullptr) {
                *summary = local_summary;
            }
            return local_summary.improved ? refined : seed;
        }

        std::vector<int> assign = seed.assignment;
        std::vector<int> branch_rows((size_t)groups, 0);
        std::vector<double> branch_pos((size_t)groups, 0.0);
        std::vector<double> branch_neg((size_t)groups, 0.0);
        std::vector<double> branch_teacher_pos((size_t)groups, 0.0);
        std::vector<double> branch_teacher_neg((size_t)groups, 0.0);
        for (int atom_pos = 0; atom_pos < atom_count; ++atom_pos) {
            const int group_idx = assign[(size_t)atom_pos];
            if (group_idx < 0 || group_idx >= groups) {
                return seed;
            }
            const AtomizedAtom &atom = atoms[(size_t)atom_pos];
            branch_rows[(size_t)group_idx] += atom.row_count;
            branch_pos[(size_t)group_idx] += atom.pos_weight;
            branch_neg[(size_t)group_idx] += atom.neg_weight;
            branch_teacher_pos[(size_t)group_idx] += atom.teacher_pos_weight;
            branch_teacher_neg[(size_t)group_idx] += atom.teacher_neg_weight;
        }
        std::vector<double> branch_loss((size_t)groups, 0.0);
        std::vector<double> branch_hard_impurity((size_t)groups, 0.0);
        std::vector<double> branch_soft_impurity((size_t)groups, 0.0);
        for (int group_idx = 0; group_idx < groups; ++group_idx) {
            branch_loss[(size_t)group_idx] = split_leaf_loss(
                branch_pos[(size_t)group_idx],
                branch_neg[(size_t)group_idx]);
            branch_hard_impurity[(size_t)group_idx] = hard_label_impurity(
                branch_pos[(size_t)group_idx],
                branch_neg[(size_t)group_idx]);
            branch_soft_impurity[(size_t)group_idx] = hard_label_impurity(
                branch_teacher_pos[(size_t)group_idx],
                branch_teacher_neg[(size_t)group_idx]);
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
                    branch_secondary[(size_t)group_idx] = hard_label_impurity(
                        branch_pos[(size_t)group_idx],
                        branch_neg[(size_t)group_idx]);
                } else {
                    branch_primary[(size_t)group_idx] = hard_label_impurity(
                        branch_pos[(size_t)group_idx],
                        branch_neg[(size_t)group_idx]);
                    branch_secondary[(size_t)group_idx] = hard_label_impurity(
                        branch_teacher_pos[(size_t)group_idx],
                        branch_teacher_neg[(size_t)group_idx]);
                }
            }

            AtomizedRefinementMove best_move;
            int component_start = 0;
            while (component_start < atom_count) {
                const int source_group = assign[(size_t)component_start];
                int component_end = component_start + 1;
                while (component_end < atom_count && assign[(size_t)component_end] == source_group) {
                    ++component_end;
                }
                if (!seen_source_groups[(size_t)source_group]) {
                    seen_source_groups[(size_t)source_group] = 1U;
                    bump_histogram(local_summary.source_group_row_size_histogram, branch_rows[(size_t)source_group]);
                }
                bump_histogram(
                    local_summary.source_component_atom_size_histogram,
                    component_end - component_start);
                int component_rows_total = 0;
                for (int pos = component_start; pos < component_end; ++pos) {
                    component_rows_total += atoms[(size_t)pos].row_count;
                }
                bump_histogram(local_summary.source_component_row_size_histogram, component_rows_total);

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
                    const double source_teacher_pos_total = branch_teacher_pos[(size_t)source_group];
                    const double source_teacher_neg_total = branch_teacher_neg[(size_t)source_group];
                    const double target_teacher_pos_total = branch_teacher_pos[(size_t)target_group];
                    const double target_teacher_neg_total = branch_teacher_neg[(size_t)target_group];
                    const double source_primary_before = branch_primary[(size_t)source_group];
                    const double target_primary_before = branch_primary[(size_t)target_group];
                    const double source_secondary_before = branch_secondary[(size_t)source_group];
                    const double target_secondary_before = branch_secondary[(size_t)target_group];
                    auto scan_component_overlap = [&](int lo, int hi) {
                        if (lo > hi) {
                            return;
                        }
                            for (int start = lo; start <= hi; ++start) {
                                int move_rows = 0;
                                double move_pos = 0.0;
                                double move_neg = 0.0;
                                double move_teacher_pos = 0.0;
                                double move_teacher_neg = 0.0;
                                for (int end = start; end <= hi; ++end) {
                                    ++local_summary.candidate_total;
                                    const AtomizedAtom &atom = atoms[(size_t)end];
                                    move_rows += atom.row_count;
                                    move_pos += atom.pos_weight;
                                    move_neg += atom.neg_weight;
                                    move_teacher_pos += atom.teacher_pos_weight;
                                move_teacher_neg += atom.teacher_neg_weight;

                                    const int source_rows_after = source_rows_total - move_rows;
                                    if (source_rows_after < min_child_size_) {
                                        ++local_summary.candidate_source_size_rejects;
                                        break;
                                    }
                                    const int target_rows_after = target_rows_total + move_rows;
                                    if (target_rows_after < min_child_size_) {
                                        ++local_summary.candidate_target_size_rejects;
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
                                    source_secondary_after = hard_label_impurity(
                                        source_teacher_pos_total - move_teacher_pos,
                                        source_teacher_neg_total - move_teacher_neg);
                                    target_secondary_after = hard_label_impurity(
                                        target_teacher_pos_total + move_teacher_pos,
                                        target_teacher_neg_total + move_teacher_neg);
                                }
                                const double delta_primary =
                                    (source_primary_before + target_primary_before) -
                                    (source_primary_after + target_primary_after);
                                const double delta_secondary =
                                    (source_secondary_before + target_secondary_before) -
                                    (source_secondary_after + target_secondary_after);
                                const int source_delta_components =
                                    (start == component_start && end == component_end - 1)
                                        ? -1
                                        : ((start == component_start || end == component_end - 1) ? 0 : 1);
                                const bool left_target = (start > 0 && assign[(size_t)(start - 1)] == target_group);
                                const bool right_target = (end + 1 < atom_count && assign[(size_t)(end + 1)] == target_group);
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
                                move.teacher_pos_weight = move_teacher_pos;
                                move.teacher_neg_weight = move_teacher_neg;
                                move.source_loss_after = hard_loss_mode ? source_primary_after : source_secondary_after;
                                move.target_loss_after = hard_loss_mode ? target_primary_after : target_secondary_after;
                                move.source_hard_impurity_after = hard_loss_mode ? source_secondary_after : source_primary_after;
                                    move.target_hard_impurity_after = hard_loss_mode ? target_secondary_after : target_primary_after;
                                    move.source_soft_impurity_after = 0.0;
                                    move.target_soft_impurity_after = 0.0;
                                    move.delta_hard = delta_primary;
                                    move.delta_soft = delta_secondary;
                                    move.delta_j = delta_primary - mu_node * (double)move.delta_components;
                                    ++local_summary.candidate_legal;
                                    if (!atomized_move_is_improving(move)) {
                                        ++local_summary.candidate_score_rejected;
                                        continue;
                                    }
                                    ++local_summary.candidate_descent_eligible;
                                    if (!best_move.valid || atomized_move_better(move, best_move)) {
                                        best_move = std::move(move);
                                    }
                                }
                            }
                        };

                    if (active_windows == nullptr || active_windows->empty()) {
                        scan_component_overlap(component_start, component_end - 1);
                    } else {
                        for (const auto &window : *active_windows) {
                            const int overlap_start = std::max(component_start, window.first);
                            const int overlap_end = std::min(component_end - 1, window.second);
                            if (overlap_start <= overlap_end) {
                                scan_component_overlap(overlap_start, overlap_end);
                            }
                        }
                    }
                }

                component_start = component_end;
            }

            if (!best_move.valid) {
                break;
            }

            for (int atom_pos = best_move.start; atom_pos <= best_move.end; ++atom_pos) {
                assign[(size_t)atom_pos] = best_move.target_group;
            }

            branch_rows[(size_t)best_move.source_group] -= best_move.row_count;
            branch_rows[(size_t)best_move.target_group] += best_move.row_count;
            branch_pos[(size_t)best_move.source_group] -= best_move.pos_weight;
            branch_pos[(size_t)best_move.target_group] += best_move.pos_weight;
            branch_neg[(size_t)best_move.source_group] -= best_move.neg_weight;
            branch_neg[(size_t)best_move.target_group] += best_move.neg_weight;
            branch_teacher_pos[(size_t)best_move.source_group] -= best_move.teacher_pos_weight;
            branch_teacher_pos[(size_t)best_move.target_group] += best_move.teacher_pos_weight;
            branch_teacher_neg[(size_t)best_move.source_group] -= best_move.teacher_neg_weight;
            branch_teacher_neg[(size_t)best_move.target_group] += best_move.teacher_neg_weight;
            branch_loss[(size_t)best_move.source_group] = best_move.source_loss_after;
            branch_loss[(size_t)best_move.target_group] = best_move.target_loss_after;
            branch_hard_impurity[(size_t)best_move.source_group] = best_move.source_hard_impurity_after;
            branch_hard_impurity[(size_t)best_move.target_group] = best_move.target_hard_impurity_after;
            branch_soft_impurity[(size_t)best_move.source_group] = best_move.source_soft_impurity_after;
            branch_soft_impurity[(size_t)best_move.target_group] = best_move.target_soft_impurity_after;

            local_summary.moves += 1;
            local_summary.hard_gain += best_move.delta_j + mu_node * (double)best_move.delta_components;
            local_summary.soft_gain += best_move.delta_soft;
            ++local_summary.descent_moves;
            local_summary.delta_j += best_move.delta_j;
            local_summary.component_delta += best_move.delta_components;
        }

        if (local_summary.moves == 0) {
            if (summary != nullptr) {
                *summary = local_summary;
            }
            return seed;
        }

        AtomizedCandidate refined = candidate_from_assignment(
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
        local_summary.improved = atomized_score_better_for_refinement(refined.score, seed.score, mu_node, mode);
        if (summary != nullptr) {
            *summary = local_summary;
        }
        return local_summary.improved ? refined : seed;
    }

    AtomizedCandidate solve_atomized_geometry_family(
        const std::vector<AtomizedAtom> &atoms,
        const AtomizedPrefixes &prefix,
        int feature,
        int groups,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        AtomizedCandidate out;
        const int m = (int)atoms.size();
        if (groups < 2 || groups > m) {
            return out;
        }

        const int stride = m + 1;
        auto at = [stride](int g, int t) -> size_t {
            return static_cast<size_t>(g) * static_cast<size_t>(stride) + static_cast<size_t>(t);
        };

        std::vector<AtomizedScore> dp(static_cast<size_t>(groups + 1) * static_cast<size_t>(stride));
        std::vector<int> parent(static_cast<size_t>(groups + 1) * static_cast<size_t>(stride), -1);
        dp[at(0, 0)] = AtomizedScore{0.0, 0.0, 0.0, 0.0, 0.0, 0};

        for (int g = 1; g <= groups; ++g) {
            for (int t = g; t <= m; ++t) {
                AtomizedScore best;
                int best_p = -1;
                for (int p = g - 1; p <= t - 1; ++p) {
                    const AtomizedScore &prev = dp[at(g - 1, p)];
                    if (!std::isfinite(prev.hard_impurity)) {
                        continue;
                    }
                    const int row_count = prefix.rows[(size_t)t] - prefix.rows[(size_t)p];
                    if (row_count < min_child_size_) {
                        continue;
                    }
                    AtomizedScore cand = prev;
                    if (binary_mode_) {
                        const double seg_pos = prefix.pos[(size_t)t] - prefix.pos[(size_t)p];
                        const double seg_neg = prefix.neg[(size_t)t] - prefix.neg[(size_t)p];
                        const double seg_teacher_pos = prefix.teacher_pos[(size_t)t] - prefix.teacher_pos[(size_t)p];
                        const double seg_teacher_neg = prefix.teacher_neg[(size_t)t] - prefix.teacher_neg[(size_t)p];
                        cand.hard_loss += split_leaf_loss(seg_pos, seg_neg);
                        cand.soft_loss += split_leaf_loss(seg_teacher_pos, seg_teacher_neg);
                        cand.hard_impurity += hard_label_impurity(seg_pos, seg_neg);
                        cand.soft_impurity += hard_label_impurity(seg_teacher_pos, seg_teacher_neg);
                    } else {
                        cand.hard_loss += split_leaf_loss_prefix_segment(
                            prefix.class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        cand.soft_loss += split_leaf_loss_prefix_segment(
                            prefix.teacher_class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        cand.hard_impurity += hard_label_impurity_prefix_segment(
                            prefix.class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        cand.soft_impurity += hard_label_impurity_prefix_segment(
                            prefix.teacher_class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                    }
                    if (p > 0) {
                        cand.boundary_penalty -= contiguous_boundary_bonus(
                            feature,
                            atoms[(size_t)(p - 1)],
                            atoms[(size_t)p]);
                    }
                    cand.components += 1;
                    if (best_p < 0 || atomized_score_better(cand, best, mode)) {
                        best = cand;
                        best_p = p;
                    }
                }
                if (best_p >= 0) {
                    dp[at(g, t)] = best;
                    parent[at(g, t)] = best_p;
                }
            }
        }

        if (!std::isfinite(atomized_primary_objective(dp[at(groups, m)], mode))) {
            return out;
        }

        std::vector<int> assignment((size_t)m, -1);
        int t = m;
        int g = groups;
        int group_idx = groups - 1;
        while (g > 0) {
            const int p = parent[at(g, t)];
            if (p < 0) {
                return AtomizedCandidate{};
            }
            for (int pos = p; pos < t; ++pos) {
                assignment[(size_t)pos] = group_idx;
            }
            t = p;
            --g;
            --group_idx;
        }

        out.feasible = true;
        out.score = dp[at(groups, m)];
        out.groups = groups;
        out.assignment = std::move(assignment);
        return out;
    }

    AtomizedCandidatePair solve_atomized_geometry_family_pair(
        const std::vector<AtomizedAtom> &atoms,
        const AtomizedPrefixes &prefix,
        int feature,
        int groups
    ) const {
        AtomizedCandidatePair out;
        const int m = (int)atoms.size();
        if (groups < 2 || groups > m) {
            return out;
        }

        const int stride = m + 1;
        auto at = [stride](int g, int t) -> size_t {
            return static_cast<size_t>(g) * static_cast<size_t>(stride) + static_cast<size_t>(t);
        };

        std::vector<AtomizedScore> dp_impurity(
            static_cast<size_t>(groups + 1) * static_cast<size_t>(stride));
        std::vector<AtomizedScore> dp_hardloss(
            static_cast<size_t>(groups + 1) * static_cast<size_t>(stride));
        std::vector<int> parent_impurity(
            static_cast<size_t>(groups + 1) * static_cast<size_t>(stride),
            -1);
        std::vector<int> parent_hardloss(
            static_cast<size_t>(groups + 1) * static_cast<size_t>(stride),
            -1);
        dp_impurity[at(0, 0)] = AtomizedScore{0.0, 0.0, 0.0, 0.0, 0.0, 0};
        dp_hardloss[at(0, 0)] = AtomizedScore{0.0, 0.0, 0.0, 0.0, 0.0, 0};

        for (int g = 1; g <= groups; ++g) {
            for (int t = g; t <= m; ++t) {
                AtomizedScore best_impurity;
                AtomizedScore best_hardloss;
                int best_impurity_p = -1;
                int best_hardloss_p = -1;
                for (int p = g - 1; p <= t - 1; ++p) {
                    const AtomizedScore &prev_impurity = dp_impurity[at(g - 1, p)];
                    const AtomizedScore &prev_hardloss = dp_hardloss[at(g - 1, p)];
                    const bool impurity_prev_feasible =
                        std::isfinite(atomized_primary_objective(
                            prev_impurity,
                            AtomizedObjectiveMode::kImpurity));
                    const bool hardloss_prev_feasible =
                        std::isfinite(atomized_primary_objective(
                            prev_hardloss,
                            AtomizedObjectiveMode::kHardLoss));
                    if (!impurity_prev_feasible && !hardloss_prev_feasible) {
                        continue;
                    }

                    const int row_count = prefix.rows[(size_t)t] - prefix.rows[(size_t)p];
                    if (row_count < min_child_size_) {
                        continue;
                    }

                    double seg_hard_loss = 0.0;
                    double seg_soft_loss = 0.0;
                    double seg_hard_impurity = 0.0;
                    double seg_soft_impurity = 0.0;
                    if (binary_mode_) {
                        const double seg_pos = prefix.pos[(size_t)t] - prefix.pos[(size_t)p];
                        const double seg_neg = prefix.neg[(size_t)t] - prefix.neg[(size_t)p];
                        const double seg_teacher_pos =
                            prefix.teacher_pos[(size_t)t] - prefix.teacher_pos[(size_t)p];
                        const double seg_teacher_neg =
                            prefix.teacher_neg[(size_t)t] - prefix.teacher_neg[(size_t)p];
                        seg_hard_loss = split_leaf_loss(seg_pos, seg_neg);
                        seg_soft_loss = split_leaf_loss(seg_teacher_pos, seg_teacher_neg);
                        seg_hard_impurity = hard_label_impurity(seg_pos, seg_neg);
                        seg_soft_impurity = hard_label_impurity(seg_teacher_pos, seg_teacher_neg);
                    } else {
                        seg_hard_loss = split_leaf_loss_prefix_segment(
                            prefix.class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        seg_soft_loss = split_leaf_loss_prefix_segment(
                            prefix.teacher_class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        seg_hard_impurity = hard_label_impurity_prefix_segment(
                            prefix.class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        seg_soft_impurity = hard_label_impurity_prefix_segment(
                            prefix.teacher_class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                    }
                    const double seg_boundary_penalty =
                        (p > 0)
                            ? -contiguous_boundary_bonus(
                                  feature,
                                  atoms[(size_t)(p - 1)],
                                  atoms[(size_t)p])
                            : 0.0;

                    auto extend = [&](const AtomizedScore &prev) {
                        AtomizedScore cand = prev;
                        cand.hard_loss += seg_hard_loss;
                        cand.soft_loss += seg_soft_loss;
                        cand.hard_impurity += seg_hard_impurity;
                        cand.soft_impurity += seg_soft_impurity;
                        cand.boundary_penalty += seg_boundary_penalty;
                        cand.components += 1;
                        return cand;
                    };

                    if (impurity_prev_feasible) {
                        AtomizedScore cand = extend(prev_impurity);
                        if (best_impurity_p < 0 ||
                            atomized_score_better(
                                cand,
                                best_impurity,
                                AtomizedObjectiveMode::kImpurity)) {
                            best_impurity = cand;
                            best_impurity_p = p;
                        }
                    }
                    if (hardloss_prev_feasible) {
                        AtomizedScore cand = extend(prev_hardloss);
                        if (best_hardloss_p < 0 ||
                            atomized_score_better(
                                cand,
                                best_hardloss,
                                AtomizedObjectiveMode::kHardLoss)) {
                            best_hardloss = cand;
                            best_hardloss_p = p;
                        }
                    }
                }
                if (best_impurity_p >= 0) {
                    dp_impurity[at(g, t)] = best_impurity;
                    parent_impurity[at(g, t)] = best_impurity_p;
                }
                if (best_hardloss_p >= 0) {
                    dp_hardloss[at(g, t)] = best_hardloss;
                    parent_hardloss[at(g, t)] = best_hardloss_p;
                }
            }
        }

        auto build_candidate = [&](const std::vector<AtomizedScore> &dp,
                                   const std::vector<int> &parent,
                                   AtomizedObjectiveMode mode) {
            AtomizedCandidate candidate;
            if (!std::isfinite(atomized_primary_objective(dp[at(groups, m)], mode))) {
                return candidate;
            }
            std::vector<int> assignment((size_t)m, -1);
            int t = m;
            int g = groups;
            int group_idx = groups - 1;
            while (g > 0) {
                const int p = parent[at(g, t)];
                if (p < 0) {
                    return AtomizedCandidate{};
                }
                for (int pos = p; pos < t; ++pos) {
                    assignment[(size_t)pos] = group_idx;
                }
                t = p;
                --g;
                --group_idx;
            }
            candidate.feasible = true;
            candidate.score = dp[at(groups, m)];
            candidate.groups = groups;
            candidate.assignment = std::move(assignment);
            candidate.feature = feature;
            return candidate;
        };

        out.impurity = build_candidate(
            dp_impurity,
            parent_impurity,
            AtomizedObjectiveMode::kImpurity);
        out.misclassification = build_candidate(
            dp_hardloss,
            parent_hardloss,
            AtomizedObjectiveMode::kHardLoss);
        return out;
    }

    void record_refinement_summary(const AtomizedRefinementSummary &summary) const {
        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        auto accumulate_histogram = [](const std::vector<long long> &src, std::vector<long long> &dst) {
            if (src.size() > dst.size()) {
                dst.resize(src.size(), 0);
            }
            for (size_t i = 0; i < src.size(); ++i) {
                dst[i] += src[i];
            }
        };
        ++telemetry.debr_refine_calls;
        telemetry.debr_total_moves += summary.moves;
        telemetry.debr_bridge_policy_calls += summary.bridge_policy_calls;
        telemetry.debr_refine_windowed_calls += summary.refine_windowed_calls;
        telemetry.debr_refine_unwindowed_calls += summary.refine_unwindowed_calls;
        telemetry.debr_refine_overlap_segments += summary.refine_overlap_segments;
        telemetry.debr_refine_calls_with_overlap += summary.refine_calls_with_overlap;
        telemetry.debr_refine_calls_without_overlap += summary.refine_calls_without_overlap;
        telemetry.debr_candidate_total += summary.candidate_total;
        telemetry.debr_candidate_legal += summary.candidate_legal;
        telemetry.debr_candidate_source_size_rejects += summary.candidate_source_size_rejects;
        telemetry.debr_candidate_target_size_rejects += summary.candidate_target_size_rejects;
        telemetry.debr_candidate_descent_eligible += summary.candidate_descent_eligible;
        telemetry.debr_candidate_descent_rejected += summary.candidate_descent_rejected;
        telemetry.debr_candidate_bridge_eligible += summary.candidate_bridge_eligible;
        telemetry.debr_candidate_bridge_window_blocked += summary.candidate_bridge_window_blocked;
        telemetry.debr_candidate_bridge_used_blocked += summary.candidate_bridge_used_blocked;
        telemetry.debr_candidate_bridge_guide_rejected += summary.candidate_bridge_guide_rejected;
        telemetry.debr_candidate_cleanup_eligible += summary.candidate_cleanup_eligible;
        telemetry.debr_candidate_cleanup_primary_rejected += summary.candidate_cleanup_primary_rejected;
        telemetry.debr_candidate_cleanup_complexity_rejected += summary.candidate_cleanup_complexity_rejected;
        telemetry.debr_candidate_score_rejected += summary.candidate_score_rejected;
        telemetry.debr_descent_moves += summary.descent_moves;
        telemetry.debr_bridge_moves += summary.bridge_moves;
        telemetry.debr_simplify_moves += summary.simplify_moves;
        accumulate_histogram(summary.source_group_row_size_histogram, telemetry.debr_source_group_row_size_histogram);
        accumulate_histogram(
            summary.source_component_atom_size_histogram,
            telemetry.debr_source_component_atom_size_histogram);
        accumulate_histogram(
            summary.source_component_row_size_histogram,
            telemetry.debr_source_component_row_size_histogram);
        telemetry.debr_total_hard_gain += summary.hard_gain;
        telemetry.debr_total_soft_gain += summary.soft_gain;
        telemetry.debr_total_delta_j += summary.delta_j;
        telemetry.debr_total_component_delta += summary.component_delta;
        if (summary.improved) {
            ++telemetry.debr_refine_improved;
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

        std::vector<int> projected_block_assignment;
        std::vector<unsigned char> mixed_block;
        if (!project_atom_assignment_to_blocks(
                prepared.blocks,
                prepared.atoms,
                raw_seed.assignment,
                projected_block_assignment,
                mixed_block)) {
            return coarse;
        }
        if (std::any_of(mixed_block.begin(), mixed_block.end(), [](unsigned char flag) { return flag != 0U; })) {
            return coarse;
        }
        coarse.initial_block_assignment = projected_block_assignment;

        AtomizedCandidate projected_block_seed = candidate_from_assignment(
            feature,
            prepared.block_atoms,
            projected_block_assignment,
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
        AtomizedCandidate refined_block = refine_atomized_candidate_debr(
            prepared.block_atoms,
            projected_block_seed,
            mu_node,
            nullptr,
            &block_summary,
            nullptr,
            0.0,
            mode);
        record_refinement_summary(block_summary);
        if (!refined_block.feasible) {
            refined_block = projected_block_seed;
        }

        coarse.refined_block_assignment = refined_block.assignment;
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
        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        prepared = PreparedFeatureAtomized{};
        if (!build_ordered_bins(indices, feature, prepared.bins)) {
            return false;
        }
        if (!build_atomized_atoms(
                prepared.bins,
                prepared.atoms,
                &prepared.atom_hard_floor,
                &prepared.atom_imp_floor)) {
            return false;
        }
        ++telemetry.atomized_features_prepared;
        auto bump_count_histogram = [](std::vector<long long> &hist, size_t bucket) {
            const size_t idx = bucket;
            if (hist.size() <= idx) {
                hist.resize(idx + 1U, 0);
            }
            ++hist[idx];
        };
        bump_count_histogram(telemetry.atomized_feature_atom_count_histogram, prepared.atoms.size());

        const int q_support = std::max(0, (int)indices.size() / std::max(1, min_child_size_));
        prepared.q_effective = std::min(max_groups_for_bins((int)prepared.atoms.size()), q_support);
        bump_count_histogram(
            telemetry.atomized_feature_q_effective_histogram,
            static_cast<size_t>(std::max(0, prepared.q_effective)));
        if (prepared.q_effective < 2) {
            return false;
        }
        prepared.atom_prefix = build_atomized_prefixes(prepared.atoms);
        prepared.atom_adjacency_bonus.clear();
        prepared.atom_adjacency_bonus_total = 0.0;
        if (prepared.atoms.size() > 1U) {
            prepared.atom_adjacency_bonus.resize(prepared.atoms.size() - 1U, 0.0);
            for (size_t atom_pos = 0; atom_pos + 1U < prepared.atoms.size(); ++atom_pos) {
                const double bonus = contiguous_boundary_bonus(
                    feature,
                    prepared.atoms[atom_pos],
                    prepared.atoms[atom_pos + 1U]);
                prepared.atom_adjacency_bonus[atom_pos] = bonus;
                prepared.atom_adjacency_bonus_total += bonus;
            }
        }

        const AtomizedCompressionRule compression_rule = atomized_compression_rule();
        prepared.has_block_compression = has_atomized_block_compression(prepared.atoms, compression_rule);
        if (prepared.has_block_compression) {
            build_atomized_blocks_and_atoms(
                prepared.atoms,
                prepared.blocks,
                prepared.block_atoms,
                compression_rule);
            prepared.has_block_compression = prepared.block_atoms.size() < prepared.atoms.size();
            if (prepared.has_block_compression) {
                const long long atom_count = static_cast<long long>(prepared.atoms.size());
                const long long block_count = static_cast<long long>(prepared.block_atoms.size());
                ++telemetry.atomized_compression_features_applied;
                if (block_count == 1LL) {
                    ++telemetry.atomized_compression_features_collapsed_to_single_block;
                }
                telemetry.atomized_compression_atoms_before_total += atom_count;
                telemetry.atomized_compression_blocks_after_total += block_count;
                telemetry.atomized_compression_atoms_merged_total += atom_count - block_count;
            }
        }
        bump_count_histogram(
            telemetry.atomized_feature_block_atom_count_histogram,
            prepared.has_block_compression && !prepared.block_atoms.empty()
                ? prepared.block_atoms.size()
                : prepared.atoms.size());
        if (prepared.has_block_compression) {
            prepared.block_prefix = build_atomized_prefixes(prepared.block_atoms);
        } else {
            prepared.blocks.clear();
            prepared.block_atoms.clear();
            prepared.block_prefix = AtomizedPrefixes{};
        }
        prepared.coarse_by_groups.assign((size_t)prepared.q_effective + 1, AtomizedCoarseCandidate{});
        prepared.coarse_by_groups_hardloss.assign((size_t)prepared.q_effective + 1, AtomizedCoarseCandidate{});

        bool any_feasible = false;
        for (int groups = 2; groups <= prepared.q_effective; ++groups) {
            const AtomizedCandidatePair raw_seed_pair = solve_atomized_geometry_family_pair(
                prepared.atoms,
                prepared.atom_prefix,
                feature,
                groups);
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
            ++telemetry.atomized_coarse_candidates;
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
            std::vector<int> projected_block_assignment;
            std::vector<unsigned char> mixed_block;
            if (project_atom_assignment_to_blocks(
                    prepared.blocks,
                    prepared.atoms,
                    raw_seed.assignment,
                    projected_block_assignment,
                    mixed_block)) {
                if (!std::any_of(
                        mixed_block.begin(),
                        mixed_block.end(),
                        [](unsigned char flag) { return flag != 0U; })) {
                    const bool can_reuse_refined =
                        !coarse.initial_block_assignment.empty() &&
                        !coarse.refined_block_assignment.empty() &&
                        coarse.initial_block_assignment.size() == projected_block_assignment.size() &&
                        coarse.refined_block_assignment.size() == projected_block_assignment.size() &&
                        projected_block_assignment == coarse.initial_block_assignment;
                    AtomizedCandidate projected_block_seed = candidate_from_assignment(
                        raw_seed.feature,
                        prepared.block_atoms,
                        projected_block_assignment,
                        raw_seed.groups,
                        nullptr,
                        0.0,
                        mode);
                    if (projected_block_seed.feasible) {
                        AtomizedCandidate refined_block = projected_block_seed;
                        if (can_reuse_refined) {
                            refined_block.assignment = coarse.refined_block_assignment;
                        } else {
                            AtomizedRefinementSummary block_summary;
                            refined_block = refine_atomized_candidate_debr(
                                prepared.block_atoms,
                                projected_block_seed,
                                mu_node,
                                nullptr,
                                &block_summary,
                                nullptr,
                                0.0,
                                mode);
                            record_refinement_summary(block_summary);
                            if (!refined_block.feasible) {
                                refined_block = projected_block_seed;
                            }
                        }
                        used_block_refinement = true;
                        std::vector<std::pair<int, int>> block_windows =
                            build_active_block_windows(
                                projected_block_assignment,
                                refined_block.assignment,
                                &mixed_block);
                        if (!block_windows.empty()) {
                            atom_windows = block_windows_to_atom_windows(
                                prepared.blocks,
                                block_windows);
                            if (!atom_windows.empty()) {
                                atom_windows_ptr = &atom_windows;
                            }
                        }
                        AtomizedCandidate lifted_atom_seed = lift_block_candidate_to_atoms(
                            raw_seed.feature,
                            prepared.blocks,
                            prepared.atoms,
                            refined_block,
                            &prepared.atom_adjacency_bonus,
                            prepared.atom_adjacency_bonus_total,
                            mode);
                        if (lifted_atom_seed.feasible) {
                            atom_seed = std::move(lifted_atom_seed);
                        } else if (can_reuse_refined && coarse.block_candidate.feasible) {
                            atom_seed = coarse.block_candidate;
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
        const AtomizedCandidate refined = refine_atomized_candidate_debr(
            prepared.atoms,
            atom_seed,
            mu_node,
            atom_windows_ptr,
            &raw_summary,
            &prepared.atom_adjacency_bonus,
            prepared.atom_adjacency_bonus_total,
            mode);
        record_refinement_summary(raw_summary);

        if (refined.feasible &&
            atomized_score_better_for_refinement(refined.score, atom_seed.score, mu_node, mode)) {
            best = refined;
        }
        return best;
    }

    #define trace_greedy_snapshot(...) ((void)0)
    #include "msplit_atomized_heuristic.cpp"
    #undef trace_greedy_snapshot
