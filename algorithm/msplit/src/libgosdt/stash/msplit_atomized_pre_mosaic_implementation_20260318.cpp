    struct AtomizedAtom {
        int atom_pos = -1;
        int bin_value = -1;
        int row_count = 0;
        double pos_weight = 0.0;
        double neg_weight = 0.0;
        double teacher_pos_weight = 0.0;
        double teacher_neg_weight = 0.0;
        double teacher_prob = 0.5;
    };

    struct AtomizedBlock {
        std::vector<int> atom_positions;
        int row_count = 0;
        double pos_weight = 0.0;
        double neg_weight = 0.0;
        double teacher_pos_weight = 0.0;
        double teacher_neg_weight = 0.0;
    };

    struct AtomizedScore {
        double hard_impurity = kInfinity;
        double soft_impurity = kInfinity;
        int components = std::numeric_limits<int>::max();
    };

    struct AtomizedCandidate {
        bool feasible = false;
        AtomizedScore score;
        std::vector<std::vector<int>> group_atom_positions;
        const char *family = "none";
    };

    struct RankedAtomizedCandidate {
        int feature = -1;
        int groups = -1;
        AtomizedCandidate candidate;
    };

    struct AtomizedRefinementMove {
        bool valid = false;
        int source_group = -1;
        int target_group = -1;
        int start = -1;
        int end = -1;
        int length = 0;
        int delta_components = 0;
        int row_count = 0;
        double pos_weight = 0.0;
        double neg_weight = 0.0;
        double teacher_pos_weight = 0.0;
        double teacher_neg_weight = 0.0;
        double delta_j = 0.0;
        double delta_hard = 0.0;
        double delta_soft = 0.0;
    };

    struct AtomizedRefinementSummary {
        int moves = 0;
        double hard_gain = 0.0;
        double soft_gain = 0.0;
        double delta_j = 0.0;
        int component_delta = 0;
        bool improved = false;
    };

    static nlohmann::json atomized_score_json(const AtomizedScore &score) {
        return nlohmann::json{
            {"hard_impurity", score.hard_impurity},
            {"soft_impurity", score.soft_impurity},
            {"components", score.components},
        };
    }

    static nlohmann::json atomized_candidate_json(const AtomizedCandidate &candidate, int feature, int groups) {
        nlohmann::json out{
            {"feasible", candidate.feasible},
            {"feature", feature},
            {"groups", groups},
            {"family", candidate.family},
        };
        if (candidate.feasible) {
            out["score"] = atomized_score_json(candidate.score);
        }
        return out;
    }

    static bool atomized_score_better(const AtomizedScore &lhs, const AtomizedScore &rhs) {
        if (lhs.hard_impurity < rhs.hard_impurity - kEpsUpdate) {
            return true;
        }
        if (rhs.hard_impurity < lhs.hard_impurity - kEpsUpdate) {
            return false;
        }
        if (lhs.soft_impurity < rhs.soft_impurity - kEpsUpdate) {
            return true;
        }
        if (rhs.soft_impurity < lhs.soft_impurity - kEpsUpdate) {
            return false;
        }
        if (lhs.components != rhs.components) {
            return lhs.components < rhs.components;
        }
        return false;
    }

    static bool atomized_score_better_for_refinement(
        const AtomizedScore &lhs,
        const AtomizedScore &rhs,
        double mu_node
    ) {
        const double lhs_j = lhs.hard_impurity + mu_node * (double)lhs.components;
        const double rhs_j = rhs.hard_impurity + mu_node * (double)rhs.components;
        if (lhs_j < rhs_j - kEpsUpdate) {
            return true;
        }
        if (rhs_j < lhs_j - kEpsUpdate) {
            return false;
        }
        if (lhs.soft_impurity < rhs.soft_impurity - kEpsUpdate) {
            return true;
        }
        if (rhs.soft_impurity < lhs.soft_impurity - kEpsUpdate) {
            return false;
        }
        if (lhs.hard_impurity < rhs.hard_impurity - kEpsUpdate) {
            return true;
        }
        if (rhs.hard_impurity < lhs.hard_impurity - kEpsUpdate) {
            return false;
        }
        if (lhs.components != rhs.components) {
            return lhs.components < rhs.components;
        }
        return false;
    }

    static bool atomized_candidate_better(
        const AtomizedCandidate &lhs,
        const AtomizedCandidate &rhs,
        int lhs_feature,
        int rhs_feature
    ) {
        if (!rhs.feasible) {
            return lhs.feasible;
        }
        if (!lhs.feasible) {
            return false;
        }
        if (atomized_score_better(lhs.score, rhs.score)) {
            return true;
        }
        if (atomized_score_better(rhs.score, lhs.score)) {
            return false;
        }
        return lhs_feature < rhs_feature;
    }

    static bool atomized_move_is_improving(const AtomizedRefinementMove &move) {
        return move.valid &&
            (move.delta_j > kEpsUpdate ||
             (std::abs(move.delta_j) <= kEpsUpdate && move.delta_soft > kEpsUpdate));
    }

    static bool atomized_move_better(
        const AtomizedRefinementMove &lhs,
        const AtomizedRefinementMove &rhs
    ) {
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
    }

    double effective_sample_unit(const std::vector<int> &indices) const {
        double sum_w = 0.0;
        double sum_sq = 0.0;
        for (int idx : indices) {
            const double w = sample_weight_[(size_t)idx];
            sum_w += w;
            sum_sq += w * w;
        }
        if (sum_w <= kEpsUpdate || sum_sq <= kEpsUpdate) {
            return 1.0 / static_cast<double>(std::max(1, (int)indices.size()));
        }
        const double n_eff = (sum_w * sum_w) / sum_sq;
        return 1.0 / std::max(1.0, n_eff);
    }

    bool build_atomized_atoms(const OrderedBins &bins, int feature, std::vector<AtomizedAtom> &atoms) const {
        atoms.clear();
        if (bins.values.size() <= 1U) {
            return false;
        }

        atoms.reserve(bins.values.size());
        for (size_t atom_pos = 0; atom_pos < bins.values.size(); ++atom_pos) {
            AtomizedAtom atom;
            atom.atom_pos = (int)atom_pos;
            atom.bin_value = bins.values[atom_pos];
            atom.row_count = (int)bins.members[atom_pos].size();

            for (int idx : bins.members[atom_pos]) {
                const double w = sample_weight_[(size_t)idx];
                if (y_[(size_t)idx] == 1) {
                    atom.pos_weight += w;
                } else {
                    atom.neg_weight += w;
                }
                const double teacher_prob = teacher_available_ ? teacher_prob_[(size_t)idx] : 0.5;
                atom.teacher_pos_weight += w * teacher_prob;
                atom.teacher_neg_weight += w * (1.0 - teacher_prob);
            }

            if (!teacher_available_) {
                atom.teacher_pos_weight = atom.pos_weight;
                atom.teacher_neg_weight = atom.neg_weight;
            }
            const double teacher_total = atom.teacher_pos_weight + atom.teacher_neg_weight;
            atom.teacher_prob = (teacher_total > kEpsUpdate) ? (atom.teacher_pos_weight / teacher_total) : 0.5;
            atoms.push_back(std::move(atom));
        }
        return atoms.size() > 1U;
    }

    static std::vector<AtomizedBlock> build_atomized_blocks(const std::vector<AtomizedAtom> &atoms) {
        std::vector<AtomizedBlock> blocks;
        if (atoms.empty()) {
            return blocks;
        }

        auto teacher_side_of = [](const AtomizedAtom &atom) { return (atom.teacher_prob >= 0.5) ? 1 : -1; };
        auto empirical_side_of = [](const AtomizedAtom &atom) { return (atom.pos_weight >= atom.neg_weight) ? 1 : -1; };

        AtomizedBlock current;
        for (size_t i = 0; i < atoms.size(); ++i) {
            if (i > 0) {
                const bool same_type =
                    teacher_side_of(atoms[i - 1]) == teacher_side_of(atoms[i]) &&
                    empirical_side_of(atoms[i - 1]) == empirical_side_of(atoms[i]);
                if (!same_type) {
                    blocks.push_back(std::move(current));
                    current = AtomizedBlock{};
                }
            }
            current.atom_positions.push_back(atoms[i].atom_pos);
            current.row_count += atoms[i].row_count;
            current.pos_weight += atoms[i].pos_weight;
            current.neg_weight += atoms[i].neg_weight;
            current.teacher_pos_weight += atoms[i].teacher_pos_weight;
            current.teacher_neg_weight += atoms[i].teacher_neg_weight;
        }

        blocks.push_back(std::move(current));
        return blocks;
    }

    static int count_components_from_sorted_positions(const std::vector<int> &positions) {
        if (positions.empty()) {
            return 0;
        }
        int components = 1;
        for (size_t i = 1; i < positions.size(); ++i) {
            if (positions[i] != positions[i - 1] + 1) {
                ++components;
            }
        }
        return components;
    }

    AtomizedScore score_group_assignment(
        const std::vector<AtomizedAtom> &atoms,
        const std::vector<std::vector<int>> &group_atom_positions
    ) const {
        AtomizedScore score{0.0, 0.0, 0};
        for (const auto &group : group_atom_positions) {
            if (group.empty()) {
                return AtomizedScore{};
            }
            std::vector<int> sorted_positions = group;
            std::sort(sorted_positions.begin(), sorted_positions.end());

            int row_count = 0;
            double pos_weight = 0.0;
            double neg_weight = 0.0;
            double teacher_pos_weight = 0.0;
            double teacher_neg_weight = 0.0;
            for (int atom_pos : sorted_positions) {
                const AtomizedAtom &atom = atoms[(size_t)atom_pos];
                row_count += atom.row_count;
                pos_weight += atom.pos_weight;
                neg_weight += atom.neg_weight;
                teacher_pos_weight += atom.teacher_pos_weight;
                teacher_neg_weight += atom.teacher_neg_weight;
            }
            if (row_count < min_child_size_) {
                return AtomizedScore{};
            }
            score.hard_impurity += hard_label_impurity(pos_weight, neg_weight);
            score.soft_impurity += hard_label_impurity(teacher_pos_weight, teacher_neg_weight);
            score.components += count_components_from_sorted_positions(sorted_positions);
        }
        return score;
    }

    static std::vector<int> assignment_from_groups(
        const std::vector<std::vector<int>> &groups,
        int atom_count
    ) {
        std::vector<int> assign((size_t)atom_count, -1);
        for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
            for (int atom_pos : groups[group_idx]) {
                assign[(size_t)atom_pos] = (int)group_idx;
            }
        }
        return assign;
    }

    static std::vector<std::vector<int>> groups_from_assignment(
        const std::vector<int> &assign,
        int groups
    ) {
        std::vector<std::vector<int>> out((size_t)groups);
        for (int atom_pos = 0; atom_pos < (int)assign.size(); ++atom_pos) {
            const int group_idx = assign[(size_t)atom_pos];
            if (group_idx >= 0 && group_idx < groups) {
                out[(size_t)group_idx].push_back(atom_pos);
            }
        }
        return out;
    }

    AtomizedCandidate candidate_from_assignment(
        const std::vector<AtomizedAtom> &atoms,
        const std::vector<int> &assign,
        int groups,
        const char *family
    ) const {
        AtomizedCandidate out;
        if (groups < 2) {
            return out;
        }
        out.group_atom_positions = groups_from_assignment(assign, groups);
        for (auto &group : out.group_atom_positions) {
            if (group.empty()) {
                return AtomizedCandidate{};
            }
            std::sort(group.begin(), group.end());
        }
        out.score = score_group_assignment(atoms, out.group_atom_positions);
        if (!std::isfinite(out.score.hard_impurity)) {
            return AtomizedCandidate{};
        }
        out.feasible = true;
        out.family = family;
        return out;
    }

    AtomizedCandidate refine_atomized_candidate_partition_locally(
        const std::vector<AtomizedAtom> &atoms,
        const AtomizedCandidate &seed,
        double mu_node,
        AtomizedRefinementSummary *summary = nullptr
    ) const {
        if (!seed.feasible) {
            return seed;
        }

        AtomizedRefinementSummary local_summary;

        const int atom_count = (int)atoms.size();
        const int groups = (int)seed.group_atom_positions.size();
        if (atom_count <= 1 || groups <= 1) {
            return seed;
        }

        std::vector<int> prefix_rows((size_t)atom_count + 1, 0);
        std::vector<double> prefix_pos((size_t)atom_count + 1, 0.0);
        std::vector<double> prefix_neg((size_t)atom_count + 1, 0.0);
        std::vector<double> prefix_teacher_pos((size_t)atom_count + 1, 0.0);
        std::vector<double> prefix_teacher_neg((size_t)atom_count + 1, 0.0);
        for (int i = 0; i < atom_count; ++i) {
            prefix_rows[(size_t)(i + 1)] = prefix_rows[(size_t)i] + atoms[(size_t)i].row_count;
            prefix_pos[(size_t)(i + 1)] = prefix_pos[(size_t)i] + atoms[(size_t)i].pos_weight;
            prefix_neg[(size_t)(i + 1)] = prefix_neg[(size_t)i] + atoms[(size_t)i].neg_weight;
            prefix_teacher_pos[(size_t)(i + 1)] = prefix_teacher_pos[(size_t)i] + atoms[(size_t)i].teacher_pos_weight;
            prefix_teacher_neg[(size_t)(i + 1)] = prefix_teacher_neg[(size_t)i] + atoms[(size_t)i].teacher_neg_weight;
        }

        auto interval_rows = [&](int lo, int hi) {
            return prefix_rows[(size_t)(hi + 1)] - prefix_rows[(size_t)lo];
        };
        auto interval_pos = [&](int lo, int hi) {
            return prefix_pos[(size_t)(hi + 1)] - prefix_pos[(size_t)lo];
        };
        auto interval_neg = [&](int lo, int hi) {
            return prefix_neg[(size_t)(hi + 1)] - prefix_neg[(size_t)lo];
        };
        auto interval_teacher_pos = [&](int lo, int hi) {
            return prefix_teacher_pos[(size_t)(hi + 1)] - prefix_teacher_pos[(size_t)lo];
        };
        auto interval_teacher_neg = [&](int lo, int hi) {
            return prefix_teacher_neg[(size_t)(hi + 1)] - prefix_teacher_neg[(size_t)lo];
        };

        std::vector<int> assign = assignment_from_groups(seed.group_atom_positions, atom_count);
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
        while (true) {
            check_timeout();

            std::vector<double> branch_hard((size_t)groups, 0.0);
            std::vector<double> branch_soft((size_t)groups, 0.0);
            for (int group_idx = 0; group_idx < groups; ++group_idx) {
                branch_hard[(size_t)group_idx] = hard_label_impurity(
                    branch_pos[(size_t)group_idx],
                    branch_neg[(size_t)group_idx]);
                branch_soft[(size_t)group_idx] = hard_label_impurity(
                    branch_teacher_pos[(size_t)group_idx],
                    branch_teacher_neg[(size_t)group_idx]);
            }

            AtomizedRefinementMove best_move;
            int component_start = 0;
            while (component_start < atom_count) {
                const int source_group = assign[(size_t)component_start];
                int component_end = component_start + 1;
                while (component_end < atom_count && assign[(size_t)component_end] == source_group) {
                    ++component_end;
                }

                for (int target_group = 0; target_group < groups; ++target_group) {
                    if (target_group == source_group) {
                        continue;
                    }
                    for (int start = component_start; start < component_end; ++start) {
                        for (int end = start; end < component_end; ++end) {
                            const int move_rows = interval_rows(start, end);
                            const int source_rows_after = branch_rows[(size_t)source_group] - move_rows;
                            if (source_rows_after < min_child_size_) {
                                break;
                            }
                            const int target_rows_after = branch_rows[(size_t)target_group] + move_rows;
                            if (target_rows_after < min_child_size_) {
                                continue;
                            }

                            const double move_pos = interval_pos(start, end);
                            const double move_neg = interval_neg(start, end);
                            const double move_teacher_pos = interval_teacher_pos(start, end);
                            const double move_teacher_neg = interval_teacher_neg(start, end);

                            const double source_hard_after = hard_label_impurity(
                                branch_pos[(size_t)source_group] - move_pos,
                                branch_neg[(size_t)source_group] - move_neg);
                            const double target_hard_after = hard_label_impurity(
                                branch_pos[(size_t)target_group] + move_pos,
                                branch_neg[(size_t)target_group] + move_neg);
                            const double delta_hard =
                                (branch_hard[(size_t)source_group] + branch_hard[(size_t)target_group]) -
                                (source_hard_after + target_hard_after);

                            const double source_soft_after = hard_label_impurity(
                                branch_teacher_pos[(size_t)source_group] - move_teacher_pos,
                                branch_teacher_neg[(size_t)source_group] - move_teacher_neg);
                            const double target_soft_after = hard_label_impurity(
                                branch_teacher_pos[(size_t)target_group] + move_teacher_pos,
                                branch_teacher_neg[(size_t)target_group] + move_teacher_neg);
                            const double delta_soft =
                                (branch_soft[(size_t)source_group] + branch_soft[(size_t)target_group]) -
                                (source_soft_after + target_soft_after);

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
                            move.delta_hard = delta_hard;
                            move.delta_soft = delta_soft;
                            move.delta_j = delta_hard - mu_node * (double)move.delta_components;
                            if (!atomized_move_is_improving(move)) {
                                continue;
                            }
                            if (!best_move.valid || atomized_move_better(move, best_move)) {
                                best_move = move;
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

            local_summary.moves += 1;
            local_summary.hard_gain += best_move.delta_hard;
            local_summary.soft_gain += best_move.delta_soft;
            local_summary.delta_j += best_move.delta_j;
            local_summary.component_delta += best_move.delta_components;
        }

        AtomizedCandidate refined = candidate_from_assignment(atoms, assign, groups, seed.family);
        if (!refined.feasible) {
            return seed;
        }
        local_summary.improved = atomized_score_better_for_refinement(refined.score, seed.score, mu_node);
        if (summary != nullptr) {
            *summary = local_summary;
        }
        return refined;
    }

    AtomizedCandidate solve_atomized_geometry_family(const std::vector<AtomizedAtom> &atoms, int groups) const {
        AtomizedCandidate out;
        const int m = (int)atoms.size();
        if (groups < 2 || groups > m) {
            return out;
        }

        std::vector<int> prefix_rows((size_t)m + 1, 0);
        std::vector<double> prefix_pos((size_t)m + 1, 0.0);
        std::vector<double> prefix_neg((size_t)m + 1, 0.0);
        std::vector<double> prefix_teacher_pos((size_t)m + 1, 0.0);
        std::vector<double> prefix_teacher_neg((size_t)m + 1, 0.0);
        for (int i = 0; i < m; ++i) {
            prefix_rows[(size_t)(i + 1)] = prefix_rows[(size_t)i] + atoms[(size_t)i].row_count;
            prefix_pos[(size_t)(i + 1)] = prefix_pos[(size_t)i] + atoms[(size_t)i].pos_weight;
            prefix_neg[(size_t)(i + 1)] = prefix_neg[(size_t)i] + atoms[(size_t)i].neg_weight;
            prefix_teacher_pos[(size_t)(i + 1)] = prefix_teacher_pos[(size_t)i] + atoms[(size_t)i].teacher_pos_weight;
            prefix_teacher_neg[(size_t)(i + 1)] = prefix_teacher_neg[(size_t)i] + atoms[(size_t)i].teacher_neg_weight;
        }

        std::vector<std::vector<AtomizedScore>> dp((size_t)groups + 1, std::vector<AtomizedScore>((size_t)m + 1));
        std::vector<std::vector<int>> parent((size_t)groups + 1, std::vector<int>((size_t)m + 1, -1));
        for (int g = 0; g <= groups; ++g) {
            for (int t = 0; t <= m; ++t) {
                dp[(size_t)g][(size_t)t] = AtomizedScore{};
            }
        }
        dp[0][0] = AtomizedScore{0.0, 0.0, 0};

        for (int g = 1; g <= groups; ++g) {
            for (int t = g; t <= m; ++t) {
                AtomizedScore best;
                int best_p = -1;
                for (int p = g - 1; p <= t - 1; ++p) {
                    if (!std::isfinite(dp[(size_t)(g - 1)][(size_t)p].hard_impurity)) {
                        continue;
                    }
                    const int row_count = prefix_rows[(size_t)t] - prefix_rows[(size_t)p];
                    if (row_count < min_child_size_) {
                        continue;
                    }
                    AtomizedScore cand = dp[(size_t)(g - 1)][(size_t)p];
                    const double seg_pos = prefix_pos[(size_t)t] - prefix_pos[(size_t)p];
                    const double seg_neg = prefix_neg[(size_t)t] - prefix_neg[(size_t)p];
                    const double seg_teacher_pos = prefix_teacher_pos[(size_t)t] - prefix_teacher_pos[(size_t)p];
                    const double seg_teacher_neg = prefix_teacher_neg[(size_t)t] - prefix_teacher_neg[(size_t)p];
                    cand.hard_impurity += hard_label_impurity(seg_pos, seg_neg);
                    cand.soft_impurity += hard_label_impurity(seg_teacher_pos, seg_teacher_neg);
                    cand.components += 1;
                    if (best_p < 0 || atomized_score_better(cand, best)) {
                        best = cand;
                        best_p = p;
                    }
                }
                if (best_p >= 0) {
                    dp[(size_t)g][(size_t)t] = best;
                    parent[(size_t)g][(size_t)t] = best_p;
                }
            }
        }

        if (!std::isfinite(dp[(size_t)groups][(size_t)m].hard_impurity)) {
            return out;
        }

        std::vector<std::vector<int>> group_positions;
        int t = m;
        int g = groups;
        while (g > 0) {
            const int p = parent[(size_t)g][(size_t)t];
            if (p < 0) {
                return AtomizedCandidate{};
            }
            std::vector<int> group;
            group.reserve((size_t)(t - p));
            for (int pos = p; pos < t; ++pos) {
                group.push_back(pos);
            }
            group_positions.push_back(std::move(group));
            t = p;
            --g;
        }
        std::reverse(group_positions.begin(), group_positions.end());

        out.feasible = true;
        out.score = dp[(size_t)groups][(size_t)m];
        out.group_atom_positions = std::move(group_positions);
        out.family = "geometry";
        return out;
    }

    AtomizedCandidate solve_atomized_block_family(const std::vector<AtomizedAtom> &atoms, int groups) const {
        AtomizedCandidate out;
        const std::vector<AtomizedBlock> blocks = build_atomized_blocks(atoms);
        const int b_count = (int)blocks.size();
        if (groups < 2 || groups > b_count) {
            return out;
        }

        std::vector<int> prefix_rows((size_t)b_count + 1, 0);
        std::vector<double> prefix_pos((size_t)b_count + 1, 0.0);
        std::vector<double> prefix_neg((size_t)b_count + 1, 0.0);
        std::vector<double> prefix_teacher_pos((size_t)b_count + 1, 0.0);
        std::vector<double> prefix_teacher_neg((size_t)b_count + 1, 0.0);
        std::vector<std::vector<int>> block_atom_positions((size_t)b_count);
        for (int i = 0; i < b_count; ++i) {
            const AtomizedBlock &block = blocks[(size_t)i];
            prefix_rows[(size_t)(i + 1)] = prefix_rows[(size_t)i] + block.row_count;
            prefix_pos[(size_t)(i + 1)] = prefix_pos[(size_t)i] + block.pos_weight;
            prefix_neg[(size_t)(i + 1)] = prefix_neg[(size_t)i] + block.neg_weight;
            prefix_teacher_pos[(size_t)(i + 1)] = prefix_teacher_pos[(size_t)i] + block.teacher_pos_weight;
            prefix_teacher_neg[(size_t)(i + 1)] = prefix_teacher_neg[(size_t)i] + block.teacher_neg_weight;
            block_atom_positions[(size_t)i] = block.atom_positions;
        }

        std::vector<std::vector<AtomizedScore>> dp((size_t)groups + 1, std::vector<AtomizedScore>((size_t)b_count + 1));
        std::vector<std::vector<int>> parent((size_t)groups + 1, std::vector<int>((size_t)b_count + 1, -1));
        for (int g = 0; g <= groups; ++g) {
            for (int t = 0; t <= b_count; ++t) {
                dp[(size_t)g][(size_t)t] = AtomizedScore{};
            }
        }
        dp[0][0] = AtomizedScore{0.0, 0.0, 0};

        for (int g = 1; g <= groups; ++g) {
            for (int t = g; t <= b_count; ++t) {
                AtomizedScore best;
                int best_p = -1;
                for (int p = g - 1; p <= t - 1; ++p) {
                    if (!std::isfinite(dp[(size_t)(g - 1)][(size_t)p].hard_impurity)) {
                        continue;
                    }
                    const int row_count = prefix_rows[(size_t)t] - prefix_rows[(size_t)p];
                    if (row_count < min_child_size_) {
                        continue;
                    }
                    AtomizedScore cand = dp[(size_t)(g - 1)][(size_t)p];
                    const double seg_pos = prefix_pos[(size_t)t] - prefix_pos[(size_t)p];
                    const double seg_neg = prefix_neg[(size_t)t] - prefix_neg[(size_t)p];
                    const double seg_teacher_pos = prefix_teacher_pos[(size_t)t] - prefix_teacher_pos[(size_t)p];
                    const double seg_teacher_neg = prefix_teacher_neg[(size_t)t] - prefix_teacher_neg[(size_t)p];
                    cand.hard_impurity += hard_label_impurity(seg_pos, seg_neg);
                    cand.soft_impurity += hard_label_impurity(seg_teacher_pos, seg_teacher_neg);
                    cand.components += 1;
                    if (best_p < 0 || atomized_score_better(cand, best)) {
                        best = cand;
                        best_p = p;
                    }
                }
                if (best_p >= 0) {
                    dp[(size_t)g][(size_t)t] = best;
                    parent[(size_t)g][(size_t)t] = best_p;
                }
            }
        }

        if (!std::isfinite(dp[(size_t)groups][(size_t)b_count].hard_impurity)) {
            return out;
        }

        std::vector<std::vector<int>> group_positions;
        int t = b_count;
        int g = groups;
        while (g > 0) {
            const int p = parent[(size_t)g][(size_t)t];
            if (p < 0) {
                return AtomizedCandidate{};
            }
            std::vector<int> merged_positions;
            for (int block_idx = p; block_idx < t; ++block_idx) {
                merged_positions.insert(
                    merged_positions.end(),
                    block_atom_positions[(size_t)block_idx].begin(),
                    block_atom_positions[(size_t)block_idx].end());
            }
            std::sort(merged_positions.begin(), merged_positions.end());
            merged_positions.erase(std::unique(merged_positions.begin(), merged_positions.end()), merged_positions.end());
            group_positions.push_back(std::move(merged_positions));
            t = p;
            --g;
        }
        std::reverse(group_positions.begin(), group_positions.end());

        out.feasible = true;
        out.score = dp[(size_t)groups][(size_t)b_count];
        out.group_atom_positions = std::move(group_positions);
        out.family = "block";
        return out;
    }

    AtomizedCandidate select_atomized_candidate_for_arity(
        const std::vector<AtomizedAtom> &atoms,
        int groups,
        double mu_node
    ) {
        auto record_refinement = [&](const AtomizedRefinementSummary &summary) {
            ++partition_refinement_refine_calls_;
            partition_refinement_total_moves_ += summary.moves;
            partition_refinement_total_hard_gain_ += summary.hard_gain;
            partition_refinement_total_soft_gain_ += summary.soft_gain;
            partition_refinement_total_delta_j_ += summary.delta_j;
            partition_refinement_total_component_delta_ += summary.component_delta;
            if (summary.improved) {
                ++partition_refinement_refine_improved_;
            }
        };

        AtomizedCandidate geometry = solve_atomized_geometry_family(atoms, groups);
        AtomizedCandidate block = solve_atomized_block_family(atoms, groups);
        if (!geometry.feasible) {
            AtomizedRefinementSummary block_summary;
            AtomizedCandidate refined = refine_atomized_candidate_partition_locally(atoms, block, mu_node, &block_summary);
            record_refinement(block_summary);
            if (refined.feasible) {
                ++partition_refinement_final_teacher_wins_;
            }
            return refined;
        }
        if (!block.feasible) {
            AtomizedRefinementSummary geometry_summary;
            AtomizedCandidate refined = refine_atomized_candidate_partition_locally(atoms, geometry, mu_node, &geometry_summary);
            record_refinement(geometry_summary);
            if (refined.feasible) {
                ++partition_refinement_final_geo_wins_;
            }
            return refined;
        }
        AtomizedRefinementSummary geometry_summary;
        AtomizedRefinementSummary block_summary;
        geometry = refine_atomized_candidate_partition_locally(atoms, geometry, mu_node, &geometry_summary);
        block = refine_atomized_candidate_partition_locally(atoms, block, mu_node, &block_summary);
        record_refinement(geometry_summary);
        record_refinement(block_summary);
        const bool block_wins = atomized_candidate_better(block, geometry, groups, groups);
        if (block_wins) {
            ++partition_refinement_final_teacher_wins_;
        } else {
            ++partition_refinement_final_geo_wins_;
        }
        return block_wins ? block : geometry;
    }

    bool evaluate_feature_atomized_local(
        const std::vector<int> &indices,
        int feature,
        AtomizedCandidate &best_candidate,
        std::vector<AtomizedCandidate> *best_by_groups = nullptr
    ) {
        OrderedBins bins;
        if (!build_ordered_bins(indices, feature, bins)) {
            return false;
        }

        std::vector<AtomizedAtom> atoms;
        if (!build_atomized_atoms(bins, feature, atoms)) {
            return false;
        }

        const int q_support = std::max(0, (int)indices.size() / std::max(1, min_child_size_));
        const int q_effective = std::min(max_groups_for_bins((int)atoms.size()), q_support);
        if (q_effective < 2) {
            return false;
        }

        const double mu_node = effective_sample_unit(indices);
        best_candidate = AtomizedCandidate{};
        if (best_by_groups != nullptr) {
            best_by_groups->assign((size_t)q_effective + 1, AtomizedCandidate{});
        }
        for (int groups = 2; groups <= q_effective; ++groups) {
            AtomizedCandidate candidate = select_atomized_candidate_for_arity(atoms, groups, mu_node);
            if (!candidate.feasible) {
                continue;
            }
            ++greedy_interval_evals_;
            if (best_by_groups != nullptr) {
                (*best_by_groups)[(size_t)groups] = candidate;
            }
            if (!best_candidate.feasible ||
                atomized_candidate_better(candidate, best_candidate, feature, feature)) {
                best_candidate = std::move(candidate);
            }
        }
        return best_candidate.feasible;
    }

    static std::vector<std::pair<int, int>> atom_positions_to_spans(
        const OrderedBins &bins,
        const std::vector<int> &group_positions
    ) {
        std::vector<std::pair<int, int>> spans;
        if (group_positions.empty()) {
            return spans;
        }
        int span_lo = bins.values[(size_t)group_positions.front()];
        int span_hi = span_lo;
        int prev_pos = group_positions.front();
        for (size_t i = 1; i < group_positions.size(); ++i) {
            const int atom_pos = group_positions[i];
            if (atom_pos == prev_pos + 1) {
                span_hi = bins.values[(size_t)atom_pos];
            } else {
                spans.push_back({span_lo, span_hi});
                span_lo = bins.values[(size_t)atom_pos];
                span_hi = span_lo;
            }
            prev_pos = atom_pos;
        }
        spans.push_back({span_lo, span_hi});
        return spans;
    }

    std::shared_ptr<Node> build_internal_node_from_group_spans(
        int feature,
        const std::vector<std::vector<std::pair<int, int>>> &group_spans,
        const std::vector<std::shared_ptr<Node>> &group_nodes,
        int fallback_prediction,
        int n_samples
    ) const {
        if (group_spans.empty() || group_spans.size() != group_nodes.size()) {
            return nullptr;
        }

        auto internal = std::make_shared<Node>();
        internal->is_leaf = false;
        internal->feature = feature;
        internal->fallback_prediction = fallback_prediction;
        internal->n_samples = n_samples;
        internal->group_count = (int)group_nodes.size();

        int largest_child_size = -1;
        for (size_t i = 0; i < group_nodes.size(); ++i) {
            if (!group_nodes[i] || group_spans[i].empty()) {
                return nullptr;
            }
            internal->group_bin_spans.push_back(group_spans[i]);
            internal->group_nodes.push_back(group_nodes[i]);
            if (group_nodes[i]->n_samples > largest_child_size) {
                largest_child_size = group_nodes[i]->n_samples;
                internal->fallback_bin = group_spans[i].front().first;
            }
        }
        if (internal->fallback_bin < 0) {
            internal->fallback_bin = group_spans.front().front().first;
        }
        return internal;
    }

    static bool subtree_is_constant_prediction(const std::shared_ptr<Node> &node, int &prediction_out) {
        if (!node) {
            return false;
        }
        if (node->is_leaf) {
            prediction_out = node->prediction;
            return true;
        }
        int shared_prediction = -1;
        bool first = true;
        for (const auto &child : node->group_nodes) {
            int child_prediction = -1;
            if (!subtree_is_constant_prediction(child, child_prediction)) {
                return false;
            }
            if (first) {
                shared_prediction = child_prediction;
                first = false;
            } else if (child_prediction != shared_prediction) {
                return false;
            }
        }
        if (first) {
            return false;
        }
        prediction_out = shared_prediction;
        return true;
    }

    void maybe_record_arity_diagnostics(
        const std::vector<int> &indices,
        int depth_remaining,
        int winner_feature,
        int winner_groups,
        const AtomizedCandidate &winner_candidate,
        const std::vector<AtomizedCandidate> &best_two_by_feature,
        const std::vector<AtomizedCandidate> &best_three_by_feature
    ) {
        if (!arity_diagnostics_.is_array()) {
            arity_diagnostics_ = nlohmann::json::array();
        }
        if (arity_diagnostics_.size() >= 64) {
            return;
        }

        int best_two_feature = -1;
        AtomizedCandidate best_two;
        for (int feature = 0; feature < (int)best_two_by_feature.size(); ++feature) {
            if (atomized_candidate_better(best_two_by_feature[(size_t)feature], best_two, feature, best_two_feature)) {
                best_two = best_two_by_feature[(size_t)feature];
                best_two_feature = feature;
            }
        }

        int best_three_feature = -1;
        AtomizedCandidate best_three;
        for (int feature = 0; feature < (int)best_three_by_feature.size(); ++feature) {
            if (atomized_candidate_better(best_three_by_feature[(size_t)feature], best_three, feature, best_three_feature)) {
                best_three = best_three_by_feature[(size_t)feature];
                best_three_feature = feature;
            }
        }

        std::string reason = "winner_is_k3";
        if (winner_groups != 3) {
            if (!best_three.feasible) {
                reason = "k3_infeasible_or_support_limited";
            } else if (!best_two.feasible) {
                reason = "k3_beats_missing_k2";
            } else if (best_two.score.hard_impurity < best_three.score.hard_impurity - kEpsUpdate) {
                reason = "k3_lost_hard_impurity";
            } else if (best_three.score.hard_impurity < best_two.score.hard_impurity - kEpsUpdate) {
                reason = "k3_won_hard_but_lost_elsewhere";
            } else if (best_two.score.soft_impurity < best_three.score.soft_impurity - kEpsUpdate) {
                reason = "k3_lost_soft_impurity";
            } else if (best_three.score.soft_impurity < best_two.score.soft_impurity - kEpsUpdate) {
                reason = "k3_won_soft_but_lost_elsewhere";
            } else if (best_two.score.components != best_three.score.components) {
                reason = (best_two.score.components < best_three.score.components)
                    ? "k3_lost_fragmentation"
                    : "k3_won_fragmentation_but_lost_tie";
            } else {
                reason = "k3_lost_feature_tie_break";
            }
        }

        arity_diagnostics_.push_back(nlohmann::json{
            {"node_depth", full_depth_budget_ - depth_remaining + 1},
            {"depth_remaining", depth_remaining},
            {"rows", (int)indices.size()},
            {"kind", "split_node"},
            {"winner", atomized_candidate_json(winner_candidate, winner_feature, winner_groups)},
            {"best_k2", atomized_candidate_json(best_two, best_two_feature, 2)},
            {"best_k3", atomized_candidate_json(best_three, best_three_feature, 3)},
            {"k3_reason", reason},
        });
    }

    void maybe_record_stop_diagnostic(
        const std::vector<int> &indices,
        int depth_remaining,
        const char *reason,
        const SubproblemStats &stats,
        const nlohmann::json &extra = nlohmann::json::object()
    ) {
        if (!arity_diagnostics_.is_array()) {
            arity_diagnostics_ = nlohmann::json::array();
        }
        if (arity_diagnostics_.size() >= 64) {
            return;
        }
        nlohmann::json payload{
            {"kind", "stop_node"},
            {"node_depth", full_depth_budget_ - depth_remaining + 1},
            {"depth_remaining", depth_remaining},
            {"rows", (int)indices.size()},
            {"reason", reason},
            {"pos_weight", stats.pos_weight},
            {"neg_weight", stats.neg_weight},
            {"prediction", stats.prediction},
            {"leaf_objective", stats.leaf_objective},
        };
        for (auto it = extra.begin(); it != extra.end(); ++it) {
            payload[it.key()] = it.value();
        }
        arity_diagnostics_.push_back(std::move(payload));
    }

    GreedyResult greedy_complete(const std::vector<int> &indices, int depth_remaining) {
        check_timeout();
        ++greedy_subproblem_calls_;

        const uint64_t key_hash = state_hash(indices, depth_remaining);
        auto bucket_it = greedy_cache_.find(key_hash);
        if (bucket_it != greedy_cache_.end()) {
            for (const auto &entry : bucket_it->second) {
                if (entry.depth_remaining == depth_remaining && entry.indices == indices) {
                    ++greedy_cache_hits_;
                    return entry.result;
                }
            }
        }

        const SubproblemStats stats = compute_subproblem_stats(indices);
        auto [leaf_objective, leaf_tree] = leaf_solution(stats);
        if (depth_remaining <= 0) {
            maybe_record_stop_diagnostic(indices, depth_remaining, "depth_limit", stats);
            GreedyResult solved{leaf_objective, leaf_tree};
            greedy_cache_[key_hash].push_back(GreedyCacheEntry{depth_remaining, indices, solved});
            ++greedy_cache_states_;
            return solved;
        }
        if (stats.pure) {
            maybe_record_stop_diagnostic(indices, depth_remaining, "pure", stats);
            GreedyResult solved{leaf_objective, leaf_tree};
            greedy_cache_[key_hash].push_back(GreedyCacheEntry{depth_remaining, indices, solved});
            ++greedy_cache_states_;
            return solved;
        }
        if ((int)indices.size() < 2 * min_child_size_) {
            maybe_record_stop_diagnostic(indices, depth_remaining, "too_small_for_split", stats);
            GreedyResult solved{leaf_objective, leaf_tree};
            greedy_cache_[key_hash].push_back(GreedyCacheEntry{depth_remaining, indices, solved});
            ++greedy_cache_states_;
            return solved;
        }

        AtomizedCandidate best_candidate;
        int best_feature = -1;
        int best_groups = -1;
        std::vector<RankedAtomizedCandidate> ranked_candidates;
        std::vector<AtomizedCandidate> best_two_by_feature((size_t)n_features_);
        std::vector<AtomizedCandidate> best_three_by_feature((size_t)n_features_);
        for (int feature = 0; feature < n_features_; ++feature) {
            AtomizedCandidate candidate;
            std::vector<AtomizedCandidate> best_by_groups;
            if (!evaluate_feature_atomized_local(indices, feature, candidate, &best_by_groups)) {
                continue;
            }
            if (best_by_groups.size() > 2) {
                best_two_by_feature[(size_t)feature] = best_by_groups[2];
                if (best_by_groups[2].feasible) {
                    ranked_candidates.push_back(RankedAtomizedCandidate{feature, 2, best_by_groups[2]});
                }
            }
            if (best_by_groups.size() > 3) {
                best_three_by_feature[(size_t)feature] = best_by_groups[3];
                if (best_by_groups[3].feasible) {
                    ranked_candidates.push_back(RankedAtomizedCandidate{feature, 3, best_by_groups[3]});
                }
            }
            for (size_t groups = 4; groups < best_by_groups.size(); ++groups) {
                if (best_by_groups[groups].feasible) {
                    ranked_candidates.push_back(RankedAtomizedCandidate{feature, (int)groups, best_by_groups[groups]});
                }
            }
            if (best_feature < 0 ||
                atomized_candidate_better(candidate, best_candidate, feature, best_feature)) {
                best_candidate = std::move(candidate);
                best_feature = feature;
                best_groups = (int)best_candidate.group_atom_positions.size();
            }
        }

        if (best_feature >= 0 && best_candidate.feasible) {
            maybe_record_arity_diagnostics(
                indices,
                depth_remaining,
                best_feature,
                best_groups,
                best_candidate,
                best_two_by_feature,
                best_three_by_feature);
        }

        std::stable_sort(
            ranked_candidates.begin(),
            ranked_candidates.end(),
            [&](const RankedAtomizedCandidate &lhs, const RankedAtomizedCandidate &rhs) {
                return atomized_candidate_better(
                    lhs.candidate,
                    rhs.candidate,
                    lhs.feature,
                    rhs.feature);
            });

        GreedyResult solved{leaf_objective, leaf_tree};
        double best_one_step_objective = kInfinity;
        bool found_nonconstant_one_step = false;
        bool built_bins_for_best = false;
        bool build_ok_for_best = false;
        for (const auto &ranked : ranked_candidates) {
            OrderedBins bins;
            if (build_ordered_bins(indices, ranked.feature, bins)) {
                if (ranked.feature == best_feature && ranked.groups == best_groups) {
                    built_bins_for_best = true;
                }
                double objective = 0.0;
                std::vector<std::shared_ptr<Node>> child_nodes;
                std::vector<std::vector<std::pair<int, int>>> group_spans;
                child_nodes.reserve(ranked.candidate.group_atom_positions.size());
                group_spans.reserve(ranked.candidate.group_atom_positions.size());
                bool build_ok = true;
                bool has_prediction_flip = false;
                for (const auto &group : ranked.candidate.group_atom_positions) {
                    std::vector<int> subset_sorted;
                    for (int atom_pos : group) {
                        append_sorted_members(subset_sorted, bins.members[(size_t)atom_pos]);
                    }
                    if ((int)subset_sorted.size() < min_child_size_) {
                        build_ok = false;
                        break;
                    }
                    if (depth_remaining == 1) {
                        const SubproblemStats child_stats = compute_subproblem_stats(subset_sorted);
                        auto [child_objective, child_tree] = leaf_solution(child_stats);
                        objective += child_objective;
                        if (child_stats.prediction != stats.prediction) {
                            has_prediction_flip = true;
                        }
                        child_nodes.push_back(child_tree);
                    } else {
                        GreedyResult child = greedy_complete(subset_sorted, depth_remaining - 1);
                        objective += child.objective;
                        child_nodes.push_back(child.tree);
                    }
                    group_spans.push_back(atom_positions_to_spans(bins, group));
                }
                if (build_ok) {
                    std::shared_ptr<Node> tree = build_internal_node_from_group_spans(
                        ranked.feature,
                        group_spans,
                        child_nodes,
                        leaf_tree->prediction,
                        (int)indices.size());
                    if (depth_remaining == 1) {
                        if (tree && has_prediction_flip) {
                            if (!found_nonconstant_one_step || objective + kEpsUpdate < best_one_step_objective) {
                                found_nonconstant_one_step = true;
                                best_one_step_objective = objective;
                                solved = GreedyResult{objective, tree};
                            }
                        }
                        if (ranked.feature == best_feature && ranked.groups == best_groups) {
                            build_ok_for_best = build_ok;
                        }
                        continue;
                    }
                    int constant_prediction = -1;
                    if (tree && !subtree_is_constant_prediction(tree, constant_prediction)) {
                        solved = GreedyResult{objective, tree};
                        break;
                    } else if (tree) {
                        nlohmann::json extra{
                            {"best_feature", ranked.feature},
                            {"best_groups", (int)ranked.candidate.group_atom_positions.size()},
                            {"best_family", ranked.candidate.family},
                            {"best_score", atomized_score_json(ranked.candidate.score)},
                            {"collapsed_constant_prediction", constant_prediction},
                            {"child_predictions", nlohmann::json::array()},
                            {"child_sizes", nlohmann::json::array()},
                        };
                        for (const auto &child : child_nodes) {
                            extra["child_sizes"].push_back(child ? child->n_samples : 0);
                            extra["child_predictions"].push_back(child ? child->fallback_prediction : -1);
                        }
                        maybe_record_stop_diagnostic(indices, depth_remaining, "collapsed_constant_subtree", stats, extra);
                    }
                }
                if (ranked.feature == best_feature && ranked.groups == best_groups) {
                    build_ok_for_best = build_ok;
                }
            }
        }
        if (solved.tree && solved.tree->is_leaf && (arity_diagnostics_.empty() || arity_diagnostics_.back().value("reason", "") != "collapsed_constant_subtree")) {
            nlohmann::json extra{
                {"had_best_candidate", best_candidate.feasible},
                {"best_feature", best_feature},
                {"best_groups", best_candidate.feasible ? (int)best_candidate.group_atom_positions.size() : -1},
                {"best_family", best_candidate.feasible ? best_candidate.family : "none"},
                {"built_bins_for_best", built_bins_for_best},
                {"build_ok_for_best", build_ok_for_best},
            };
            if (best_candidate.feasible) {
                extra["best_score"] = atomized_score_json(best_candidate.score);
            }
            maybe_record_stop_diagnostic(indices, depth_remaining, "no_usable_local_split", stats, extra);
        }

        if (solved.tree && !solved.tree->is_leaf) {
            ++greedy_internal_nodes_;
        }
        greedy_cache_[key_hash].push_back(GreedyCacheEntry{depth_remaining, indices, solved});
        ++greedy_cache_states_;
        return solved;
    }
