    struct AtomizedAtom {
        int atom_pos = -1;
        int bin_value = -1;
        int row_count = 0;
        double pos_weight = 0.0;
        double neg_weight = 0.0;
        double teacher_pos_weight = 0.0;
        double teacher_neg_weight = 0.0;
        double teacher_prob = 0.5;
        int empirical_prediction = 0;
        int teacher_prediction = 0;
        std::vector<double> class_weight;
        std::vector<double> teacher_class_weight;
    };

    struct AtomizedBlock {
        std::vector<int> atom_positions;
        int row_count = 0;
        double pos_weight = 0.0;
        double neg_weight = 0.0;
        double teacher_pos_weight = 0.0;
        double teacher_neg_weight = 0.0;
        int empirical_prediction = 0;
        int teacher_prediction = 0;
        std::vector<double> class_weight;
        std::vector<double> teacher_class_weight;
    };

    struct AtomizedScore {
        double hard_loss = kInfinity;
        double soft_loss = kInfinity;
        double hard_impurity = kInfinity;
        double soft_impurity = kInfinity;
        double boundary_penalty = kInfinity;
        int components = std::numeric_limits<int>::max();
    };

    struct AtomizedCandidate {
        bool feasible = false;
        AtomizedScore score;
        int feature = -1;
        int groups = 0;
        bool hard_loss_mode = false;
        std::vector<int> assignment;
    };

    struct AtomizedCoarseCandidate {
        AtomizedCandidate geometry_seed_candidate;
        AtomizedCandidate block_candidate;
        AtomizedCandidate candidate;
        std::vector<int> initial_block_assignment;
        std::vector<int> refined_block_assignment;
    };

    struct AtomizedCandidatePair {
        AtomizedCandidate impurity;
        AtomizedCandidate misclassification;
    };

    struct AtomizedPrefixes {
        std::vector<int> rows;
        std::vector<double> pos;
        std::vector<double> neg;
        std::vector<double> teacher_pos;
        std::vector<double> teacher_neg;
        std::vector<double> class_weight_prefix;
        std::vector<double> teacher_class_weight_prefix;
    };

    struct PreparedFeatureAtomized {
        bool valid = false;
        bool has_block_compression = false;
        OrderedBins bins;
        std::vector<AtomizedAtom> atoms;
        double atom_hard_floor = 0.0;
        double atom_imp_floor = 0.0;
        AtomizedPrefixes atom_prefix;
        std::vector<double> atom_adjacency_bonus;
        double atom_adjacency_bonus_total = 0.0;
        std::vector<AtomizedBlock> blocks;
        std::vector<AtomizedAtom> block_atoms;
        AtomizedPrefixes block_prefix;
        std::vector<AtomizedCoarseCandidate> coarse_by_groups;
        std::vector<AtomizedCoarseCandidate> coarse_by_groups_hardloss;
        int q_effective = 0;
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
        double source_loss_after = 0.0;
        double target_loss_after = 0.0;
        double source_hard_impurity_after = 0.0;
        double target_hard_impurity_after = 0.0;
        double source_soft_impurity_after = 0.0;
        double target_soft_impurity_after = 0.0;
        std::vector<double> class_weight;
        std::vector<double> teacher_class_weight;
    };

    struct AtomizedRefinementSummary {
        int moves = 0;
        int bridge_policy_calls = 0;
        int refine_windowed_calls = 0;
        int refine_unwindowed_calls = 0;
        int refine_overlap_segments = 0;
        int refine_calls_with_overlap = 0;
        int refine_calls_without_overlap = 0;
        int candidate_total = 0;
        int candidate_legal = 0;
        int candidate_source_size_rejects = 0;
        int candidate_target_size_rejects = 0;
        int candidate_descent_eligible = 0;
        int candidate_descent_rejected = 0;
        int candidate_bridge_eligible = 0;
        int candidate_bridge_window_blocked = 0;
        int candidate_bridge_used_blocked = 0;
        int candidate_bridge_guide_rejected = 0;
        int candidate_cleanup_eligible = 0;
        int candidate_cleanup_primary_rejected = 0;
        int candidate_cleanup_complexity_rejected = 0;
        int candidate_score_rejected = 0;
        int descent_moves = 0;
        int bridge_moves = 0;
        int simplify_moves = 0;
        std::vector<long long> source_group_row_size_histogram;
        std::vector<long long> source_component_atom_size_histogram;
        std::vector<long long> source_component_row_size_histogram;
        double hard_gain = 0.0;
        double soft_gain = 0.0;
        double delta_j = 0.0;
        int component_delta = 0;
        bool improved = false;
    };

    static constexpr double kBoundaryPenaltyWeight = 1.0;

    enum class AtomizedObjectiveMode {
        kImpurity,
        kHardLoss
    };

    enum class AtomizedCompressionRule {
        kCurrent,
        kPureSameClass,
        kProportionalProfile,
        kBic,
        kConfidenceOverlap,
        kPlateau,
        kNone
    };

    AtomizedCompressionRule atomized_compression_rule() const {
        const char *raw = std::getenv("MSPLIT_ATOM_COMPRESSION_RULE");
        if (raw == nullptr || raw[0] == '\0') {
            return AtomizedCompressionRule::kPureSameClass;
        }
        const std::string value(raw);
        if (value == "current") {
            return AtomizedCompressionRule::kCurrent;
        }
        if (value == "pure" || value == "pure_same_class") {
            return AtomizedCompressionRule::kPureSameClass;
        }
        if (value == "proportional" || value == "proportional_profile") {
            return AtomizedCompressionRule::kProportionalProfile;
        }
        if (value == "bic" || value == "mdl" || value == "bic_mdl") {
            return AtomizedCompressionRule::kBic;
        }
        if (value == "confidence" || value == "confidence_overlap") {
            return AtomizedCompressionRule::kConfidenceOverlap;
        }
        if (value == "plateau" || value == "plateau_denoise" || value == "global_plateau") {
            return AtomizedCompressionRule::kPlateau;
        }
        if (value == "none" || value == "raw") {
            return AtomizedCompressionRule::kNone;
        }
        return AtomizedCompressionRule::kCurrent;
    }

    static bool atom_is_empirically_pure(const AtomizedAtom &atom) {
        if (!atom.class_weight.empty()) {
            double total = 0.0;
            double best = 0.0;
            for (double value : atom.class_weight) {
                total += value;
                best = std::max(best, value);
            }
            return total <= kEpsUpdate || (total - best) <= kEpsUpdate;
        }
        return std::min(atom.pos_weight, atom.neg_weight) <= kEpsUpdate;
    }

    static bool normalized_profile_equal(
        const std::vector<double> &lhs,
        const std::vector<double> &rhs,
        double lhs_total,
        double rhs_total
    ) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        if (lhs_total <= kEpsUpdate || rhs_total <= kEpsUpdate) {
            return false;
        }
        constexpr double kProfileTol = 1e-10;
        for (size_t idx = 0; idx < lhs.size(); ++idx) {
            const double lhs_norm = lhs[idx] / lhs_total;
            const double rhs_norm = rhs[idx] / rhs_total;
            if (std::fabs(lhs_norm - rhs_norm) > kProfileTol) {
                return false;
            }
        }
        return true;
    }

    static bool proportional_binary_profile_equal(
        double lhs_pos,
        double lhs_neg,
        double rhs_pos,
        double rhs_neg
    ) {
        const double lhs_total = lhs_pos + lhs_neg;
        const double rhs_total = rhs_pos + rhs_neg;
        if (lhs_total <= kEpsUpdate || rhs_total <= kEpsUpdate) {
            return false;
        }
        constexpr double kProfileTol = 1e-10;
        const double cross_delta = std::fabs(lhs_pos * rhs_total - rhs_pos * lhs_total);
        const double scale = std::max({1.0, std::fabs(lhs_pos * rhs_total), std::fabs(rhs_pos * lhs_total)});
        return cross_delta <= kProfileTol * scale;
    }

    static std::pair<double, double> atomized_wilson_interval(double successes, double trials) {
        if (trials <= kEpsUpdate) {
            return {0.0, 1.0};
        }
        const double clamped_successes = std::max(0.0, std::min(successes, trials));
        const double p = clamped_successes / trials;
        constexpr double kConfidenceZ = 1.959963984540054;
        const double z2 = kConfidenceZ * kConfidenceZ;
        const double denom = 1.0 + z2 / trials;
        const double center = p + z2 / (2.0 * trials);
        const double variance = (p * (1.0 - p) + z2 / (4.0 * trials)) / trials;
        const double margin = kConfidenceZ * std::sqrt(std::max(0.0, variance));
        const double lower = std::max(0.0, (center - margin) / denom);
        const double upper = std::min(1.0, (center + margin) / denom);
        return {lower, upper};
    }

    static bool atomized_intervals_overlap(
        const std::pair<double, double> &lhs,
        const std::pair<double, double> &rhs
    ) {
        return std::max(lhs.first, rhs.first) <= std::min(lhs.second, rhs.second) + kEpsUpdate;
    }

    template <typename AtomLike>
    static bool atomized_confidence_overlap(const AtomLike &left, const AtomLike &right) {
        if (left.row_count <= 0 || right.row_count <= 0) {
            return false;
        }
        const bool lhs_multiclass = !left.class_weight.empty();
        const bool rhs_multiclass = !right.class_weight.empty();
        if (lhs_multiclass != rhs_multiclass) {
            return false;
        }
        if (!lhs_multiclass) {
            const double lhs_total = left.pos_weight + left.neg_weight;
            const double rhs_total = right.pos_weight + right.neg_weight;
            if (lhs_total <= kEpsUpdate || rhs_total <= kEpsUpdate) {
                return false;
            }
            const double lhs_success = (left.pos_weight / lhs_total) * static_cast<double>(left.row_count);
            const double rhs_success = (right.pos_weight / rhs_total) * static_cast<double>(right.row_count);
            return atomized_intervals_overlap(
                atomized_wilson_interval(lhs_success, static_cast<double>(left.row_count)),
                atomized_wilson_interval(rhs_success, static_cast<double>(right.row_count)));
        }

        if (left.class_weight.size() != right.class_weight.size()) {
            return false;
        }
        double lhs_total = 0.0;
        double rhs_total = 0.0;
        for (double value : left.class_weight) {
            lhs_total += value;
        }
        for (double value : right.class_weight) {
            rhs_total += value;
        }
        if (lhs_total <= kEpsUpdate || rhs_total <= kEpsUpdate) {
            return false;
        }
        for (size_t cls = 0; cls < left.class_weight.size(); ++cls) {
            const double lhs_success =
                (left.class_weight[cls] / lhs_total) * static_cast<double>(left.row_count);
            const double rhs_success =
                (right.class_weight[cls] / rhs_total) * static_cast<double>(right.row_count);
            if (!atomized_intervals_overlap(
                    atomized_wilson_interval(lhs_success, static_cast<double>(left.row_count)),
                    atomized_wilson_interval(rhs_success, static_cast<double>(right.row_count)))) {
                return false;
            }
        }
        return true;
    }

    template <typename AtomLike>
    static double atomized_total_variation_distance(const AtomLike &left, const AtomLike &right) {
        const bool lhs_multiclass = !left.class_weight.empty();
        const bool rhs_multiclass = !right.class_weight.empty();
        if (lhs_multiclass != rhs_multiclass) {
            return kInfinity;
        }
        if (!lhs_multiclass) {
            const double lhs_total = left.pos_weight + left.neg_weight;
            const double rhs_total = right.pos_weight + right.neg_weight;
            if (lhs_total <= kEpsUpdate || rhs_total <= kEpsUpdate) {
                return kInfinity;
            }
            const double lhs_prob = left.pos_weight / lhs_total;
            const double rhs_prob = right.pos_weight / rhs_total;
            return std::fabs(lhs_prob - rhs_prob);
        }

        if (left.class_weight.size() != right.class_weight.size()) {
            return kInfinity;
        }
        double lhs_total = 0.0;
        double rhs_total = 0.0;
        for (double value : left.class_weight) {
            lhs_total += value;
        }
        for (double value : right.class_weight) {
            rhs_total += value;
        }
        if (lhs_total <= kEpsUpdate || rhs_total <= kEpsUpdate) {
            return kInfinity;
        }
        double tv = 0.0;
        for (size_t cls = 0; cls < left.class_weight.size(); ++cls) {
            const double lhs_prob = left.class_weight[cls] / lhs_total;
            const double rhs_prob = right.class_weight[cls] / rhs_total;
            tv += std::fabs(lhs_prob - rhs_prob);
        }
        return 0.5 * tv;
    }

    static bool atomized_atoms_should_merge(
        const AtomizedAtom &left,
        const AtomizedAtom &right,
        AtomizedCompressionRule rule
    ) {
        switch (rule) {
            case AtomizedCompressionRule::kCurrent:
                return left.teacher_prediction == right.teacher_prediction &&
                    left.empirical_prediction == right.empirical_prediction;
            case AtomizedCompressionRule::kPureSameClass:
                return atom_is_empirically_pure(left) &&
                    atom_is_empirically_pure(right) &&
                    left.empirical_prediction == right.empirical_prediction;
            case AtomizedCompressionRule::kProportionalProfile:
                if (!left.class_weight.empty() && !right.class_weight.empty()) {
                    double lhs_total = 0.0;
                    double rhs_total = 0.0;
                    for (double value : left.class_weight) {
                        lhs_total += value;
                    }
                    for (double value : right.class_weight) {
                        rhs_total += value;
                    }
                    double lhs_teacher_total = 0.0;
                    double rhs_teacher_total = 0.0;
                    for (double value : left.teacher_class_weight) {
                        lhs_teacher_total += value;
                    }
                    for (double value : right.teacher_class_weight) {
                        rhs_teacher_total += value;
                    }
                    return normalized_profile_equal(
                               left.class_weight,
                               right.class_weight,
                               lhs_total,
                               rhs_total) &&
                        normalized_profile_equal(
                               left.teacher_class_weight,
                               right.teacher_class_weight,
                               lhs_teacher_total,
                               rhs_teacher_total);
                }
                return proportional_binary_profile_equal(
                           left.pos_weight,
                           left.neg_weight,
                           right.pos_weight,
                           right.neg_weight) &&
                    proportional_binary_profile_equal(
                           left.teacher_pos_weight,
                           left.teacher_neg_weight,
                           right.teacher_pos_weight,
                           right.teacher_neg_weight);
            case AtomizedCompressionRule::kConfidenceOverlap:
                return atomized_confidence_overlap(left, right);
            case AtomizedCompressionRule::kBic:
            case AtomizedCompressionRule::kPlateau:
            case AtomizedCompressionRule::kNone:
                return false;
        }
        return false;
    }

    static double atomized_binary_neg_log_likelihood(double pos, double neg) {
        const double total = pos + neg;
        if (total <= kEpsUpdate) {
            return 0.0;
        }
        double out = total * std::log(total);
        if (pos > kEpsUpdate) {
            out -= pos * std::log(pos);
        }
        if (neg > kEpsUpdate) {
            out -= neg * std::log(neg);
        }
        return out;
    }

    static double atomized_categorical_neg_log_likelihood(const std::vector<double> &counts) {
        double total = 0.0;
        for (double value : counts) {
            total += value;
        }
        if (total <= kEpsUpdate) {
            return 0.0;
        }
        double out = total * std::log(total);
        for (double value : counts) {
            if (value > kEpsUpdate) {
                out -= value * std::log(value);
            }
        }
        return out;
    }

    AtomizedBlock atomized_block_from_atom(const AtomizedAtom &atom) const {
        AtomizedBlock block;
        block.atom_positions.push_back(atom.atom_pos);
        block.row_count = atom.row_count;
        block.pos_weight = atom.pos_weight;
        block.neg_weight = atom.neg_weight;
        block.teacher_pos_weight = atom.teacher_pos_weight;
        block.teacher_neg_weight = atom.teacher_neg_weight;
        block.empirical_prediction = atom.empirical_prediction;
        block.teacher_prediction = atom.teacher_prediction;
        block.class_weight = atom.class_weight;
        block.teacher_class_weight = atom.teacher_class_weight;
        return block;
    }

    void atomized_refresh_block_predictions(AtomizedBlock &block) const {
        if (binary_mode_) {
            block.empirical_prediction = (block.pos_weight >= block.neg_weight) ? 1 : 0;
            block.teacher_prediction = (block.teacher_pos_weight >= block.teacher_neg_weight) ? 1 : 0;
        } else {
            block.empirical_prediction = argmax_index(block.class_weight);
            block.teacher_prediction = argmax_index(block.teacher_class_weight);
        }
    }

    AtomizedBlock atomized_merge_blocks(const AtomizedBlock &left, const AtomizedBlock &right) const {
        AtomizedBlock merged;
        merged.atom_positions.reserve(left.atom_positions.size() + right.atom_positions.size());
        merged.atom_positions.insert(
            merged.atom_positions.end(),
            left.atom_positions.begin(),
            left.atom_positions.end());
        merged.atom_positions.insert(
            merged.atom_positions.end(),
            right.atom_positions.begin(),
            right.atom_positions.end());
        merged.row_count = left.row_count + right.row_count;
        merged.pos_weight = left.pos_weight + right.pos_weight;
        merged.neg_weight = left.neg_weight + right.neg_weight;
        merged.teacher_pos_weight = left.teacher_pos_weight + right.teacher_pos_weight;
        merged.teacher_neg_weight = left.teacher_neg_weight + right.teacher_neg_weight;
        if (!left.class_weight.empty() || !right.class_weight.empty()) {
            const size_t width = std::max(left.class_weight.size(), right.class_weight.size());
            merged.class_weight.assign(width, 0.0);
            merged.teacher_class_weight.assign(width, 0.0);
            for (size_t cls = 0; cls < width; ++cls) {
                const double left_class = (cls < left.class_weight.size()) ? left.class_weight[cls] : 0.0;
                const double right_class = (cls < right.class_weight.size()) ? right.class_weight[cls] : 0.0;
                const double left_teacher =
                    (cls < left.teacher_class_weight.size()) ? left.teacher_class_weight[cls] : 0.0;
                const double right_teacher =
                    (cls < right.teacher_class_weight.size()) ? right.teacher_class_weight[cls] : 0.0;
                merged.class_weight[cls] = left_class + right_class;
                merged.teacher_class_weight[cls] = left_teacher + right_teacher;
            }
        }
        atomized_refresh_block_predictions(merged);
        return merged;
    }

    double atomized_block_neg_log_likelihood(const AtomizedBlock &block) const {
        const double count_scale = static_cast<double>(std::max(1, n_rows_));
        if (binary_mode_) {
            return count_scale * atomized_binary_neg_log_likelihood(block.pos_weight, block.neg_weight);
        }
        return count_scale * atomized_categorical_neg_log_likelihood(block.class_weight);
    }

    int atomized_distribution_parameter_count() const {
        return binary_mode_ ? 1 : std::max(1, n_classes_ - 1);
    }

    bool atomized_bic_merge_preferred(const AtomizedBlock &left, const AtomizedBlock &right) const {
        const AtomizedBlock merged = atomized_merge_blocks(left, right);
        const double complexity_weight = std::log(std::max(2, merged.row_count));
        const double split_bic =
            2.0 * (atomized_block_neg_log_likelihood(left) + atomized_block_neg_log_likelihood(right)) +
            2.0 * atomized_distribution_parameter_count() * complexity_weight;
        const double merged_bic =
            2.0 * atomized_block_neg_log_likelihood(merged) +
            atomized_distribution_parameter_count() * complexity_weight;
        return merged_bic <= split_bic;
    }

    double atomized_joint_impurity(const AtomizedScore &score) const {
        return score.hard_impurity + score.soft_impurity;
    }

    double atomized_primary_objective(
        const AtomizedScore &score,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        if (mode == AtomizedObjectiveMode::kHardLoss) {
            return score.hard_loss;
        }
        return score.hard_impurity + score.soft_impurity;
    }

    double atomized_candidate_primary_objective(const AtomizedCandidate &candidate) const {
        return candidate.hard_loss_mode
            ? candidate.score.hard_loss
            : atomized_primary_objective(candidate.score, AtomizedObjectiveMode::kImpurity);
    }

    double atomized_candidate_excess_components(const AtomizedCandidate &candidate) const {
        return std::max(0.0, (double)(candidate.score.components - candidate.groups));
    }

    double atomized_candidate_bad_boundary(const AtomizedCandidate &candidate) const {
        return candidate.score.boundary_penalty;
    }

    double atomized_score_bad_boundary(const AtomizedScore &score) const {
        return score.boundary_penalty;
    }

    bool atomized_score_better(
        const AtomizedScore &lhs,
        const AtomizedScore &rhs,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        const double lhs_primary = atomized_primary_objective(lhs, mode);
        const double rhs_primary = atomized_primary_objective(rhs, mode);
        if (lhs_primary < rhs_primary - kEpsUpdate) {
            return true;
        }
        if (rhs_primary < lhs_primary - kEpsUpdate) {
            return false;
        }
        if (lhs.components != rhs.components) {
            return lhs.components < rhs.components;
        }
        const double lhs_boundary = atomized_score_bad_boundary(lhs);
        const double rhs_boundary = atomized_score_bad_boundary(rhs);
        if (lhs_boundary < rhs_boundary - kEpsUpdate) {
            return true;
        }
        if (rhs_boundary < lhs_boundary - kEpsUpdate) {
            return false;
        }
        return false;
    }

    double atomized_score_proxy(
        const AtomizedScore &score,
        double mu_node,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        return atomized_primary_objective(score, mode) +
            mu_node * (kBoundaryPenaltyWeight * score.boundary_penalty + (double)score.components);
    }

    bool atomized_score_better_for_refinement(
        const AtomizedScore &lhs,
        const AtomizedScore &rhs,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        const double lhs_primary = atomized_primary_objective(lhs, mode);
        const double rhs_primary = atomized_primary_objective(rhs, mode);
        if (lhs_primary < rhs_primary - kEpsUpdate) {
            return true;
        }
        if (rhs_primary < lhs_primary - kEpsUpdate) {
            return false;
        }
        if (lhs.components != rhs.components) {
            return lhs.components < rhs.components;
        }
        const double lhs_boundary = atomized_score_bad_boundary(lhs);
        const double rhs_boundary = atomized_score_bad_boundary(rhs);
        if (lhs_boundary < rhs_boundary - kEpsUpdate) {
            return true;
        }
        if (rhs_boundary < lhs_boundary - kEpsUpdate) {
            return false;
        }
        return false;
    }

    bool atomized_score_better_for_refinement(
        const AtomizedScore &lhs,
        const AtomizedScore &rhs,
        double mu_node,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        (void)mu_node;
        return atomized_score_better_for_refinement(lhs, rhs, mode);
    }

    bool atomized_candidate_better_for_objective(
        const AtomizedCandidate &lhs,
        const AtomizedCandidate &rhs,
        int lhs_feature,
        int rhs_feature
    ) const {
        if (!rhs.feasible) {
            return lhs.feasible;
        }
        if (!lhs.feasible) {
            return false;
        }
        const double lhs_primary = atomized_candidate_primary_objective(lhs);
        const double rhs_primary = atomized_candidate_primary_objective(rhs);
        if (lhs_primary < rhs_primary - kEpsUpdate) {
            return true;
        }
        if (rhs_primary < lhs_primary - kEpsUpdate) {
            return false;
        }
        const double lhs_excess = std::max(0.0, (double)(lhs.score.components - lhs.groups));
        const double rhs_excess = std::max(0.0, (double)(rhs.score.components - rhs.groups));
        if (lhs_excess < rhs_excess - kEpsUpdate) {
            return true;
        }
        if (rhs_excess < lhs_excess - kEpsUpdate) {
            return false;
        }
        const double lhs_boundary = atomized_candidate_bad_boundary(lhs);
        const double rhs_boundary = atomized_candidate_bad_boundary(rhs);
        if (lhs_boundary < rhs_boundary - kEpsUpdate) {
            return true;
        }
        if (rhs_boundary < lhs_boundary - kEpsUpdate) {
            return false;
        }
        if (lhs.groups != rhs.groups) {
            return lhs.groups < rhs.groups;
        }
        if (lhs_feature != rhs_feature) {
            return lhs_feature < rhs_feature;
        }
        if (lhs.assignment != rhs.assignment) {
            return lhs.assignment < rhs.assignment;
        }
        return false;
    }

    bool atomized_candidate_better_for_objective(
        const AtomizedCandidate &lhs,
        const AtomizedCandidate &rhs,
        int lhs_feature,
        int rhs_feature,
        AtomizedObjectiveMode mode
    ) const {
        (void)mode;
        return atomized_candidate_better_for_objective(lhs, rhs, lhs_feature, rhs_feature);
    }

    bool atomized_candidate_dominates(
        const AtomizedCandidate &lhs,
        const AtomizedCandidate &rhs
    ) const {
        if (!lhs.feasible || !rhs.feasible) {
            return false;
        }
        const bool loss_not_worse = lhs.score.hard_loss <= rhs.score.hard_loss + kEpsUpdate;
        const bool impurity_not_worse =
            lhs.score.hard_impurity <= rhs.score.hard_impurity + kEpsUpdate;
        const bool loss_better = lhs.score.hard_loss < rhs.score.hard_loss - kEpsUpdate;
        const bool impurity_better =
            lhs.score.hard_impurity < rhs.score.hard_impurity - kEpsUpdate;
        return loss_not_worse && impurity_not_worse && (loss_better || impurity_better);
    }

    bool atomized_candidate_better_global(
        const AtomizedCandidate &lhs,
        const AtomizedCandidate &rhs,
        int lhs_feature,
        int rhs_feature
    ) const {
        if (!rhs.feasible) {
            return lhs.feasible;
        }
        if (!lhs.feasible) {
            return false;
        }
        const double lhs_primary = atomized_candidate_primary_objective(lhs);
        const double rhs_primary = atomized_candidate_primary_objective(rhs);
        if (lhs_primary < rhs_primary - kEpsUpdate) {
            return true;
        }
        if (rhs_primary < lhs_primary - kEpsUpdate) {
            return false;
        }
        const double lhs_excess = atomized_candidate_excess_components(lhs);
        const double rhs_excess = atomized_candidate_excess_components(rhs);
        if (lhs_excess < rhs_excess - kEpsUpdate) {
            return true;
        }
        if (rhs_excess < lhs_excess - kEpsUpdate) {
            return false;
        }
        const double lhs_boundary = atomized_candidate_bad_boundary(lhs);
        const double rhs_boundary = atomized_candidate_bad_boundary(rhs);
        if (lhs_boundary < rhs_boundary - kEpsUpdate) {
            return true;
        }
        if (rhs_boundary < lhs_boundary - kEpsUpdate) {
            return false;
        }
        if (lhs.groups != rhs.groups) {
            return lhs.groups < rhs.groups;
        }
        if (lhs_feature != rhs_feature) {
            return lhs_feature < rhs_feature;
        }
        if (lhs.assignment != rhs.assignment) {
            return lhs.assignment < rhs.assignment;
        }
        return false;
    }

    static bool atomized_assignment_equivalent(
        const AtomizedCandidate &lhs,
        const AtomizedCandidate &rhs
    ) {
        if (!lhs.feasible || !rhs.feasible) {
            return false;
        }
        if (lhs.assignment.size() != rhs.assignment.size()) {
            return false;
        }
        std::vector<int> lhs_to_rhs;
        std::vector<int> rhs_to_lhs;
        int lhs_groups = 0;
        int rhs_groups = 0;
        for (size_t i = 0; i < lhs.assignment.size(); ++i) {
            lhs_groups = std::max(lhs_groups, lhs.assignment[i] + 1);
            rhs_groups = std::max(rhs_groups, rhs.assignment[i] + 1);
        }
        lhs_to_rhs.assign((size_t)lhs_groups, -1);
        rhs_to_lhs.assign((size_t)rhs_groups, -1);
        for (size_t i = 0; i < lhs.assignment.size(); ++i) {
            const int lhs_group = lhs.assignment[i];
            const int rhs_group = rhs.assignment[i];
            if (lhs_group < 0 || rhs_group < 0) {
                return false;
            }
            int &mapped_rhs = lhs_to_rhs[(size_t)lhs_group];
            int &mapped_lhs = rhs_to_lhs[(size_t)rhs_group];
            if (mapped_rhs < 0 && mapped_lhs < 0) {
                mapped_rhs = rhs_group;
                mapped_lhs = lhs_group;
                continue;
            }
            if (mapped_rhs != rhs_group || mapped_lhs != lhs_group) {
                return false;
            }
        }
        return true;
    }

    void record_family_compare_stats(
        const AtomizedCandidate &impurity,
        const AtomizedCandidate &misclassification
    ) const {
        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        ++telemetry.family_compare_total;
        telemetry.family1_hard_loss_sum += impurity.score.hard_loss;
        telemetry.family2_hard_loss_sum += misclassification.score.hard_loss;
        telemetry.family_hard_loss_delta_sum +=
            (misclassification.score.hard_loss - impurity.score.hard_loss);
        telemetry.family1_hard_impurity_sum += impurity.score.hard_impurity;
        telemetry.family2_hard_impurity_sum += misclassification.score.hard_impurity;
        telemetry.family_hard_impurity_delta_sum +=
            (misclassification.score.hard_impurity - impurity.score.hard_impurity);
        telemetry.family1_soft_impurity_sum += impurity.score.soft_impurity;
        telemetry.family2_soft_impurity_sum += misclassification.score.soft_impurity;
        telemetry.family_soft_impurity_delta_sum +=
            (misclassification.score.soft_impurity - impurity.score.soft_impurity);
        telemetry.family1_joint_impurity_sum += atomized_joint_impurity(impurity.score);
        telemetry.family2_joint_impurity_sum += atomized_joint_impurity(misclassification.score);
        telemetry.family_joint_impurity_delta_sum +=
            (atomized_joint_impurity(misclassification.score) - atomized_joint_impurity(impurity.score));

        const bool family2_loss_better =
            misclassification.score.hard_loss < impurity.score.hard_loss - kEpsUpdate;
        const bool family1_loss_better =
            impurity.score.hard_loss < misclassification.score.hard_loss - kEpsUpdate;
        const bool family2_hard_impurity_better =
            misclassification.score.hard_impurity < impurity.score.hard_impurity - kEpsUpdate;
        const bool family1_hard_impurity_better =
            impurity.score.hard_impurity < misclassification.score.hard_impurity - kEpsUpdate;
        const double impurity_metric_1 = atomized_joint_impurity(impurity.score);
        const double impurity_metric_2 = atomized_joint_impurity(misclassification.score);
        const bool family2_joint_better = impurity_metric_2 < impurity_metric_1 - kEpsUpdate;
        const bool family1_joint_better = impurity_metric_1 < impurity_metric_2 - kEpsUpdate;
        if (!family2_loss_better && !family1_loss_better) {
            ++telemetry.family_hard_loss_ties;
        }
        if (!family2_hard_impurity_better && !family1_hard_impurity_better) {
            ++telemetry.family_hard_impurity_ties;
        }
        if (!family2_joint_better && !family1_joint_better) {
            ++telemetry.family_joint_impurity_ties;
        }
        if (family2_loss_better) {
            ++telemetry.family2_hard_loss_wins;
        }
        if (family2_hard_impurity_better) {
            ++telemetry.family2_hard_impurity_wins;
        }
        if (family2_joint_better) {
            ++telemetry.family2_joint_impurity_wins;
        }
        if (family2_loss_better && family2_joint_better) {
            ++telemetry.family2_both_wins;
        }
        if (family1_loss_better && family1_joint_better) {
            ++telemetry.family1_both_wins;
        }
        if (!family2_loss_better && !family1_loss_better &&
            !family2_joint_better && !family1_joint_better) {
            ++telemetry.family_neither_both_wins;
        }
    }

    std::vector<AtomizedCandidate> select_family_nominees(
        AtomizedCandidate impurity,
        AtomizedCandidate misclassification
    ) const {
        std::vector<AtomizedCandidate> selected;
        selected.reserve(2);
        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        if (!impurity.feasible && !misclassification.feasible) {
            return selected;
        }
        if (!impurity.feasible) {
            ++telemetry.atomized_coarse_pruned_candidates;
            ++telemetry.debr_final_block_wins;
            selected.push_back(std::move(misclassification));
            return selected;
        }
        if (!misclassification.feasible) {
            ++telemetry.atomized_coarse_pruned_candidates;
            ++telemetry.debr_final_geo_wins;
            selected.push_back(std::move(impurity));
            return selected;
        }

        record_family_compare_stats(impurity, misclassification);
        if (atomized_assignment_equivalent(impurity, misclassification)) {
            ++telemetry.family_compare_equivalent;
            ++telemetry.family1_selected_by_equivalence;
            ++telemetry.debr_final_geo_wins;
            ++telemetry.atomized_coarse_pruned_candidates;
            selected.push_back(std::move(impurity));
            return selected;
        }
        ++telemetry.family_sent_both;
        selected.push_back(std::move(impurity));
        selected.push_back(std::move(misclassification));
        return selected;
    }

    double effective_sample_unit(const SubproblemStats &stats) const {
        if (sample_weight_uniform_) {
            return 1.0 / static_cast<double>(std::max(1, stats.total_count));
        }
        if (stats.sum_weight <= kEpsUpdate || stats.sum_weight_sq <= kEpsUpdate) {
            return 1.0 / static_cast<double>(std::max(1, stats.total_count));
        }
        const double n_eff = (stats.sum_weight * stats.sum_weight) / stats.sum_weight_sq;
        return 1.0 / std::max(1.0, n_eff);
    }

    double noncontiguous_boundary_penalty(
        int feature,
        const AtomizedAtom &left,
        const AtomizedAtom &right
    ) const {
        const int gap_width = right.bin_value - left.bin_value;
        if (gap_width <= 0) {
            return 0.0;
        }
        const double strength =
            boundary_strength_between_bins(feature, left.bin_value, right.bin_value);
        return strength / static_cast<double>(gap_width);
    }

    double contiguous_boundary_bonus(
        int feature,
        const AtomizedAtom &left,
        const AtomizedAtom &right
    ) const {
        if (feature < 0) {
            return 0.0;
        }
        return noncontiguous_boundary_penalty(feature, left, right);
    }

    bool build_atomized_atoms(
        const OrderedBins &bins,
        std::vector<AtomizedAtom> &atoms,
        double *hard_floor_out = nullptr,
        double *imp_floor_out = nullptr
    ) const {
        atoms.clear();
        if (bins.values.size() <= 1U) {
            return false;
        }

        atoms.reserve(bins.values.size());
        double hard_floor = 0.0;
        double imp_floor = 0.0;
        for (size_t atom_pos = 0; atom_pos < bins.values.size(); ++atom_pos) {
            AtomizedAtom atom;
            atom.atom_pos = (int)atom_pos;
            atom.bin_value = bins.values[atom_pos];
            atom.row_count = (int)bins.members[atom_pos].size();
            if (!binary_mode_) {
                atom.class_weight.assign((size_t)n_classes_, 0.0);
                atom.teacher_class_weight.assign((size_t)n_classes_, 0.0);
            }

            for (int idx : bins.members[atom_pos]) {
                const double w = sample_weight_[(size_t)idx];
                const int label = y_[(size_t)idx];
                if (binary_mode_ && label == 1) {
                    atom.pos_weight += w;
                } else if (binary_mode_) {
                    atom.neg_weight += w;
                } else {
                    atom.class_weight[(size_t)label] += w;
                }
                if (binary_mode_) {
                    const double teacher_prob = teacher_prob_[(size_t)idx];
                    atom.teacher_pos_weight += w * teacher_prob;
                    atom.teacher_neg_weight += w * (1.0 - teacher_prob);
                } else {
                    const size_t teacher_base = static_cast<size_t>(idx) * static_cast<size_t>(n_classes_);
                    for (int cls = 0; cls < n_classes_; ++cls) {
                        atom.teacher_class_weight[(size_t)cls] +=
                            w * teacher_prob_flat_[teacher_base + static_cast<size_t>(cls)];
                    }
                }
            }

            if (binary_mode_) {
                const double teacher_total = atom.teacher_pos_weight + atom.teacher_neg_weight;
                atom.teacher_prob = (teacher_total > kEpsUpdate) ? (atom.teacher_pos_weight / teacher_total) : 0.5;
                atom.empirical_prediction = (atom.pos_weight >= atom.neg_weight) ? 1 : 0;
                atom.teacher_prediction = (atom.teacher_prob >= 0.5) ? 1 : 0;
            } else {
                atom.empirical_prediction = argmax_index(atom.class_weight);
                atom.teacher_prediction = argmax_index(atom.teacher_class_weight);
            }

            if (binary_mode_) {
                const double total = atom.pos_weight + atom.neg_weight;
                if (total > kEpsUpdate) {
                    hard_floor += total - std::max(atom.pos_weight, atom.neg_weight);
                    const double teacher_total = atom.teacher_pos_weight + atom.teacher_neg_weight;
                    if (teacher_total > kEpsUpdate) {
                        imp_floor += teacher_total - std::max(atom.teacher_pos_weight, atom.teacher_neg_weight);
                    } else {
                        imp_floor += total - std::max(atom.pos_weight, atom.neg_weight);
                    }
                }
            } else {
                double total = 0.0;
                double best = 0.0;
                for (double value : atom.class_weight) {
                    total += value;
                    best = std::max(best, value);
                }
                if (total > kEpsUpdate) {
                    hard_floor += total - best;
                }
                double teacher_total = 0.0;
                double teacher_best = 0.0;
                for (double value : atom.teacher_class_weight) {
                    teacher_total += value;
                    teacher_best = std::max(teacher_best, value);
                }
                if (teacher_total > kEpsUpdate) {
                    imp_floor += teacher_total - teacher_best;
                } else if (total > kEpsUpdate) {
                    imp_floor += total - best;
                }
            }
            atoms.push_back(std::move(atom));
        }
        if (hard_floor_out != nullptr) {
            *hard_floor_out = hard_floor;
        }
        if (imp_floor_out != nullptr) {
            *imp_floor_out = imp_floor;
        }
        return atoms.size() > 1U;
    }

    static void append_block_atom(
        const AtomizedBlock &block,
        int block_idx,
        std::vector<AtomizedAtom> &block_atoms
    ) {
        AtomizedAtom atom;
        atom.atom_pos = block_idx;
        atom.bin_value = block_idx;
        atom.row_count = block.row_count;
        atom.pos_weight = block.pos_weight;
        atom.neg_weight = block.neg_weight;
        atom.teacher_pos_weight = block.teacher_pos_weight;
        atom.teacher_neg_weight = block.teacher_neg_weight;
        atom.empirical_prediction = block.empirical_prediction;
        atom.teacher_prediction = block.teacher_prediction;
        atom.class_weight = block.class_weight;
        atom.teacher_class_weight = block.teacher_class_weight;
        if (!atom.class_weight.empty()) {
            atom.teacher_prob = 0.5;
        } else {
            const double teacher_total = atom.teacher_pos_weight + atom.teacher_neg_weight;
            atom.teacher_prob = (teacher_total > kEpsUpdate) ? (atom.teacher_pos_weight / teacher_total) : 0.5;
        }
        block_atoms.push_back(std::move(atom));
    }

    bool has_atomized_block_compression(
        const std::vector<AtomizedAtom> &atoms,
        AtomizedCompressionRule rule = AtomizedCompressionRule::kCurrent
    ) const {
        if (rule == AtomizedCompressionRule::kConfidenceOverlap) {
            if (atoms.size() <= 1U) {
                return false;
            }
            AtomizedBlock current = atomized_block_from_atom(atoms.front());
            for (size_t i = 1; i < atoms.size(); ++i) {
                const AtomizedBlock next = atomized_block_from_atom(atoms[i]);
                if (atomized_confidence_overlap(current, next)) {
                    return true;
                }
                current = next;
            }
            return false;
        }
        if (rule == AtomizedCompressionRule::kPlateau) {
            if (atoms.size() <= 2U) {
                return false;
            }
            for (size_t i = 1; i < atoms.size(); ++i) {
                if (atomized_confidence_overlap(
                        atomized_block_from_atom(atoms[i - 1]),
                        atomized_block_from_atom(atoms[i]))) {
                    return true;
                }
            }
            return false;
        }
        if (rule == AtomizedCompressionRule::kBic) {
            // BIC compression is only meant to coarsen over-segmented features,
            // not delete the only boundary of a binary feature.
            if (atoms.size() <= 2U) {
                return false;
            }
            for (size_t i = 1; i < atoms.size(); ++i) {
                if (atomized_bic_merge_preferred(
                        atomized_block_from_atom(atoms[i - 1]),
                        atomized_block_from_atom(atoms[i]))) {
                    return true;
                }
            }
            return false;
        }
        for (size_t i = 1; i < atoms.size(); ++i) {
            if (atomized_atoms_should_merge(atoms[i - 1], atoms[i], rule)) {
                return true;
            }
        }
        return false;
    }

    void build_atomized_blocks_and_atoms(
        const std::vector<AtomizedAtom> &atoms,
        std::vector<AtomizedBlock> &blocks,
        std::vector<AtomizedAtom> &block_atoms,
        AtomizedCompressionRule rule = AtomizedCompressionRule::kCurrent
    ) const {
        blocks.clear();
        block_atoms.clear();
        if (atoms.empty()) {
            return;
        }

        if (rule == AtomizedCompressionRule::kConfidenceOverlap) {
            blocks.reserve(atoms.size());
            block_atoms.reserve(atoms.size());
            AtomizedBlock current = atomized_block_from_atom(atoms.front());
            for (size_t i = 1; i < atoms.size(); ++i) {
                const AtomizedBlock next = atomized_block_from_atom(atoms[i]);
                if (atomized_confidence_overlap(current, next)) {
                    current = atomized_merge_blocks(current, next);
                    continue;
                }
                append_block_atom(current, static_cast<int>(blocks.size()), block_atoms);
                blocks.push_back(std::move(current));
                current = next;
            }
            append_block_atom(current, static_cast<int>(blocks.size()), block_atoms);
            blocks.push_back(std::move(current));
            return;
        }

        if (rule == AtomizedCompressionRule::kPlateau) {
            std::vector<AtomizedBlock> active_blocks;
            active_blocks.reserve(atoms.size());
            for (const AtomizedAtom &atom : atoms) {
                active_blocks.push_back(atomized_block_from_atom(atom));
            }
            while (active_blocks.size() > 2U) {
                int best_idx = -1;
                double best_score = kInfinity;
                for (size_t i = 1; i < active_blocks.size(); ++i) {
                    if (!atomized_confidence_overlap(active_blocks[i - 1], active_blocks[i])) {
                        continue;
                    }
                    const double score =
                        atomized_total_variation_distance(active_blocks[i - 1], active_blocks[i]);
                    if (score < best_score - kEpsUpdate) {
                        best_idx = static_cast<int>(i - 1);
                        best_score = score;
                    }
                }
                if (best_idx < 0) {
                    break;
                }
                active_blocks[static_cast<size_t>(best_idx)] = atomized_merge_blocks(
                    active_blocks[static_cast<size_t>(best_idx)],
                    active_blocks[static_cast<size_t>(best_idx) + 1U]);
                active_blocks.erase(active_blocks.begin() + best_idx + 1);
            }

            blocks = std::move(active_blocks);
            block_atoms.reserve(blocks.size());
            for (size_t i = 0; i < blocks.size(); ++i) {
                append_block_atom(blocks[i], static_cast<int>(i), block_atoms);
            }
            return;
        }

        if (rule == AtomizedCompressionRule::kBic) {
            std::vector<AtomizedBlock> active_blocks;
            active_blocks.reserve(atoms.size());
            for (const AtomizedAtom &atom : atoms) {
                active_blocks.push_back(atomized_block_from_atom(atom));
            }
            while (active_blocks.size() > 2U) {
                int best_idx = -1;
                double best_gain = 0.0;
                for (size_t i = 1; i < active_blocks.size(); ++i) {
                    const AtomizedBlock merged = atomized_merge_blocks(active_blocks[i - 1], active_blocks[i]);
                    const double complexity_weight = std::log(std::max(2, merged.row_count));
                    const double split_bic =
                        2.0 * (atomized_block_neg_log_likelihood(active_blocks[i - 1]) +
                               atomized_block_neg_log_likelihood(active_blocks[i])) +
                        2.0 * atomized_distribution_parameter_count() * complexity_weight;
                    const double merged_bic =
                        2.0 * atomized_block_neg_log_likelihood(merged) +
                        atomized_distribution_parameter_count() * complexity_weight;
                    const double gain = split_bic - merged_bic;
                    if (gain >= 0.0 && (best_idx < 0 || gain > best_gain)) {
                        best_idx = static_cast<int>(i - 1);
                        best_gain = gain;
                    }
                }
                if (best_idx < 0) {
                    break;
                }
                active_blocks[(size_t)best_idx] = atomized_merge_blocks(
                    active_blocks[(size_t)best_idx],
                    active_blocks[(size_t)best_idx + 1U]);
                active_blocks.erase(active_blocks.begin() + best_idx + 1);
            }

            blocks = std::move(active_blocks);
            block_atoms.reserve(blocks.size());
            for (size_t i = 0; i < blocks.size(); ++i) {
                append_block_atom(blocks[i], static_cast<int>(i), block_atoms);
            }
            return;
        }

        blocks.reserve(atoms.size());
        block_atoms.reserve(atoms.size());
        AtomizedBlock current;
        for (size_t i = 0; i < atoms.size(); ++i) {
            if (i > 0) {
                const bool same_type = atomized_atoms_should_merge(atoms[i - 1], atoms[i], rule);
                if (!same_type) {
                    append_block_atom(current, (int)blocks.size(), block_atoms);
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
            current.empirical_prediction = atoms[i].empirical_prediction;
            current.teacher_prediction = atoms[i].teacher_prediction;
            if (!atoms[i].class_weight.empty()) {
                if (current.class_weight.empty()) {
                    current.class_weight.assign(atoms[i].class_weight.size(), 0.0);
                    current.teacher_class_weight.assign(atoms[i].teacher_class_weight.size(), 0.0);
                }
                for (size_t cls = 0; cls < atoms[i].class_weight.size(); ++cls) {
                    current.class_weight[cls] += atoms[i].class_weight[cls];
                    current.teacher_class_weight[cls] += atoms[i].teacher_class_weight[cls];
                }
            }
        }

        append_block_atom(current, (int)blocks.size(), block_atoms);
        blocks.push_back(std::move(current));
    }

    AtomizedPrefixes build_atomized_prefixes(const std::vector<AtomizedAtom> &atoms) const {
        AtomizedPrefixes prefix;
        const size_t count = atoms.size();
        prefix.rows.assign(count + 1, 0);
        prefix.pos.assign(count + 1, 0.0);
        prefix.neg.assign(count + 1, 0.0);
        prefix.teacher_pos.assign(count + 1, 0.0);
        prefix.teacher_neg.assign(count + 1, 0.0);
        if (!binary_mode_) {
            prefix.class_weight_prefix.assign((count + 1U) * static_cast<size_t>(n_classes_), 0.0);
            prefix.teacher_class_weight_prefix.assign((count + 1U) * static_cast<size_t>(n_classes_), 0.0);
        }
        for (size_t i = 0; i < count; ++i) {
            prefix.rows[i + 1] = prefix.rows[i] + atoms[i].row_count;
            prefix.pos[i + 1] = prefix.pos[i] + atoms[i].pos_weight;
            prefix.neg[i + 1] = prefix.neg[i] + atoms[i].neg_weight;
            prefix.teacher_pos[i + 1] = prefix.teacher_pos[i] + atoms[i].teacher_pos_weight;
            prefix.teacher_neg[i + 1] = prefix.teacher_neg[i] + atoms[i].teacher_neg_weight;
            if (!binary_mode_) {
                const size_t prev_base = i * static_cast<size_t>(n_classes_);
                const size_t next_base = (i + 1U) * static_cast<size_t>(n_classes_);
                for (int cls = 0; cls < n_classes_; ++cls) {
                    prefix.class_weight_prefix[next_base + static_cast<size_t>(cls)] =
                        prefix.class_weight_prefix[prev_base + static_cast<size_t>(cls)] +
                        atoms[i].class_weight[(size_t)cls];
                    prefix.teacher_class_weight_prefix[next_base + static_cast<size_t>(cls)] =
                        prefix.teacher_class_weight_prefix[prev_base + static_cast<size_t>(cls)] +
                        atoms[i].teacher_class_weight[(size_t)cls];
                }
            }
        }
        return prefix;
    }

    static std::vector<int> lift_block_assignment_to_atoms(
        const std::vector<AtomizedBlock> &blocks,
        const std::vector<int> &block_assignment,
        int atom_count
    ) {
        std::vector<int> atom_assignment((size_t)atom_count, -1);
        for (size_t block_idx = 0; block_idx < blocks.size() && block_idx < block_assignment.size(); ++block_idx) {
            const int group_idx = block_assignment[block_idx];
            for (int atom_pos : blocks[block_idx].atom_positions) {
                atom_assignment[(size_t)atom_pos] = group_idx;
            }
        }
        return atom_assignment;
    }

    static bool project_atom_assignment_to_blocks(
        const std::vector<AtomizedBlock> &blocks,
        const std::vector<AtomizedAtom> &atoms,
        const std::vector<int> &atom_assignment,
        std::vector<int> &block_assignment,
        std::vector<unsigned char> &mixed_block
    ) {
        block_assignment.assign(blocks.size(), -1);
        mixed_block.assign(blocks.size(), 0);
        for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
            const auto &positions = blocks[block_idx].atom_positions;
            if (positions.empty()) {
                return false;
            }
            std::unordered_map<int, int> group_rows;
            int selected_group = -1;
            int selected_rows = -1;
            for (int atom_pos : positions) {
                if (atom_pos < 0 || atom_pos >= (int)atom_assignment.size() ||
                    atom_pos >= (int)atoms.size()) {
                    return false;
                }
                const int group_idx = atom_assignment[(size_t)atom_pos];
                if (group_idx < 0) {
                    return false;
                }
                const int rows = std::max(1, atoms[(size_t)atom_pos].row_count);
                int &group_total = group_rows[group_idx];
                group_total += rows;
                if (group_total > selected_rows) {
                    selected_rows = group_total;
                    selected_group = group_idx;
                }
            }
            if (selected_group < 0) {
                return false;
            }
            block_assignment[block_idx] = selected_group;
            for (int atom_pos : positions) {
                if (atom_assignment[(size_t)atom_pos] != selected_group) {
                    mixed_block[block_idx] = 1;
                    break;
                }
            }
        }
        return true;
    }

    static std::vector<std::pair<int, int>> build_active_block_windows(
        const std::vector<int> &before,
        const std::vector<int> &after,
        const std::vector<unsigned char> *extra_active = nullptr
    ) {
        std::vector<std::pair<int, int>> windows;
        const int count = (int)after.size();
        if (count <= 0 || before.size() != after.size()) {
            return windows;
        }

        std::vector<unsigned char> active((size_t)count, 0);
        bool any_active = false;
        for (int idx = 0; idx < count; ++idx) {
            const bool changed = before[(size_t)idx] != after[(size_t)idx];
            const bool extra = extra_active != nullptr && idx < (int)extra_active->size() && (*extra_active)[(size_t)idx] != 0;
            const unsigned char is_active = (changed || extra) ? 1U : 0U;
            active[(size_t)idx] = is_active;
            any_active = any_active || (is_active != 0U);
        }
        if (!any_active) {
            return windows;
        }

        int idx = 0;
        while (idx < count) {
            if (!active[(size_t)idx]) {
                ++idx;
                continue;
            }
            int start = idx;
            int end = idx;
            while (end + 1 < count && active[(size_t)(end + 1)]) {
                ++end;
            }
            start = std::max(0, start - 1);
            end = std::min(count - 1, end + 1);
            while (start > 0 && after[(size_t)(start - 1)] == after[(size_t)start]) {
                --start;
            }
            while (end + 1 < count && after[(size_t)(end + 1)] == after[(size_t)end]) {
                ++end;
            }
            if (!windows.empty() && start <= windows.back().second + 1) {
                windows.back().second = std::max(windows.back().second, end);
            } else {
                windows.push_back({start, end});
            }
            idx = end + 1;
        }
        return windows;
    }

    static std::vector<std::pair<int, int>> block_windows_to_atom_windows(
        const std::vector<AtomizedBlock> &blocks,
        const std::vector<std::pair<int, int>> &block_windows
    ) {
        std::vector<std::pair<int, int>> atom_windows;
        atom_windows.reserve(block_windows.size());
        for (const auto &window : block_windows) {
            const auto &left_positions = blocks[(size_t)window.first].atom_positions;
            const auto &right_positions = blocks[(size_t)window.second].atom_positions;
            if (left_positions.empty() || right_positions.empty()) {
                continue;
            }
            atom_windows.push_back({left_positions.front(), right_positions.back()});
        }
        return atom_windows;
    }

    AtomizedScore score_group_assignment(
        int feature,
        const std::vector<AtomizedAtom> &atoms,
        const std::vector<std::vector<int>> &group_atom_positions,
        const std::vector<int> *assignment = nullptr,
        const std::vector<double> *adjacency_bonus = nullptr,
        double adjacency_bonus_total = 0.0
    ) const {
        AtomizedScore score{0.0, 0.0, 0.0, 0.0, 0.0, 0};
        double kept_adjacency_bonus = 0.0;
        const bool has_adjacency_bonus =
            adjacency_bonus != nullptr && !adjacency_bonus->empty();
        for (const auto &group : group_atom_positions) {
            if (group.empty()) {
                return AtomizedScore{};
            }
            int row_count = 0;
            double pos_weight = 0.0;
            double neg_weight = 0.0;
            double teacher_pos_weight = 0.0;
            double teacher_neg_weight = 0.0;
            std::vector<double> class_weight;
            std::vector<double> teacher_class_weight;
            if (!binary_mode_) {
                class_weight.assign((size_t)n_classes_, 0.0);
                teacher_class_weight.assign((size_t)n_classes_, 0.0);
            }
            int components = 1;
            int prev_pos = group.front();
            for (size_t idx = 0; idx < group.size(); ++idx) {
                const int atom_pos = group[idx];
                if (idx > 0 && atom_pos != prev_pos + 1) {
                    ++components;
                    score.boundary_penalty += noncontiguous_boundary_penalty(
                        feature,
                        atoms[(size_t)prev_pos],
                        atoms[(size_t)atom_pos]);
                } else if (idx > 0 && has_adjacency_bonus && (size_t)prev_pos < adjacency_bonus->size()) {
                    kept_adjacency_bonus += (*adjacency_bonus)[(size_t)prev_pos];
                }
                const AtomizedAtom &atom = atoms[(size_t)atom_pos];
                row_count += atom.row_count;
                pos_weight += atom.pos_weight;
                neg_weight += atom.neg_weight;
                teacher_pos_weight += atom.teacher_pos_weight;
                teacher_neg_weight += atom.teacher_neg_weight;
                if (!binary_mode_) {
                    for (int cls = 0; cls < n_classes_; ++cls) {
                        class_weight[(size_t)cls] += atom.class_weight[(size_t)cls];
                        teacher_class_weight[(size_t)cls] += atom.teacher_class_weight[(size_t)cls];
                    }
                }
                prev_pos = atom_pos;
            }
            if (row_count < min_child_size_) {
                return AtomizedScore{};
            }
            if (binary_mode_) {
                score.hard_loss += split_leaf_loss(pos_weight, neg_weight);
                score.soft_loss += split_leaf_loss(teacher_pos_weight, teacher_neg_weight);
                score.hard_impurity += hard_label_impurity(pos_weight, neg_weight);
                score.soft_impurity += hard_label_impurity(teacher_pos_weight, teacher_neg_weight);
            } else {
                score.hard_loss += split_leaf_loss(class_weight);
                score.soft_loss += split_leaf_loss(teacher_class_weight);
                score.hard_impurity += hard_label_impurity(class_weight);
                score.soft_impurity += hard_label_impurity(teacher_class_weight);
            }
            score.components += components;
        }
        if (adjacency_bonus != nullptr) {
            score.boundary_penalty += kept_adjacency_bonus - adjacency_bonus_total;
        } else if (feature >= 0 && assignment != nullptr) {
            for (int atom_pos = 1; atom_pos < (int)atoms.size(); ++atom_pos) {
                if ((*assignment)[(size_t)(atom_pos - 1)] != (*assignment)[(size_t)atom_pos]) {
                    score.boundary_penalty -= contiguous_boundary_bonus(
                        feature,
                        atoms[(size_t)(atom_pos - 1)],
                        atoms[(size_t)atom_pos]);
                }
            }
        }
        return score;
    }

    static bool fill_groups_from_assignment(
        const std::vector<int> &assign,
        int groups,
        std::vector<std::vector<int>> &out,
        std::vector<int> &counts
    ) {
        out.resize((size_t)groups);
        counts.assign((size_t)groups, 0);
        for (auto &group : out) {
            group.clear();
        }
        for (int atom_pos = 0; atom_pos < (int)assign.size(); ++atom_pos) {
            const int group_idx = assign[(size_t)atom_pos];
            if (group_idx >= 0 && group_idx < groups) {
                ++counts[(size_t)group_idx];
            } else {
                return false;
            }
        }
        for (int group_idx = 0; group_idx < groups; ++group_idx) {
            if (counts[(size_t)group_idx] <= 0) {
                return false;
            }
            out[(size_t)group_idx].reserve((size_t)counts[(size_t)group_idx]);
        }
        for (int atom_pos = 0; atom_pos < (int)assign.size(); ++atom_pos) {
            const int group_idx = assign[(size_t)atom_pos];
            out[(size_t)group_idx].push_back(atom_pos);
        }
        return true;
    }

    AtomizedCandidate candidate_from_assignment(
        int feature,
        const std::vector<AtomizedAtom> &atoms,
        const std::vector<int> &assign,
        int groups,
        const std::vector<double> *adjacency_bonus = nullptr,
        double adjacency_bonus_total = 0.0,
        AtomizedObjectiveMode objective_mode = AtomizedObjectiveMode::kImpurity
    ) const {
        AtomizedCandidate out;
        if (groups < 2) {
            return out;
        }
        if (binary_mode_) {
            const int atom_count = (int)assign.size();
            std::vector<int> counts((size_t)groups, 0);
            const bool has_adjacency_bonus =
                adjacency_bonus != nullptr && !adjacency_bonus->empty();
            for (int atom_pos = 0; atom_pos < atom_count; ++atom_pos) {
                const int group_idx = assign[(size_t)atom_pos];
                if (group_idx < 0 || group_idx >= groups) {
                    return AtomizedCandidate{};
                }
                ++counts[(size_t)group_idx];
            }

            for (int group_idx = 0; group_idx < groups; ++group_idx) {
                if (counts[(size_t)group_idx] <= 0) {
                    return AtomizedCandidate{};
                }
            }

            std::vector<int> group_rows((size_t)groups, 0);
            std::vector<int> group_last_pos((size_t)groups, -1);
            std::vector<int> group_components((size_t)groups, 0);
            std::vector<double> group_pos((size_t)groups, 0.0);
            std::vector<double> group_neg((size_t)groups, 0.0);
            std::vector<double> group_teacher_pos((size_t)groups, 0.0);
            std::vector<double> group_teacher_neg((size_t)groups, 0.0);
            double kept_adjacency_bonus = 0.0;
            out.score = AtomizedScore{0.0, 0.0, 0.0, 0.0, 0.0, 0};

            for (int atom_pos = 0; atom_pos < atom_count; ++atom_pos) {
                const int group_idx = assign[(size_t)atom_pos];
                const int last_pos = group_last_pos[(size_t)group_idx];
                if (last_pos >= 0) {
                    if (atom_pos != last_pos + 1) {
                        out.score.boundary_penalty += noncontiguous_boundary_penalty(
                            feature,
                            atoms[(size_t)last_pos],
                            atoms[(size_t)atom_pos]);
                        ++group_components[(size_t)group_idx];
                    } else if (has_adjacency_bonus && (size_t)last_pos < adjacency_bonus->size()) {
                        kept_adjacency_bonus += (*adjacency_bonus)[(size_t)last_pos];
                    }
                } else {
                    group_components[(size_t)group_idx] = 1;
                }

                group_last_pos[(size_t)group_idx] = atom_pos;
                const AtomizedAtom &atom = atoms[(size_t)atom_pos];
                group_rows[(size_t)group_idx] += atom.row_count;
                group_pos[(size_t)group_idx] += atom.pos_weight;
                group_neg[(size_t)group_idx] += atom.neg_weight;
                group_teacher_pos[(size_t)group_idx] += atom.teacher_pos_weight;
                group_teacher_neg[(size_t)group_idx] += atom.teacher_neg_weight;
            }

            for (int group_idx = 0; group_idx < groups; ++group_idx) {
                if (group_rows[(size_t)group_idx] < min_child_size_) {
                    return AtomizedCandidate{};
                }
                const double branch_hard_loss = split_leaf_loss(
                    group_pos[(size_t)group_idx],
                    group_neg[(size_t)group_idx]);
                out.score.hard_loss += branch_hard_loss;
                out.score.soft_loss += split_leaf_loss(
                    group_teacher_pos[(size_t)group_idx],
                    group_teacher_neg[(size_t)group_idx]);
                out.score.hard_impurity += hard_label_impurity(
                    group_pos[(size_t)group_idx],
                    group_neg[(size_t)group_idx]);
                out.score.soft_impurity += hard_label_impurity(
                    group_teacher_pos[(size_t)group_idx],
                    group_teacher_neg[(size_t)group_idx]);
                out.score.components += group_components[(size_t)group_idx];
            }

            if (adjacency_bonus != nullptr) {
                out.score.boundary_penalty += kept_adjacency_bonus - adjacency_bonus_total;
            } else if (feature >= 0) {
                for (int atom_pos = 1; atom_pos < atom_count; ++atom_pos) {
                    if (assign[(size_t)(atom_pos - 1)] != assign[(size_t)atom_pos]) {
                        out.score.boundary_penalty -= contiguous_boundary_bonus(
                            feature,
                            atoms[(size_t)(atom_pos - 1)],
                            atoms[(size_t)atom_pos]);
                    }
                }
            }

            out.feasible = true;
            out.feature = feature;
            out.groups = groups;
            out.hard_loss_mode = (objective_mode == AtomizedObjectiveMode::kHardLoss);
            out.assignment = assign;
            return out;
        }
        std::vector<std::vector<int>> group_atom_positions;
        std::vector<int> group_counts;
        if (!fill_groups_from_assignment(assign, groups, group_atom_positions, group_counts)) {
            return AtomizedCandidate{};
        }
        out.score = score_group_assignment(
            feature,
            atoms,
            group_atom_positions,
            &assign,
            adjacency_bonus,
            adjacency_bonus_total);
        if (!std::isfinite(out.score.hard_impurity)) {
            return AtomizedCandidate{};
        }
        out.feasible = true;
        out.feature = feature;
        out.groups = groups;
        out.hard_loss_mode = (objective_mode == AtomizedObjectiveMode::kHardLoss);
        out.assignment = assign;
        return out;
    }

    AtomizedCandidate lift_block_candidate_to_atoms(
        int feature,
        const std::vector<AtomizedBlock> &blocks,
        const std::vector<AtomizedAtom> &atoms,
        const AtomizedCandidate &block_candidate,
        const std::vector<double> *adjacency_bonus = nullptr,
        double adjacency_bonus_total = 0.0,
        AtomizedObjectiveMode objective_mode = AtomizedObjectiveMode::kImpurity
    ) const {
        if (!block_candidate.feasible) {
            return AtomizedCandidate{};
        }
        const std::vector<int> atom_assignment =
            lift_block_assignment_to_atoms(blocks, block_candidate.assignment, (int)atoms.size());
        return candidate_from_assignment(
            feature,
            atoms,
            atom_assignment,
            block_candidate.groups,
            adjacency_bonus,
            adjacency_bonus_total,
            objective_mode);
    }
