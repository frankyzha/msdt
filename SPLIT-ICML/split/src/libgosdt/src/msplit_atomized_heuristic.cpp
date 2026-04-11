    std::vector<AtomizedCandidate> select_atomized_candidates_for_arity(
        const PreparedFeatureAtomized &prepared,
        int groups,
        double mu_node
    ) const {
        std::vector<AtomizedCandidate> selected;
        selected.reserve(3);
        if (groups < 2 ||
            groups >= (int)prepared.coarse_by_groups.size() ||
            groups >= (int)prepared.coarse_by_groups_hardloss.size()) {
            return selected;
        }

        const AtomizedCoarseCandidate &impurity_coarse =
            prepared.coarse_by_groups[(size_t)groups];
        const AtomizedCandidate &impurity_raw_seed =
            prepared.coarse_by_groups[(size_t)groups].geometry_seed_candidate;
        AtomizedCandidate impurity = nominate_folded_family_candidate(
            prepared,
            mu_node,
            impurity_coarse,
            impurity_raw_seed,
            AtomizedObjectiveMode::kImpurity);

        const AtomizedCandidate &misclassification_raw_seed =
            prepared.coarse_by_groups_hardloss[(size_t)groups].geometry_seed_candidate;
        AtomizedCandidate misclassification = nominate_folded_family_candidate(
            prepared,
            mu_node,
            prepared.coarse_by_groups_hardloss[(size_t)groups],
            misclassification_raw_seed,
            AtomizedObjectiveMode::kHardLoss);

        if (impurity.feasible && misclassification.feasible) {
            selected = select_family_nominees(std::move(impurity), std::move(misclassification));
        } else if (impurity.feasible) {
            ++const_cast<Solver *>(this)->debr_final_geo_wins_;
            selected.push_back(std::move(impurity));
        } else if (misclassification.feasible) {
            ++const_cast<Solver *>(this)->debr_final_block_wins_;
            selected.push_back(std::move(misclassification));
        }
        return selected;
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

    std::string serialized_indices_depth_key(
        const std::vector<int> &indices,
        int depth_remaining
    ) const {
        std::string key = serialized_indices_key(indices);
        const int64_t depth_value = depth_remaining;
        key.append(reinterpret_cast<const char *>(&depth_value), sizeof(depth_value));
        return key;
    }

    std::string serialized_indices_key(const std::vector<int> &indices) const {
        std::string key;
        key.reserve(sizeof(uint64_t) + indices.size() * 2U);
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

    struct CandidateEval {
        int feature = -1;
        int groups = 0;
        double cheap_score = kInfinity;
        double cheap_lower_bound = kInfinity;
    };

    struct NomineeEval {
        int feature = -1;
        int groups = 0;
        AtomizedCandidate candidate;
        double cheap_score = kInfinity;
        double cheap_lower_bound = kInfinity;
        std::vector<std::vector<int>> child_indices;
        std::vector<SubproblemStats> child_stats;
        std::vector<std::vector<std::pair<int, int>>> group_spans;
        bool materialized = false;
        double lower_bound = kInfinity;
        double upper_bound = kInfinity;
    };

    size_t collect_cheap_candidates(
        const std::vector<int> &indices,
        double mu_node,
        bool above_lookahead,
        std::vector<PreparedFeatureAtomized> &prepared_by_feature,
        std::vector<CandidateEval> &candidate_evals
    ) const {
        prepared_by_feature.assign((size_t)n_features_, PreparedFeatureAtomized{});
        candidate_evals.clear();
        candidate_evals.reserve((size_t)n_features_ * 2U);
        size_t preserved_feature_count = 0U;

        for (int feature = 0; feature < n_features_; ++feature) {
            PreparedFeatureAtomized prepared;
            {
                ScopedTimer feature_timer(profiling_feature_prepare_sec_, profiling_enabled_);
                if (!prepare_feature_atomized_local(
                        indices,
                        feature,
                        mu_node,
                        prepared)) {
                    continue;
                }
            }
            ++preserved_feature_count;
            prepared_by_feature[(size_t)feature] = std::move(prepared);

            const PreparedFeatureAtomized &prepared_ref = prepared_by_feature[(size_t)feature];
            if (!prepared_ref.valid) {
                continue;
            }

            for (size_t groups = 2; groups < prepared_ref.coarse_by_groups.size(); ++groups) {
                const AtomizedCoarseCandidate &impurity_coarse = prepared_ref.coarse_by_groups[groups];
                const bool impurity_feasible = impurity_coarse.candidate.feasible;
                const bool hardloss_feasible =
                    groups < prepared_ref.coarse_by_groups_hardloss.size() &&
                    prepared_ref.coarse_by_groups_hardloss[groups].candidate.feasible;
                if (above_lookahead) {
                    auto *self = const_cast<Solver *>(this);
                    if (impurity_feasible) {
                        ++self->above_lookahead_impurity_pairs_total_;
                    }
                    if (hardloss_feasible) {
                        ++self->above_lookahead_hardloss_pairs_total_;
                    }
                }
                if (!impurity_feasible && !hardloss_feasible) {
                    continue;
                }

                CandidateEval eval;
                eval.feature = feature;
                eval.groups = static_cast<int>(groups);

                if (impurity_feasible) {
                    const double score =
                        atomized_score_proxy(
                            impurity_coarse.candidate.score,
                            mu_node,
                            AtomizedObjectiveMode::kImpurity);
                    eval.cheap_score = score;
                    eval.cheap_lower_bound = std::min(prepared_ref.atom_hard_floor, prepared_ref.atom_imp_floor) +
                        mu_node * static_cast<double>(eval.groups);
                }

                if (hardloss_feasible) {
                    const AtomizedCoarseCandidate &hardloss_coarse =
                        prepared_ref.coarse_by_groups_hardloss[groups];
                    const double score =
                        atomized_score_proxy(
                            hardloss_coarse.candidate.score,
                            mu_node,
                            AtomizedObjectiveMode::kHardLoss);
                    if (!std::isfinite(eval.cheap_score) || score < eval.cheap_score - kEpsUpdate) {
                        eval.cheap_score = score;
                        eval.cheap_lower_bound = std::min(prepared_ref.atom_hard_floor, prepared_ref.atom_imp_floor) +
                            mu_node * static_cast<double>(eval.groups);
                    }
                }

                if (std::isfinite(eval.cheap_score)) {
                    candidate_evals.push_back(eval);
                }
            }
        }
        return preserved_feature_count;
    }

    std::vector<NomineeEval> build_nominee_evals(
        const std::vector<PreparedFeatureAtomized> &prepared_by_feature,
        const std::vector<CandidateEval> &candidate_evals,
        double mu_node,
        int depth_remaining
    ) const {
        std::vector<NomineeEval> nominees;
        nominees.reserve(candidate_evals.size() * 3U);

        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        std::unordered_set<std::string> nominee_signature_seen;
        nominee_signature_seen.reserve(candidate_evals.size() * 3U);
        auto make_signature_key = [&](int feature, int groups, std::vector<int> &assignment) {
            if (!assignment.empty()) {
                std::vector<int> label_map((size_t)groups, -1);
                int next_label = 0;
                for (int &label : assignment) {
                    if (label < 0 || label >= groups) {
                        label = -1;
                        continue;
                    }
                    int &mapped = label_map[(size_t)label];
                    if (mapped < 0) {
                        mapped = next_label++;
                    }
                    label = mapped;
                }
            }
            std::string key;
            key.reserve(sizeof(uint64_t) * (assignment.size() + 4U));
            auto append_varint = [&](uint64_t value) {
                while (value >= 0x80U) {
                    key.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
                    value >>= 7U;
                }
                key.push_back(static_cast<char>(value));
            };
            append_varint((uint64_t)feature);
            append_varint((uint64_t)groups);
            append_varint((uint64_t)assignment.size());
            for (int value : assignment) {
                append_varint((uint64_t)std::max(0, value));
            }
            return key;
        };

        for (const CandidateEval &eval : candidate_evals) {
            const PreparedFeatureAtomized &prepared = prepared_by_feature[(size_t)eval.feature];
            if (!prepared.valid) {
                continue;
            }
            std::vector<AtomizedCandidate> candidates = select_atomized_candidates_for_arity(
                prepared,
                eval.groups,
                mu_node);
            if (candidates.empty()) {
                continue;
            }
            telemetry.greedy_interval_evals += static_cast<long long>(candidates.size());
            telemetry.atomized_final_candidates += static_cast<long long>(candidates.size());

            for (AtomizedCandidate &candidate : candidates) {
                if (!candidate.feasible) {
                    continue;
                }
                NomineeEval nominee;
                nominee.feature = eval.feature;
                nominee.groups = eval.groups;
                const bool hard_loss_mode = candidate.hard_loss_mode;
                const AtomizedObjectiveMode nominee_mode = hard_loss_mode
                    ? AtomizedObjectiveMode::kHardLoss
                    : AtomizedObjectiveMode::kImpurity;
                const auto &family_coarse = hard_loss_mode
                    ? prepared.coarse_by_groups_hardloss[(size_t)eval.groups]
                    : prepared.coarse_by_groups[(size_t)eval.groups];
                nominee.cheap_score = family_coarse.candidate.feasible
                    ? atomized_score_proxy(
                        family_coarse.candidate.score,
                        mu_node,
                        nominee_mode)
                    : eval.cheap_score;
                nominee.cheap_lower_bound = eval.cheap_lower_bound;
                const std::string signature_key = make_signature_key(
                    nominee.feature,
                    nominee.groups,
                    candidate.assignment);
                if (nominee_signature_seen.find(signature_key) != nominee_signature_seen.end()) {
                    continue;
                }
                nominee_signature_seen.emplace(std::move(signature_key));
                nominee.candidate = std::move(candidate);
                nominees.push_back(std::move(nominee));
            }
        }
        auto *self = const_cast<Solver *>(this);
        self->nominee_unique_total_ += static_cast<long long>(nominees.size());
        return nominees;
    }

    GreedyResult greedy_complete_impl(
        std::vector<int> indices,
        int depth_remaining,
        bool exact_mode) {
        ++profiling_greedy_complete_calls_;
        record_greedy_complete_call(depth_remaining);
        const int current_depth = full_depth_budget_ - depth_remaining;
        const bool above_lookahead = exact_mode && current_depth < effective_lookahead_depth_;
        if (above_lookahead) {
            ++exact_dp_subproblem_calls_above_lookahead_;
        } else {
            ++greedy_subproblem_calls_;
        }
        ScopedTimer greedy_timer(profiling_greedy_complete_sec_, profiling_enabled_);
        check_timeout();

        const bool use_greedy_cache = greedy_cache_max_depth_ >= 0;
        std::string state_key;
        if (use_greedy_cache) {
            state_key = serialized_indices_depth_key(indices, depth_remaining);
            auto cache_it = greedy_cache_.find(state_key);
            if (cache_it != greedy_cache_.end()) {
                ++greedy_cache_hits_;
                trace_greedy_snapshot(
                    "cache_hit",
                    depth_remaining,
                    indices.size(),
                    0U,
                    0U,
                    0U,
                    0U,
                    0U,
                    cache_it->second.result.objective,
                    cache_it->second.result.objective,
                    0U);
                return cache_it->second.result;
            }
        }
        trace_greedy_snapshot(
            "enter",
            depth_remaining,
            indices.size(),
            0U,
            0U,
            0U,
            0U,
            0U,
            kInfinity,
            kInfinity,
            0U);

        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        auto record_empty_candidate_curves = [&]() {
            if (!detailed_selector_telemetry_enabled_) {
                return;
            }
            telemetry.per_node_candidate_upper_bounds.emplace_back();
            telemetry.per_node_candidate_lower_bounds.emplace_back();
            telemetry.per_node_candidate_hard_loss.emplace_back();
            telemetry.per_node_candidate_impurity_objective.emplace_back();
            telemetry.per_node_candidate_hard_impurity.emplace_back();
            telemetry.per_node_candidate_soft_impurity.emplace_back();
            telemetry.per_node_candidate_boundary_penalty.emplace_back();
            telemetry.per_node_candidate_components.emplace_back();
        };
        auto record_candidate_curves = [&](
            std::vector<double> upper_bounds,
            std::vector<double> lower_bounds,
            std::vector<double> hard_loss,
            std::vector<double> impurity_objective,
            std::vector<double> hard_impurity,
            std::vector<double> soft_impurity,
            std::vector<double> boundary_penalty,
            std::vector<long long> components
        ) {
            if (!detailed_selector_telemetry_enabled_) {
                return;
            }
            telemetry.per_node_candidate_upper_bounds.emplace_back(std::move(upper_bounds));
            telemetry.per_node_candidate_lower_bounds.emplace_back(std::move(lower_bounds));
            telemetry.per_node_candidate_hard_loss.emplace_back(std::move(hard_loss));
            telemetry.per_node_candidate_impurity_objective.emplace_back(std::move(impurity_objective));
            telemetry.per_node_candidate_hard_impurity.emplace_back(std::move(hard_impurity));
            telemetry.per_node_candidate_soft_impurity.emplace_back(std::move(soft_impurity));
            telemetry.per_node_candidate_boundary_penalty.emplace_back(std::move(boundary_penalty));
            telemetry.per_node_candidate_components.emplace_back(std::move(components));
        };
        auto record_node_scale = [&](const SubproblemStats &node_stats) {
            if (!detailed_selector_telemetry_enabled_) {
                return;
            }
            telemetry.per_node_total_weight.push_back(node_stats.sum_weight);
            telemetry.per_node_mu_node.push_back(effective_sample_unit(node_stats));
        };

        const SubproblemStats stats = compute_subproblem_stats(indices);
        auto [leaf_objective, leaf_tree] = leaf_solution(stats);
        const double reference_floor_leaf =
            teacher_available_
                ? (stats.reference_error_weight + regularization_)
                : -kInfinity;
        if (depth_remaining <= 0) {
            record_node_scale(stats);
            record_empty_candidate_curves();
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_leaf_depth",
                depth_remaining,
                indices.size(),
                0U,
                0U,
                0U,
                0U,
                0U,
                solved.objective,
                solved.objective,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }
        if (stats.pure) {
            record_node_scale(stats);
            record_empty_candidate_curves();
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_pure",
                depth_remaining,
                indices.size(),
                0U,
                0U,
                0U,
                0U,
                0U,
                solved.objective,
                solved.objective,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }
        if ((int)indices.size() < min_split_size_) {
            record_node_scale(stats);
            record_empty_candidate_curves();
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_min_split",
                depth_remaining,
                indices.size(),
                0U,
                0U,
                0U,
                0U,
                0U,
                solved.objective,
                solved.objective,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }
        if (above_lookahead && teacher_available_ &&
            reference_floor_leaf + kEpsUpdate >= leaf_objective) {
            record_node_scale(stats);
            record_empty_candidate_curves();
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_reference_leaf",
                depth_remaining,
                indices.size(),
                0U,
                0U,
                0U,
                0U,
                0U,
                solved.objective,
                reference_floor_leaf,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }

        const double mu_node = effective_sample_unit(stats);
        const double prune_slack = mu_node;
        const bool disable_pair_prune_globally = disable_coarse_pruning_;
        auto bump_count_histogram = [](std::vector<long long> &hist, size_t bucket) {
            if (hist.size() <= bucket) {
                hist.resize(bucket + 1, 0LL);
            }
            ++hist[bucket];
        };
        std::vector<CandidateEval> candidate_evals;
        std::vector<PreparedFeatureAtomized> prepared_by_feature((size_t)n_features_);
        size_t preserved_feature_count = 0U;
        {
            ScopedTimer candidate_timer(profiling_candidate_generation_sec_, profiling_enabled_);
            preserved_feature_count = collect_cheap_candidates(
                indices,
                mu_node,
                above_lookahead,
                prepared_by_feature,
                candidate_evals);
        }
        if (candidate_evals.empty()) {
            record_node_scale(stats);
            record_empty_candidate_curves();
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_no_candidate",
                depth_remaining,
                indices.size(),
                preserved_feature_count,
                0U,
                0U,
                0U,
                0U,
                solved.objective,
                solved.objective,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }

        double best_coarse_proxy = kInfinity;
        for (const CandidateEval &eval : candidate_evals) {
            best_coarse_proxy = std::min(best_coarse_proxy, eval.cheap_score);
        }
        if (!std::isfinite(best_coarse_proxy)) {
            record_node_scale(stats);
            record_empty_candidate_curves();
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_no_candidate",
                depth_remaining,
                indices.size(),
                preserved_feature_count,
                0U,
                0U,
                0U,
                0U,
                solved.objective,
                leaf_objective,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }

        std::vector<CandidateEval> surviving_candidate_evals;
        surviving_candidate_evals.reserve(candidate_evals.size());
        std::vector<size_t> feature_survivor_pair_count((size_t)n_features_, 0U);
        size_t pruned_pair_count = 0U;
        if (disable_pair_prune_globally) {
            surviving_candidate_evals = candidate_evals;
            for (const CandidateEval &eval : candidate_evals) {
                ++feature_survivor_pair_count[(size_t)eval.feature];
            }
        } else {
            const double pair_prune_cutoff = best_coarse_proxy + mu_node + kEpsUpdate;
            for (const CandidateEval &eval : candidate_evals) {
                if (eval.cheap_score <= pair_prune_cutoff) {
                    surviving_candidate_evals.push_back(eval);
                    ++feature_survivor_pair_count[(size_t)eval.feature];
                } else {
                    ++pruned_pair_count;
                }
            }
        }
        for (size_t feature = 0; feature < feature_survivor_pair_count.size(); ++feature) {
            bump_count_histogram(greedy_feature_survivor_histogram_, feature_survivor_pair_count[feature]);
        }
        if (surviving_candidate_evals.empty()) {
            record_node_scale(stats);
            record_empty_candidate_curves();
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_no_pair_survivor",
                depth_remaining,
                indices.size(),
                preserved_feature_count,
                static_cast<unsigned long long>(candidate_evals.size()),
                0U,
                0U,
                0U,
                solved.objective,
                best_coarse_proxy,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }

        std::vector<NomineeEval> nominee_evals = build_nominee_evals(
            prepared_by_feature,
            surviving_candidate_evals,
            mu_node,
            depth_remaining);
        if (nominee_evals.empty()) {
            record_node_scale(stats);
            record_empty_candidate_curves();
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_no_nominee",
                depth_remaining,
                indices.size(),
                preserved_feature_count,
                static_cast<unsigned long long>(surviving_candidate_evals.size()),
                0U,
                0U,
                0U,
                solved.objective,
                best_coarse_proxy,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }

        auto nominee_hard_ceiling = [&](const NomineeEval &eval) {
            return eval.candidate.score.hard_loss +
                regularization_ * static_cast<double>(eval.candidate.groups);
        };

        auto nominee_prefer = [&](size_t lhs_idx, size_t rhs_idx) {
            const NomineeEval &lhs = nominee_evals[lhs_idx];
            const NomineeEval &rhs = nominee_evals[rhs_idx];
            const double lhs_impurity = atomized_primary_objective(
                lhs.candidate.score,
                AtomizedObjectiveMode::kImpurity);
            const double rhs_impurity = atomized_primary_objective(
                rhs.candidate.score,
                AtomizedObjectiveMode::kImpurity);
            if (lhs_impurity < rhs_impurity - kEpsUpdate) {
                return true;
            }
            if (rhs_impurity < lhs_impurity - kEpsUpdate) {
                return false;
            }
            const double lhs_hard_ceiling = nominee_hard_ceiling(lhs);
            const double rhs_hard_ceiling = nominee_hard_ceiling(rhs);
            if (lhs_hard_ceiling < rhs_hard_ceiling - kEpsUpdate) {
                return true;
            }
            if (rhs_hard_ceiling < lhs_hard_ceiling - kEpsUpdate) {
                return false;
            }
            if (lhs.candidate.score.boundary_penalty < rhs.candidate.score.boundary_penalty - kEpsUpdate) {
                return true;
            }
            if (rhs.candidate.score.boundary_penalty < lhs.candidate.score.boundary_penalty - kEpsUpdate) {
                return false;
            }
            if (lhs.candidate.score.components != rhs.candidate.score.components) {
                return lhs.candidate.score.components < rhs.candidate.score.components;
            }
            if (lhs.feature != rhs.feature) {
                return lhs.feature < rhs.feature;
            }
            if (lhs.groups != rhs.groups) {
                return lhs.groups < rhs.groups;
            }
            return lhs_idx < rhs_idx;
        };

        auto resolve_exactify_budget = [&](size_t candidate_count) -> size_t {
            if (candidate_count == 0U) {
                return 0U;
            }
            if (exactify_top_k_ > 0) {
                return std::max<size_t>(
                    1U,
                    std::min(candidate_count, static_cast<size_t>(exactify_top_k_)));
            }
            const size_t budget = static_cast<size_t>(
                std::ceil(std::sqrt(static_cast<double>(candidate_count))));
            return std::max<size_t>(1U, std::min(candidate_count, budget));
        };
        auto nominee_family_prefer = [&](size_t lhs_idx, size_t rhs_idx) {
            const NomineeEval &lhs = nominee_evals[lhs_idx];
            const NomineeEval &rhs = nominee_evals[rhs_idx];
            const AtomizedObjectiveMode lhs_mode = lhs.candidate.hard_loss_mode
                ? AtomizedObjectiveMode::kHardLoss
                : AtomizedObjectiveMode::kImpurity;
            const AtomizedObjectiveMode rhs_mode = rhs.candidate.hard_loss_mode
                ? AtomizedObjectiveMode::kHardLoss
                : AtomizedObjectiveMode::kImpurity;
            const double lhs_primary = atomized_primary_objective(lhs.candidate.score, lhs_mode);
            const double rhs_primary = atomized_primary_objective(rhs.candidate.score, rhs_mode);
            if (lhs_primary < rhs_primary - kEpsUpdate) {
                return true;
            }
            if (rhs_primary < lhs_primary - kEpsUpdate) {
                return false;
            }
            return nominee_prefer(lhs_idx, rhs_idx);
        };

        double search_upper_bound = leaf_objective;
        for (const NomineeEval &eval : nominee_evals) {
            if (eval.candidate.hard_loss_mode) {
                search_upper_bound = std::min(search_upper_bound, nominee_hard_ceiling(eval));
            }
        }

        if (above_lookahead) {
            long long impurity_before = 0;
            long long hardloss_before = 0;
            for (const NomineeEval &eval : nominee_evals) {
                if (eval.candidate.hard_loss_mode) {
                    ++hardloss_before;
                } else {
                    ++impurity_before;
                }
            }
            above_lookahead_impurity_bucket_before_prune_total_ += impurity_before;
            above_lookahead_hardloss_bucket_before_prune_total_ += hardloss_before;
        }

        if (teacher_available_ && above_lookahead) {
            std::vector<NomineeEval> reference_survivors;
            reference_survivors.reserve(nominee_evals.size());
            size_t early_stop_idx = std::numeric_limits<size_t>::max();
            long long reference_pruned = 0;
            for (NomineeEval &eval : nominee_evals) {
                const double candidate_floor =
                    stats.reference_error_weight +
                    regularization_ * static_cast<double>(eval.groups);
                if (candidate_floor + kEpsUpdate >= search_upper_bound) {
                    ++reference_pruned;
                    continue;
                }
                reference_survivors.push_back(std::move(eval));
                const size_t survivor_idx = reference_survivors.size() - 1U;
                if (nominee_hard_ceiling(reference_survivors.back()) <= candidate_floor + kEpsUpdate) {
                    if (early_stop_idx == std::numeric_limits<size_t>::max()) {
                        early_stop_idx = survivor_idx;
                    } else {
                        const double lhs_ceiling = nominee_hard_ceiling(reference_survivors[survivor_idx]);
                        const double rhs_ceiling = nominee_hard_ceiling(reference_survivors[early_stop_idx]);
                        if (lhs_ceiling < rhs_ceiling - kEpsUpdate ||
                            (std::abs(lhs_ceiling - rhs_ceiling) <= kEpsUpdate &&
                             (reference_survivors[survivor_idx].feature < reference_survivors[early_stop_idx].feature ||
                              (reference_survivors[survivor_idx].feature == reference_survivors[early_stop_idx].feature &&
                               (reference_survivors[survivor_idx].groups < reference_survivors[early_stop_idx].groups ||
                                (reference_survivors[survivor_idx].groups == reference_survivors[early_stop_idx].groups &&
                                 reference_survivors[survivor_idx].candidate.score.components <
                                     reference_survivors[early_stop_idx].candidate.score.components)))))) {
                            early_stop_idx = survivor_idx;
                        }
                    }
                }
            }
            nominee_evals.swap(reference_survivors);
            if (early_stop_idx != std::numeric_limits<size_t>::max()) {
                NomineeEval winner = std::move(nominee_evals[early_stop_idx]);
                nominee_evals.clear();
                nominee_evals.push_back(std::move(winner));
            }
            if (above_lookahead) {
                long long impurity_after = 0;
                long long hardloss_after = 0;
                for (const NomineeEval &eval : nominee_evals) {
                    if (eval.candidate.hard_loss_mode) {
                        ++hardloss_after;
                    } else {
                        ++impurity_after;
                    }
                }
                above_lookahead_impurity_bucket_after_prune_total_ += impurity_after;
                above_lookahead_hardloss_bucket_after_prune_total_ += hardloss_after;
            }
            telemetry.atomized_coarse_pruned_candidates += reference_pruned;
        } else if (above_lookahead) {
            long long impurity_after = 0;
            long long hardloss_after = 0;
            for (const NomineeEval &eval : nominee_evals) {
                if (eval.candidate.hard_loss_mode) {
                    ++hardloss_after;
                } else {
                    ++impurity_after;
                }
            }
            above_lookahead_impurity_bucket_after_prune_total_ += impurity_after;
            above_lookahead_hardloss_bucket_after_prune_total_ += hardloss_after;
        }

        if (nominee_evals.empty()) {
            record_node_scale(stats);
            record_empty_candidate_curves();
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_reference_pruned",
                depth_remaining,
                indices.size(),
                preserved_feature_count,
                static_cast<unsigned long long>(surviving_candidate_evals.size()),
                0U,
                0U,
                0U,
                solved.objective,
                search_upper_bound,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }

        std::vector<size_t> alive_indices;
        alive_indices.reserve(nominee_evals.size());
        size_t exactify_budget_count = 0U;
        if (exact_mode) {
            for (size_t idx = 0; idx < nominee_evals.size(); ++idx) {
                alive_indices.push_back(idx);
            }
            std::stable_sort(
                alive_indices.begin(),
                alive_indices.end(),
                nominee_prefer);
            exactify_budget_count = resolve_exactify_budget(nominee_evals.size());
            const auto impurity_it = std::find_if(
                alive_indices.begin(),
                alive_indices.end(),
                [&](size_t idx) { return !nominee_evals[idx].candidate.hard_loss_mode; });
            if (impurity_it != alive_indices.end() &&
                exactify_budget_count > 0U &&
                static_cast<size_t>(std::distance(alive_indices.begin(), impurity_it)) >= exactify_budget_count) {
                alive_indices[exactify_budget_count - 1U] = *impurity_it;
                std::stable_sort(
                    alive_indices.begin(),
                    alive_indices.begin() + static_cast<std::ptrdiff_t>(exactify_budget_count),
                    nominee_prefer);
            }
        } else {
            size_t best_impurity_idx = std::numeric_limits<size_t>::max();
            size_t best_hardloss_idx = std::numeric_limits<size_t>::max();
            for (size_t idx = 0; idx < nominee_evals.size(); ++idx) {
                if (nominee_evals[idx].candidate.hard_loss_mode) {
                    if (best_hardloss_idx == std::numeric_limits<size_t>::max() ||
                        nominee_family_prefer(idx, best_hardloss_idx)) {
                        best_hardloss_idx = idx;
                    }
                } else {
                    if (best_impurity_idx == std::numeric_limits<size_t>::max() ||
                        nominee_family_prefer(idx, best_impurity_idx)) {
                        best_impurity_idx = idx;
                    }
                }
            }
            if (best_impurity_idx != std::numeric_limits<size_t>::max()) {
                alive_indices.push_back(best_impurity_idx);
            }
            if (best_hardloss_idx != std::numeric_limits<size_t>::max() &&
                best_hardloss_idx != best_impurity_idx) {
                alive_indices.push_back(best_hardloss_idx);
            }
            if (alive_indices.empty()) {
                for (size_t idx = 0; idx < nominee_evals.size(); ++idx) {
                    alive_indices.push_back(idx);
                }
            }
            std::stable_sort(
                alive_indices.begin(),
                alive_indices.end(),
                nominee_prefer);
            exactify_budget_count = alive_indices.size();
        }

        std::vector<double> node_candidate_upper_bounds;
        std::vector<double> node_candidate_lower_bounds;
        std::vector<double> node_candidate_hard_loss;
        std::vector<double> node_candidate_impurity_objective;
        std::vector<double> node_candidate_hard_impurity;
        std::vector<double> node_candidate_soft_impurity;
        std::vector<double> node_candidate_boundary_penalty;
        std::vector<long long> node_candidate_components;
        if (detailed_selector_telemetry_enabled_) {
            node_candidate_upper_bounds.reserve(alive_indices.size());
            node_candidate_lower_bounds.reserve(alive_indices.size());
            node_candidate_hard_loss.reserve(alive_indices.size());
            node_candidate_impurity_objective.reserve(alive_indices.size());
            node_candidate_hard_impurity.reserve(alive_indices.size());
            node_candidate_soft_impurity.reserve(alive_indices.size());
            node_candidate_boundary_penalty.reserve(alive_indices.size());
            node_candidate_components.reserve(alive_indices.size());
            for (size_t idx : alive_indices) {
                const NomineeEval &eval = nominee_evals[idx];
                const AtomizedScore &score = eval.candidate.score;
                node_candidate_upper_bounds.push_back(eval.upper_bound);
                node_candidate_lower_bounds.push_back(eval.lower_bound);
                node_candidate_hard_loss.push_back(score.hard_loss);
                node_candidate_impurity_objective.push_back(
                    score.hard_impurity + score.soft_impurity);
                node_candidate_hard_impurity.push_back(score.hard_impurity);
                node_candidate_soft_impurity.push_back(score.soft_impurity);
                node_candidate_boundary_penalty.push_back(score.boundary_penalty);
                node_candidate_components.push_back((long long)score.components);
            }
        }
        record_node_scale(stats);
        telemetry.nominee_exactify_prefix_total += static_cast<long long>(exactify_budget_count);
        telemetry.nominee_exactify_prefix_max = std::max(
            telemetry.nominee_exactify_prefix_max,
            static_cast<long long>(exactify_budget_count));
        bump_count_histogram(
            telemetry.nominee_exactify_prefix_histogram,
            exactify_budget_count);

        record_candidate_curves(
            std::move(node_candidate_upper_bounds),
            std::move(node_candidate_lower_bounds),
            std::move(node_candidate_hard_loss),
            std::move(node_candidate_impurity_objective),
            std::move(node_candidate_hard_impurity),
            std::move(node_candidate_soft_impurity),
            std::move(node_candidate_boundary_penalty),
            std::move(node_candidate_components));

        struct ExactNomineeResult {
            bool valid = false;
            double objective = kInfinity;
            std::shared_ptr<Node> tree;
        };

        std::vector<ExactNomineeResult> exact_results(nominee_evals.size());
        std::vector<unsigned char> finalized(nominee_evals.size(), 0U);
        size_t exact_evaluated_total = 0U;
        size_t incumbent_update_count = 0U;
        size_t processed_candidate_count = 0U;
        size_t recurse_attempt_count = 0U;
        const size_t depth_bucket = depth_remaining < 0 ? 0U : static_cast<size_t>(depth_remaining);
        std::shared_ptr<Node> best_exact_tree = leaf_tree;
        double best_exact_objective = leaf_objective;
        size_t best_exact_idx = std::numeric_limits<size_t>::max();
        std::vector<std::shared_ptr<Node>> child_nodes;

        auto materialize_nominee = [&](NomineeEval &eval, const PreparedFeatureAtomized &prepared) -> bool {
            if (eval.materialized) {
                return !eval.child_indices.empty();
            }
            std::vector<std::vector<int>> group_atom_positions;
            std::vector<int> group_counts;
            if (!fill_groups_from_assignment(
                    eval.candidate.assignment,
                    eval.candidate.groups,
                    group_atom_positions,
                    group_counts)) {
                eval.materialized = true;
                return false;
            }
            if (group_atom_positions.empty()) {
                eval.materialized = true;
                return false;
            }
            eval.child_indices.clear();
            eval.child_stats.clear();
            eval.group_spans.clear();
            eval.child_indices.reserve(group_atom_positions.size());
            eval.child_stats.reserve(group_atom_positions.size());
            eval.group_spans.reserve(group_atom_positions.size());
            double lower_bound = 0.0;
            double upper_bound = 0.0;
            for (const auto &group_positions : group_atom_positions) {
                std::vector<int> subset_sorted;
                gather_group_members_sorted(
                    prepared.bins,
                    group_positions,
                    subset_sorted);
                if ((int)subset_sorted.size() < min_child_size_) {
                    eval.child_indices.clear();
                    eval.child_stats.clear();
                    eval.group_spans.clear();
                    eval.materialized = true;
                    return false;
                }
                SubproblemStats child_stats = compute_subproblem_stats(subset_sorted);
                const double child_lower_bound =
                    (depth_remaining == 1 ||
                     child_stats.pure ||
                     (int)subset_sorted.size() < min_split_size_)
                        ? child_stats.leaf_objective
                        : signature_bound_for_indices(subset_sorted);
                lower_bound += child_lower_bound;
                upper_bound += child_stats.leaf_objective;
                eval.child_indices.push_back(std::move(subset_sorted));
                eval.child_stats.push_back(std::move(child_stats));
                eval.group_spans.push_back(atom_positions_to_spans(prepared.bins, group_positions));
            }
            eval.lower_bound = lower_bound;
            eval.upper_bound = upper_bound;
            eval.materialized = true;
            std::vector<int>().swap(eval.candidate.assignment);
            return true;
        };

        auto exactify_nominee = [&](size_t idx) -> const ExactNomineeResult & {
            if (finalized[idx]) {
                return exact_results[idx];
            }
            NomineeEval &eval = nominee_evals[idx];
            const PreparedFeatureAtomized &prepared = prepared_by_feature[(size_t)eval.feature];
            ExactNomineeResult result;
            if (!prepared.valid) {
                finalized[idx] = 1U;
                exact_results[idx] = std::move(result);
                ++exact_evaluated_total;
                return exact_results[idx];
            }
            if (!materialize_nominee(eval, prepared)) {
                finalized[idx] = 1U;
                exact_results[idx] = std::move(result);
                ++exact_evaluated_total;
                return exact_results[idx];
            }
            if (eval.lower_bound + kEpsUpdate >= best_exact_objective) {
                finalized[idx] = 1U;
                exact_results[idx] = std::move(result);
                ++exact_evaluated_total;
                return exact_results[idx];
            }

            ++processed_candidate_count;
            double objective = 0.0;
            child_nodes.clear();
            child_nodes.reserve(eval.child_indices.size());
            bool build_ok = true;
            for (size_t group_idx = 0; group_idx < eval.child_indices.size(); ++group_idx) {
                const std::vector<int> &subset_sorted = eval.child_indices[group_idx];
                const SubproblemStats &child_stats = eval.child_stats[group_idx];
                if ((int)subset_sorted.size() < min_child_size_) {
                    build_ok = false;
                    break;
                }
                if (depth_remaining == 1 ||
                    child_stats.pure ||
                    (int)subset_sorted.size() < min_split_size_) {
                    auto [child_objective, child_tree] = leaf_solution(child_stats);
                    objective += child_objective;
                    child_nodes.push_back(child_tree);
                } else {
                    ++recurse_attempt_count;
                    ScopedTimer recursion_timer(profiling_recursive_child_eval_sec_, profiling_enabled_);
                    GreedyResult child = solve_subproblem(std::move(eval.child_indices[group_idx]), depth_remaining - 1);
                    objective += child.objective;
                    child_nodes.push_back(child.tree);
                }
            }

            if (build_ok) {
                std::shared_ptr<Node> tree = build_internal_node_from_group_spans(
                    eval.feature,
                    eval.group_spans,
                    child_nodes,
                    leaf_tree->prediction,
                    (int)indices.size());
                if (tree) {
                    result.valid = true;
                    result.objective = objective;
                    result.tree = tree;
                }
            }

            finalized[idx] = 1U;
            exact_results[idx] = std::move(result);
            ++exact_evaluated_total;
            return exact_results[idx];
        };

        const size_t exactify_limit = std::min(exactify_budget_count, alive_indices.size());
        for (size_t order = 0; order < exactify_limit; ++order) {
            const size_t idx = alive_indices[order];
            if (finalized[idx]) {
                continue;
            }
            const ExactNomineeResult &result = exactify_nominee(idx);
            if (result.valid && result.tree &&
                (!best_exact_tree || result.objective + kEpsUpdate < best_exact_objective ||
                 (best_exact_idx < nominee_evals.size() &&
                  std::abs(result.objective - best_exact_objective) <= kEpsUpdate &&
                  nominee_prefer(idx, best_exact_idx)))) {
                best_exact_objective = result.objective;
                best_exact_tree = result.tree;
                best_exact_idx = idx;
                ++incumbent_update_count;
            }
        }
        if (!exact_mode && depth_remaining > 1 && best_exact_objective + kEpsUpdate >= leaf_objective) {
            for (size_t order = exactify_limit; order < alive_indices.size(); ++order) {
                const size_t idx = alive_indices[order];
                if (finalized[idx]) {
                    continue;
                }
                const ExactNomineeResult &result = exactify_nominee(idx);
                if (!result.valid || !result.tree) {
                    continue;
                }
                if (result.objective + kEpsUpdate < leaf_objective) {
                    best_exact_objective = result.objective;
                    best_exact_tree = result.tree;
                    best_exact_idx = idx;
                    ++incumbent_update_count;
                    break;
                }
            }
        }
        const GreedyResult solved = (best_exact_tree && best_exact_objective + kEpsUpdate < leaf_objective)
            ? GreedyResult{best_exact_objective, best_exact_tree}
            : GreedyResult{leaf_objective, leaf_tree};

        auto ensure_hist_size = [&](auto &hist, auto fill_value) {
            if (hist.size() <= depth_bucket) {
                hist.resize(depth_bucket + 1U, fill_value);
            }
        };
        ensure_hist_size(telemetry.heuristic_selector_nodes_by_depth, 0LL);
        ensure_hist_size(telemetry.heuristic_selector_candidate_total_by_depth, 0LL);
        ensure_hist_size(telemetry.heuristic_selector_candidate_pruned_total_by_depth, 0LL);
        ensure_hist_size(telemetry.heuristic_selector_survivor_total_by_depth, 0LL);
        ensure_hist_size(telemetry.heuristic_selector_leaf_optimal_nodes_by_depth, 0LL);
        ensure_hist_size(telemetry.heuristic_selector_improving_split_nodes_by_depth, 0LL);
        ensure_hist_size(telemetry.heuristic_selector_improving_split_retained_nodes_by_depth, 0LL);
        const long long exactified_total = static_cast<long long>(exact_evaluated_total);
        nominee_exactified_total_ += exactified_total;
        nominee_incumbent_updates_ += static_cast<long long>(incumbent_update_count);
        telemetry.atomized_coarse_pruned_candidates += static_cast<long long>(pruned_pair_count);
        telemetry.heuristic_selector_nodes += 1;
        ++telemetry.heuristic_selector_nodes_by_depth[depth_bucket];
        telemetry.heuristic_selector_candidate_total += static_cast<long long>(candidate_evals.size());
        telemetry.heuristic_selector_candidate_pruned_total += static_cast<long long>(pruned_pair_count);
        telemetry.heuristic_selector_survivor_total += static_cast<long long>(surviving_candidate_evals.size());
        telemetry.heuristic_selector_candidate_total_by_depth[depth_bucket] +=
            static_cast<long long>(candidate_evals.size());
        telemetry.heuristic_selector_candidate_pruned_total_by_depth[depth_bucket] +=
            static_cast<long long>(pruned_pair_count);
        telemetry.heuristic_selector_survivor_total_by_depth[depth_bucket] +=
            static_cast<long long>(surviving_candidate_evals.size());

        bump_count_histogram(greedy_candidate_count_histogram_, exactified_total);

        if (best_exact_tree && best_exact_objective + kEpsUpdate < leaf_objective) {
            ++telemetry.heuristic_selector_improving_split_nodes;
            ++telemetry.heuristic_selector_improving_split_nodes_by_depth[depth_bucket];
            const double split_margin = std::max(0.0, leaf_objective - best_exact_objective);
            telemetry.heuristic_selector_improving_split_margin_sum += split_margin;
            telemetry.heuristic_selector_improving_split_margin_max = std::max(
                telemetry.heuristic_selector_improving_split_margin_max,
                split_margin);
            telemetry.heuristic_selector_improving_split_margin_sum_by_depth[depth_bucket] += split_margin;
            telemetry.heuristic_selector_improving_split_margin_max_by_depth[depth_bucket] = std::max(
                telemetry.heuristic_selector_improving_split_margin_max_by_depth[depth_bucket],
                split_margin);
            if (pruned_pair_count > 0U) {
                ++telemetry.heuristic_selector_improving_split_retained_nodes;
                ++telemetry.heuristic_selector_improving_split_retained_nodes_by_depth[depth_bucket];
            }
        } else {
            ++telemetry.heuristic_selector_leaf_optimal_nodes;
            ++telemetry.heuristic_selector_leaf_optimal_nodes_by_depth[depth_bucket];
        }

        const double best_lower_bound = best_exact_objective;
        trace_greedy_snapshot(
            "exit",
            depth_remaining,
            indices.size(),
            preserved_feature_count,
            candidate_evals.size(),
            exactified_total,
            processed_candidate_count,
            recurse_attempt_count,
            solved.objective,
            best_lower_bound,
            incumbent_update_count);

        if (solved.tree && !solved.tree->is_leaf) {
            ++greedy_internal_nodes_;
        }
        cache_store(state_key, solved, depth_remaining);
        return solved;
    }

    GreedyResult solve_subproblem(
        std::vector<int> indices,
        int depth_remaining) {
        const int current_depth = full_depth_budget_ - depth_remaining;
        if (current_depth < effective_lookahead_depth_) {
            return greedy_complete_impl(std::move(indices), depth_remaining, true);
        }
        return greedy_complete_impl(std::move(indices), depth_remaining, false);
    }
