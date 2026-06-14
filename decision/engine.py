"""Decision-theory engine for expected-value, MCDA, robustness, and VOI."""

from __future__ import annotations

from collections.abc import Iterable

from decision.models import (
    BeneficiaryScope,
    ConstraintSeverity,
    Criterion,
    CriterionContribution,
    CriterionDirection,
    Crux,
    DecisionQuestion,
    DecisionResult,
    ObjectiveProfile,
    Option,
    OptionEvaluation,
    OptionStatus,
)


class DecisionEngine:
    """Evaluate practical options under explicit assumptions and value weights."""

    def evaluate(self, question: DecisionQuestion) -> DecisionResult:
        objective = question.objective
        warnings: list[str] = []
        criterion_map = {criterion.id: criterion for criterion in question.criteria}
        if not criterion_map and not any(option.outcomes for option in question.options):
            warnings.append(
                "No criteria or outcome branches supplied; all options are effectively tied."
            )

        raw_evaluations = [
            self._evaluate_option(option, question, criterion_map) for option in question.options
        ]
        best_total = max(
            (ev.total_score for ev in raw_evaluations if ev.status != OptionStatus.IMPERMISSIBLE),
            default=None,
        )
        best_robust = max(
            (ev.robust_score for ev in raw_evaluations if ev.status != OptionStatus.IMPERMISSIBLE),
            default=None,
        )

        evaluations: list[OptionEvaluation] = []
        for ev in raw_evaluations:
            regret = (
                0.0
                if best_total is None or ev.status == OptionStatus.IMPERMISSIBLE
                else max(0.0, best_total - ev.total_score)
            )
            status = ev.status
            notes = list(ev.notes)
            if status != OptionStatus.IMPERMISSIBLE and best_total is not None:
                if abs(ev.total_score - best_total) < 1e-12:
                    status = OptionStatus.RECOMMENDED
                elif best_robust is not None and ev.robust_score < 0.35 and regret < 0.08:
                    status = OptionStatus.TOO_RISKY
                    notes.append("Close on expected score, but weak on robust/worst-case score.")
                elif regret < 0.05 and any(c.estimate.width > 0.25 for c in ev.contributions):
                    status = OptionStatus.INVESTIGATE_FIRST
                    notes.append(
                        "Close runner with material uncertainty; investigate before treating "
                        "the ranking as settled."
                    )
                elif regret > 0.20 and ev.robust_score <= (best_robust or 0.0) + 0.02:
                    status = OptionStatus.DOMINATED
            evaluations.append(
                ev.model_copy(update={"regret": regret, "status": status, "notes": notes})
            )

        ranked = sorted(
            evaluations,
            key=lambda ev: (
                ev.status == OptionStatus.IMPERMISSIBLE,
                -ev.total_score,
                -ev.robust_score,
            ),
        )
        recommended = next(
            (ev.option_id for ev in ranked if ev.status == OptionStatus.RECOMMENDED), None
        )
        if recommended is None:
            recommended = next(
                (ev.option_id for ev in ranked if ev.status != OptionStatus.IMPERMISSIBLE), None
            )

        cruxes = self._cruxes(ranked, question)
        return DecisionResult(
            prompt=question.prompt,
            scope=objective.scope,
            risk_attitude=objective.risk_attitude,
            recommended_option_id=recommended,
            ranked_options=ranked,
            cruxes=cruxes,
            warnings=warnings,
            metadata={
                "engine": "episteme.decision.DecisionEngine",
                "outcome_mix_weight": question.outcome_mix_weight,
                "criterion_count": len(question.criteria),
                "option_count": len(question.options),
            },
        )

    def evaluate_scopes(
        self,
        question: DecisionQuestion,
        scopes: Iterable[BeneficiaryScope],
    ) -> dict[BeneficiaryScope, DecisionResult]:
        results: dict[BeneficiaryScope, DecisionResult] = {}
        for scope in scopes:
            scoped = question.model_copy(
                update={"objective": question.objective.model_copy(update={"scope": scope})},
                deep=True,
            )
            results[scope] = self.evaluate(scoped)
        return results

    def _evaluate_option(
        self,
        option: Option,
        question: DecisionQuestion,
        criterion_map: dict[str, Criterion],
    ) -> OptionEvaluation:
        hard_failures = [
            c.description
            for c in option.constraints
            if c.severity == ConstraintSeverity.HARD and not c.satisfied
        ]
        soft_penalty = sum(
            c.penalty
            for c in option.constraints
            if c.severity == ConstraintSeverity.SOFT and not c.satisfied
        )
        soft_penalty = min(1.0, soft_penalty)
        contributions: list[CriterionContribution] = []
        total_weight = 0.0
        weighted_score = 0.0
        weighted_robust = 0.0

        for criterion in question.criteria:
            effective_weight = self._effective_weight(criterion, question.objective)
            if effective_weight <= 0:
                continue
            estimate = option.criterion_estimates.get(
                criterion.id, question.missing_estimate_default
            )
            missing = criterion.id not in option.criterion_estimates
            if criterion.direction == CriterionDirection.COST:
                estimate = estimate.inverted()
            score = estimate.risk_adjusted(question.objective.risk_attitude)
            robust_score = estimate.low_value
            contribution = effective_weight * score
            contributions.append(
                CriterionContribution(
                    criterion_id=criterion.id,
                    criterion_name=criterion.name,
                    weight=effective_weight,
                    estimate=estimate,
                    score=score,
                    contribution=contribution,
                    missing=missing,
                )
            )
            weighted_score += contribution
            weighted_robust += effective_weight * robust_score
            total_weight += effective_weight

        criteria_score = weighted_score / total_weight if total_weight else 0.0
        robust_score = weighted_robust / total_weight if total_weight else 0.0
        outcome_ev = self._outcome_expected_value(option, question.objective)
        if option.outcomes and total_weight == 0:
            base_score = outcome_ev or 0.0
        elif option.outcomes:
            mix = question.outcome_mix_weight
            base_score = (1.0 - mix) * criteria_score + mix * (outcome_ev or 0.0)
        else:
            base_score = criteria_score

        exploration_bonus = 0.0
        if question.objective.include_exploration_value:
            exploration_bonus = 0.05 * option.exploration_value.risk_adjusted(
                question.objective.risk_attitude
            )
        total_score = _clamp01(base_score + exploration_bonus - soft_penalty)
        robust_score = _clamp01(robust_score - soft_penalty)
        status = OptionStatus.PERMISSIBLE
        notes: list[str] = []
        if hard_failures:
            status = OptionStatus.IMPERMISSIBLE
            total_score = 0.0
            robust_score = 0.0
            notes.append("Excluded by hard constraints.")
        if any(c.missing for c in contributions):
            missing_names = ", ".join(c.criterion_name for c in contributions if c.missing)
            notes.append(f"Missing estimates defaulted for: {missing_names}.")

        return OptionEvaluation(
            option_id=option.id,
            option_name=option.name,
            status=status,
            total_score=total_score,
            criteria_score=criteria_score,
            robust_score=robust_score,
            outcome_expected_value=outcome_ev,
            exploration_bonus=exploration_bonus,
            soft_penalty=soft_penalty,
            hard_constraint_failures=hard_failures,
            contributions=contributions,
            notes=notes,
        )

    def _effective_weight(self, criterion: Criterion, objective: ObjectiveProfile) -> float:
        weight = criterion.effective_weight(objective.scope)
        # Optional explicit beneficiary weights can dampen or amplify broad scopes.
        if objective.beneficiary_weights:
            weight *= objective.beneficiary_weights.get(objective.scope, 1.0)
        return weight

    def _outcome_expected_value(self, option: Option, objective: ObjectiveProfile) -> float | None:
        if not option.outcomes:
            return None
        total = 0.0
        probability_mass = 0.0
        for outcome in option.outcomes:
            p = outcome.probability.expected
            if outcome.scope is not None and outcome.scope != objective.scope:
                p *= objective.beneficiary_weights.get(outcome.scope, 0.5)
            value = outcome.value.risk_adjusted(objective.risk_attitude)
            total += p * value
            probability_mass += p
        # Keep this normalized for display and mixing.  Branches need not be exhaustive.
        if probability_mass > 1.0:
            total /= probability_mass
        return _clamp01(total)

    def _cruxes(self, ranked: list[OptionEvaluation], question: DecisionQuestion) -> list[Crux]:
        viable = [ev for ev in ranked if ev.status != OptionStatus.IMPERMISSIBLE]
        if not viable:
            return [
                Crux(
                    description="All options violate hard constraints.",
                    investigation="Relax constraints or add new options.",
                )
            ]
        if len(viable) == 1:
            return [
                Crux(
                    description="Only one permissible option was supplied.",
                    option_id=viable[0].option_id,
                )
            ]
        top, runner = viable[0], viable[1]
        gap = max(0.0, top.total_score - runner.total_score)
        top_by_criterion = {c.criterion_id: c for c in top.contributions}
        runner_by_criterion = {c.criterion_id: c for c in runner.contributions}
        cruxes: list[Crux] = []
        total_weight = sum(c.weight for c in top.contributions) or 1.0
        for criterion in question.criteria:
            a = top_by_criterion.get(criterion.id)
            b = runner_by_criterion.get(criterion.id)
            if a is None or b is None or a.weight <= 0:
                continue
            contribution_gap = (a.contribution - b.contribution) / total_weight
            uncertainty = max(a.estimate.width, b.estimate.width)
            confidence_drag = 1.0 - min(a.estimate.confidence, b.estimate.confidence)
            voi = abs(a.weight / total_weight) * uncertainty * (0.5 + confidence_drag)
            if abs(contribution_gap) >= max(0.015, gap * 0.25) or voi >= 0.05:
                flip_threshold = None
                if a.weight > 0:
                    flip_threshold = min(1.0, gap * total_weight / a.weight) if gap > 0 else 0.0
                direction = "supports" if contribution_gap >= 0 else "weakens"
                cruxes.append(
                    Crux(
                        description=(
                            f"{criterion.name} {direction} {top.option_name} "
                            f"over {runner.option_name} by about {contribution_gap:+.3f}."
                        ),
                        option_id=top.option_id,
                        criterion_id=criterion.id,
                        current_gap=gap,
                        flip_threshold=flip_threshold,
                        value_of_information=voi,
                        investigation=(
                            f"Test or research {criterion.name.lower()} for both leading options; "
                            "this is likely to change the ranking if the estimate moves enough."
                        ),
                    )
                )
        if not cruxes:
            cruxes.append(
                Crux(
                    description=(
                        "The top option is not driven by a single dominant crux under "
                        "the current weights."
                    ),
                    current_gap=gap,
                    investigation=(
                        "Review value weights, then run a cheap test for the "
                        "highest-uncertainty criterion."
                    ),
                )
            )
        cruxes.sort(key=lambda c: c.value_of_information, reverse=True)
        return cruxes[:8]


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
