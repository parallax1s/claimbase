"""Large-scale direction prioritization for practical philosophy.

This module is the "many claims -> high-level direction" layer.  It is meant
for questions like "should I work on AI safety, cancer, pandemic prevention, or
animal welfare?" where the input may be thousands or millions of extracted
claims, assumptions, and evidence signals.

The implementation is deliberately streaming-friendly: callers can pass any
``Iterable[ClaimSignal]`` including a JSONL reader.  The in-memory state is
proportional to ``direction × factor × dependency_group`` rather than to the
number of raw claims, so repeated claims from one source can be capped instead
of swamping the result.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from decision.models import (
    BeneficiaryScope,
    Constraint,
    ConstraintSeverity,
    CriterionDirection,
    Crux,
    Estimate,
    ObjectiveProfile,
    OptionStatus,
    Proposition,
    RiskAttitude,
)


class ClaimPolarity(str, Enum):
    """Whether a claim pushes a factor up or down for a direction."""

    SUPPORTS = "supports"
    WEAKENS = "weakens"


class PrioritizationModel(str, Enum):
    """How factor scores are combined into a direction score."""

    ADDITIVE = "additive"
    GEOMETRIC = "geometric"
    HYBRID = "hybrid"


class Direction(BaseModel):
    """A large-scale candidate direction, cause area, agenda, or life path."""

    id: str
    name: str
    description: str = ""
    base_estimates: dict[str, Estimate] = Field(default_factory=dict)
    constraints: list[Constraint] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("direction id cannot be empty")
        return value


class PrioritizationFactor(BaseModel):
    """A high-level dimension used to rank directions.

    Examples: problem scale, neglectedness, tractability, personal fit,
    contribution leverage, moral scope, downside risk, option value.
    """

    id: str
    name: str
    description: str = ""
    weight: float = Field(default=1.0, ge=0.0)
    direction: CriterionDirection = CriterionDirection.BENEFIT
    scope_weights: dict[BeneficiaryScope, float] = Field(default_factory=dict)
    prior: Estimate = Field(
        default_factory=lambda: Estimate(expected=0.5, low=0.25, high=0.75, confidence=0.1)
    )
    prior_weight: float = Field(default=1.0, ge=0.0)
    geometric: bool = True
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("factor id cannot be empty")
        return value

    def effective_weight(self, scope: BeneficiaryScope) -> float:
        return self.weight * self.scope_weights.get(scope, 1.0)


class ClaimSignal(BaseModel):
    """A directional evidence signal derived from a claim or assumption.

    This is intentionally smaller than a full claim object.  It says how one
    claim bears on one factor for one direction.  A single proposition can emit
    several signals, for example one about AI safety scale and another about
    tractability.
    """

    id: str
    direction_id: str
    factor_id: str
    polarity: ClaimPolarity = ClaimPolarity.SUPPORTS
    credence: float = Field(default=0.5, ge=0.0, le=1.0)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    extraction_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    source_weight: float = Field(default=1.0, ge=0.0)
    dependency_key: str | None = None
    proposition_id: str | None = None
    source_id: str | None = None
    text: str = ""
    assumptions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "direction_id", "factor_id")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("id fields cannot be empty")
        return value

    @property
    def grouping_key(self) -> str:
        """Return a dependency key used for duplicate/correlation caps."""

        # Fall back to a SHARED sentinel, not the unique signal id, so a flood
        # of un-keyed signals for one direction/factor collapses into a single
        # capped group instead of evading the dependency cap entirely.
        return self.dependency_key or self.source_id or self.proposition_id or "__ungrouped__"

    @property
    def effective_weight(self) -> float:
        """Evidence weight before dependency capping.

        The formula is intentionally transparent rather than magical: source
        weight × relevance × extraction confidence × evidence quality, with a
        small floor for low-strength claims so weak claims can still register as
        weak evidence.
        """

        return (
            self.source_weight
            * self.relevance
            * self.extraction_confidence
            * self.evidence_quality
            * max(0.05, self.strength)
        )

    @property
    def signal_value(self) -> float:
        """Map polarity/credence/strength to a bounded factor value.

        0.5 is neutral.  A fully credible, fully strong supporting signal maps
        to 1.0; a fully credible weakening signal maps to 0.0.  A disbelieved
        claim pushes in the opposite direction.
        """

        sign = 1.0 if self.polarity == ClaimPolarity.SUPPORTS else -1.0
        return _clamp01(0.5 + sign * (self.credence - 0.5) * self.strength)


class ScaleConfig(BaseModel):
    """Scalability/aggregation knobs."""

    dependency_cap: float = Field(default=3.0, gt=0.0)
    evidence_saturation_weight: float = Field(default=12.0, gt=0.0)
    min_signal_weight: float = Field(default=1e-9, ge=0.0)
    geometric_floor: float = Field(default=0.02, gt=0.0, le=0.5)
    retain_group_count: bool = True


class PrioritizationQuestion(BaseModel):
    prompt: str
    directions: list[Direction]
    factors: list[PrioritizationFactor]
    objective: ObjectiveProfile = Field(default_factory=ObjectiveProfile)
    model: PrioritizationModel = PrioritizationModel.HYBRID
    missing_estimate_default: Estimate = Field(
        default_factory=lambda: Estimate(expected=0.5, low=0.25, high=0.75, confidence=0.1)
    )
    assumptions: list[str] = Field(default_factory=list)
    horizon: str = "unspecified"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> PrioritizationQuestion:
        direction_ids = [direction.id for direction in self.directions]
        if len(direction_ids) != len(set(direction_ids)):
            raise ValueError("direction ids must be unique")
        factor_ids = [factor.id for factor in self.factors]
        if len(factor_ids) != len(set(factor_ids)):
            raise ValueError("factor ids must be unique")
        return self


class FactorAggregate(BaseModel):
    direction_id: str
    factor_id: str
    factor_name: str
    estimate: Estimate
    adjusted_score: float = Field(ge=0.0, le=1.0)
    contribution: float
    evidence_count: int = 0
    effective_weight: float = 0.0
    dependency_group_count: int = 0
    missing: bool = False


class DirectionEvaluation(BaseModel):
    direction_id: str
    direction_name: str
    status: OptionStatus
    expected_return_index: float
    normalized_score: float = Field(ge=0.0, le=1.0)
    additive_score: float = Field(ge=0.0, le=1.0)
    geometric_score: float = Field(ge=0.0, le=1.0)
    robust_score: float = Field(ge=0.0, le=1.0)
    regret: float = 0.0
    evidence_count: int = 0
    effective_evidence_weight: float = 0.0
    dependency_group_count: int = 0
    factor_scores: list[FactorAggregate] = Field(default_factory=list)
    hard_constraint_failures: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PrioritizationResult(BaseModel):
    prompt: str
    scope: BeneficiaryScope
    risk_attitude: RiskAttitude
    recommended_direction_id: str | None
    ranked_directions: list[DirectionEvaluation]
    cruxes: list[Crux] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def recommended_direction(self) -> DirectionEvaluation | None:
        if self.recommended_direction_id is None:
            return None
        for direction in self.ranked_directions:
            if direction.direction_id == self.recommended_direction_id:
                return direction
        return None


class _Accumulator:
    __slots__ = ("count", "sum_value", "sum_weight", "sum_weighted_sq")

    def __init__(self) -> None:
        self.count = 0
        self.sum_weight = 0.0
        self.sum_value = 0.0
        self.sum_weighted_sq = 0.0

    def add(self, value: float, weight: float) -> None:
        if weight <= 0:
            return
        self.count += 1
        self.sum_weight += weight
        self.sum_value += value * weight
        self.sum_weighted_sq += value * value * weight

    @property
    def mean(self) -> float:
        if self.sum_weight <= 0:
            return 0.5
        return self.sum_value / self.sum_weight

    @property
    def variance(self) -> float:
        if self.sum_weight <= 0:
            return 0.0
        mean = self.mean
        return max(0.0, self.sum_weighted_sq / self.sum_weight - mean * mean)


class DirectionPrioritizer:
    """Aggregate many evidence signals into ranked high-level directions."""

    def __init__(self, config: ScaleConfig | None = None) -> None:
        self.config = config or ScaleConfig()

    def evaluate(
        self,
        question: PrioritizationQuestion,
        signals: Iterable[ClaimSignal] = (),
    ) -> PrioritizationResult:
        factor_map = {factor.id: factor for factor in question.factors}
        direction_map = {direction.id: direction for direction in question.directions}
        warnings: list[str] = []
        unknown_directions: set[str] = set()
        unknown_factors: set[str] = set()
        signal_count = 0

        grouped: dict[tuple[str, str, str], _Accumulator] = {}
        raw_signal_counts: dict[tuple[str, str], int] = defaultdict(int)
        for signal in signals:
            signal_count += 1
            if signal.direction_id not in direction_map:
                unknown_directions.add(signal.direction_id)
                continue
            if signal.factor_id not in factor_map:
                unknown_factors.add(signal.factor_id)
                continue
            weight = signal.effective_weight
            if weight < self.config.min_signal_weight:
                continue
            raw_signal_counts[(signal.direction_id, signal.factor_id)] += 1
            group_key = (signal.direction_id, signal.factor_id, signal.grouping_key)
            acc = grouped.get(group_key)
            if acc is None:
                acc = _Accumulator()
                grouped[group_key] = acc
            acc.add(signal.signal_value, weight)

        if unknown_directions:
            warnings.append(f"Ignored signals for unknown directions: {sorted(unknown_directions)}")
        if unknown_factors:
            warnings.append(f"Ignored signals for unknown factors: {sorted(unknown_factors)}")

        factor_accs: dict[tuple[str, str], _Accumulator] = {}
        group_counts: dict[tuple[str, str], int] = defaultdict(int)
        for (direction_id, factor_id, _dependency_key), group_acc in grouped.items():
            if group_acc.sum_weight <= 0:
                continue
            capped_weight = min(group_acc.sum_weight, self.config.dependency_cap)
            factor_key = (direction_id, factor_id)
            acc = factor_accs.get(factor_key)
            if acc is None:
                acc = _Accumulator()
                factor_accs[factor_key] = acc
            acc.add(group_acc.mean, capped_weight)
            group_counts[factor_key] += 1

        raw_evaluations = [
            self._evaluate_direction(
                direction, question, factor_accs, group_counts, raw_signal_counts
            )
            for direction in question.directions
        ]
        best_score = max(
            (
                ev.normalized_score
                for ev in raw_evaluations
                if ev.status != OptionStatus.IMPERMISSIBLE
            ),
            default=None,
        )
        best_return = max(
            (
                ev.expected_return_index
                for ev in raw_evaluations
                if ev.status != OptionStatus.IMPERMISSIBLE
            ),
            default=0.0,
        )

        evaluations: list[DirectionEvaluation] = []
        for ev in raw_evaluations:
            status = ev.status
            notes = list(ev.notes)
            regret = 0.0
            if status != OptionStatus.IMPERMISSIBLE and best_score is not None:
                regret = max(0.0, best_score - ev.normalized_score)
                if abs(ev.normalized_score - best_score) < 1e-12:
                    status = OptionStatus.RECOMMENDED
                elif regret < 0.04:
                    status = OptionStatus.INVESTIGATE_FIRST
                    notes.append(
                        "Near-tie at direction level; prioritize crux resolution before locking in."
                    )
                elif ev.robust_score < 0.30 and regret < 0.12:
                    status = OptionStatus.TOO_RISKY
                    notes.append("High upside but weak robust score under current uncertainty.")
                elif regret > 0.20:
                    status = OptionStatus.DOMINATED
            relative_return = ev.expected_return_index / best_return if best_return > 0 else 0.0
            evaluations.append(
                ev.model_copy(
                    update={
                        "regret": regret,
                        "status": status,
                        "notes": notes,
                        "expected_return_index": relative_return,
                    }
                )
            )

        ranked = sorted(
            evaluations,
            key=lambda ev: (
                ev.status == OptionStatus.IMPERMISSIBLE,
                -ev.normalized_score,
                -ev.robust_score,
            ),
        )
        recommended = next(
            (ev.direction_id for ev in ranked if ev.status == OptionStatus.RECOMMENDED), None
        )
        if recommended is None:
            recommended = next(
                (ev.direction_id for ev in ranked if ev.status != OptionStatus.IMPERMISSIBLE), None
            )

        return PrioritizationResult(
            prompt=question.prompt,
            scope=question.objective.scope,
            risk_attitude=question.objective.risk_attitude,
            recommended_direction_id=recommended,
            ranked_directions=ranked,
            cruxes=self._cruxes(ranked, question),
            warnings=warnings,
            metadata={
                "engine": "episteme.decision.DirectionPrioritizer",
                "model": question.model.value,
                "signal_count": signal_count,
                "dependency_group_count": len(grouped),
                "direction_count": len(question.directions),
                "factor_count": len(question.factors),
                "dependency_cap": self.config.dependency_cap,
                "scale_note": (
                    "Streaming aggregation; memory is proportional to "
                    "direction×factor×dependency groups."
                ),
            },
        )

    def _evaluate_direction(
        self,
        direction: Direction,
        question: PrioritizationQuestion,
        factor_accs: dict[tuple[str, str], _Accumulator],
        group_counts: dict[tuple[str, str], int],
        raw_signal_counts: dict[tuple[str, str], int],
    ) -> DirectionEvaluation:
        hard_failures = [
            c.description
            for c in direction.constraints
            if c.severity == ConstraintSeverity.HARD and not c.satisfied
        ]
        factor_scores: list[FactorAggregate] = []
        total_weight = 0.0
        additive_sum = 0.0
        robust_sum = 0.0
        log_sum = 0.0
        log_weight = 0.0
        arith_sum = 0.0
        arith_weight = 0.0
        total_evidence_count = 0
        total_evidence_weight = 0.0
        total_dependency_groups = 0

        for factor in question.factors:
            weight = factor.effective_weight(question.objective.scope)
            if question.objective.beneficiary_weights:
                weight *= question.objective.beneficiary_weights.get(question.objective.scope, 1.0)
            if weight <= 0:
                continue
            estimate, evidence_count, evidence_weight, dependency_group_count, missing = (
                self._factor_estimate(
                    direction,
                    factor,
                    question,
                    factor_accs.get((direction.id, factor.id)),
                    group_counts.get((direction.id, factor.id), 0),
                    raw_signal_counts.get((direction.id, factor.id), 0),
                )
            )
            # For COST factors, invert the estimate FIRST and then risk-adjust,
            # so risk attitude acts on the desirability framing (low cost = good).
            # Negating after risk_adjusted would flip the sign of the risk
            # penalty/bonus (e.g. risk-aversion making a big cost look better).
            is_cost = factor.direction == CriterionDirection.COST
            scored = estimate.inverted() if is_cost else estimate
            adjusted = _clamp01(scored.risk_adjusted(question.objective.risk_attitude))
            robust = _clamp01(scored.low_value)
            contribution = weight * adjusted
            factor_scores.append(
                FactorAggregate(
                    direction_id=direction.id,
                    factor_id=factor.id,
                    factor_name=factor.name,
                    estimate=estimate,
                    adjusted_score=adjusted,
                    contribution=contribution,
                    evidence_count=evidence_count,
                    effective_weight=evidence_weight,
                    dependency_group_count=dependency_group_count,
                    missing=missing,
                )
            )
            additive_sum += contribution
            robust_sum += weight * robust
            total_weight += weight
            total_evidence_count += evidence_count
            total_evidence_weight += evidence_weight
            total_dependency_groups += dependency_group_count
            if factor.geometric:
                log_sum += weight * math.log(max(self.config.geometric_floor, adjusted))
                log_weight += weight
            else:
                # geometric=False factors combine additively rather than
                # multiplicatively, but must still enter the geometric branch
                # instead of silently vanishing from it.
                arith_sum += contribution  # weight * adjusted
                arith_weight += weight

        additive_score = additive_sum / total_weight if total_weight else 0.0
        robust_score = robust_sum / total_weight if total_weight else 0.0
        # Geometric (ITN-style) factors combine as a weighted geometric mean; any
        # geometric=False factors fold in by their weighted arithmetic mean so a
        # near-zero multiplicative factor still tanks the score while additive
        # factors keep their influence (and never silently disappear).
        if log_weight:
            geo_core = math.exp(log_sum / log_weight)
            if arith_weight:
                geometric_score = (log_weight * geo_core + arith_sum) / (log_weight + arith_weight)
            else:
                geometric_score = geo_core
        elif arith_weight:
            geometric_score = arith_sum / arith_weight
        else:
            geometric_score = additive_score
        if question.model == PrioritizationModel.ADDITIVE:
            score = additive_score
        elif question.model == PrioritizationModel.GEOMETRIC:
            score = geometric_score
        else:
            score = 0.35 * additive_score + 0.65 * geometric_score
        score = _clamp01(score)
        status = OptionStatus.PERMISSIBLE
        notes: list[str] = []
        if hard_failures:
            score = 0.0
            robust_score = 0.0
            geometric_score = 0.0
            additive_score = 0.0
            status = OptionStatus.IMPERMISSIBLE
            notes.append("Excluded by hard constraints.")
        if any(item.missing for item in factor_scores):
            missing_names = ", ".join(item.factor_name for item in factor_scores if item.missing)
            notes.append(f"Missing large-scale evidence defaulted for: {missing_names}.")
        if total_evidence_count == 0:
            notes.append("No claim signals supplied; ranking uses priors/base estimates only.")

        return DirectionEvaluation(
            direction_id=direction.id,
            direction_name=direction.name,
            status=status,
            expected_return_index=score,
            normalized_score=score,
            additive_score=additive_score,
            geometric_score=geometric_score,
            robust_score=_clamp01(robust_score),
            evidence_count=total_evidence_count,
            effective_evidence_weight=total_evidence_weight,
            dependency_group_count=total_dependency_groups,
            factor_scores=factor_scores,
            hard_constraint_failures=hard_failures,
            notes=notes,
        )

    def _factor_estimate(
        self,
        direction: Direction,
        factor: PrioritizationFactor,
        question: PrioritizationQuestion,
        evidence_acc: _Accumulator | None,
        dependency_group_count: int,
        raw_signal_count: int,
    ) -> tuple[Estimate, int, float, int, bool]:
        prior = direction.base_estimates.get(
            factor.id, factor.prior or question.missing_estimate_default
        )
        prior_weight = factor.prior_weight * max(0.05, prior.confidence)
        combined = _Accumulator()
        if prior_weight > 0:
            combined.add(prior.expected, prior_weight)
        evidence_count = 0
        evidence_weight = 0.0
        if evidence_acc is not None and evidence_acc.sum_weight > 0:
            combined.add(evidence_acc.mean, evidence_acc.sum_weight)
            evidence_count = raw_signal_count
            evidence_weight = evidence_acc.sum_weight
        if combined.sum_weight <= 0:
            return question.missing_estimate_default, 0, 0.0, 0, True

        mean = _clamp01(combined.mean)
        evidence_conf = evidence_weight / (evidence_weight + self.config.evidence_saturation_weight)
        prior_conf = prior.confidence * (prior_weight / (prior_weight + evidence_weight + 1e-12))
        confidence = _clamp01(max(evidence_conf, prior_conf))
        variance = max(combined.variance, evidence_acc.variance if evidence_acc else 0.0)
        spread = min(0.95, math.sqrt(variance) + 0.45 * (1.0 - confidence))
        low = _clamp01(mean - spread / 2.0)
        high = _clamp01(mean + spread / 2.0)
        estimate = Estimate(expected=mean, low=low, high=high, confidence=confidence)
        missing = evidence_count == 0 and factor.id not in direction.base_estimates
        return estimate, evidence_count, evidence_weight, dependency_group_count, missing

    def _cruxes(
        self, ranked: list[DirectionEvaluation], question: PrioritizationQuestion
    ) -> list[Crux]:
        viable = [ev for ev in ranked if ev.status != OptionStatus.IMPERMISSIBLE]
        if not viable:
            return [Crux(description="All directions violate hard constraints.")]
        if len(viable) == 1:
            return [
                Crux(
                    description="Only one permissible direction was supplied.",
                    option_id=viable[0].direction_id,
                )
            ]
        top, runner = viable[0], viable[1]
        gap = max(0.0, top.normalized_score - runner.normalized_score)
        top_factors = {factor.factor_id: factor for factor in top.factor_scores}
        runner_factors = {factor.factor_id: factor for factor in runner.factor_scores}
        total_weight = (
            sum(f.effective_weight(question.objective.scope) for f in question.factors) or 1.0
        )
        cruxes: list[Crux] = []
        for factor in question.factors:
            a = top_factors.get(factor.id)
            b = runner_factors.get(factor.id)
            if a is None or b is None:
                continue
            delta = (a.contribution - b.contribution) / total_weight
            uncertainty = max(a.estimate.width, b.estimate.width)
            evidence_gap = abs(a.effective_weight - b.effective_weight)
            voi = factor.effective_weight(question.objective.scope) / total_weight
            voi *= uncertainty * (0.7 + min(1.0, evidence_gap / 10.0))
            if abs(delta) >= max(0.01, gap * 0.20) or voi >= 0.04:
                cruxes.append(
                    Crux(
                        description=(
                            f"{factor.name} changes {top.direction_name} "
                            f"vs {runner.direction_name} by about {delta:+.3f}."
                        ),
                        option_id=top.direction_id,
                        criterion_id=factor.id,
                        current_gap=gap,
                        flip_threshold=min(
                            1.0,
                            gap
                            / max(
                                1e-9,
                                factor.effective_weight(question.objective.scope) / total_weight,
                            ),
                        ),
                        value_of_information=voi,
                        investigation=(
                            f"Add or verify high-quality claims about {factor.name.lower()} for "
                            f"{top.direction_name} and {runner.direction_name}; this is a "
                            "likely ranking crux."
                        ),
                    )
                )
        cruxes.sort(key=lambda item: item.value_of_information, reverse=True)
        if not cruxes:
            cruxes.append(
                Crux(
                    description=(
                        "The leading direction is supported by a broad bundle rather "
                        "than one obvious crux."
                    ),
                    current_gap=gap,
                    investigation=(
                        "Run sensitivity by beneficiary scope and add evidence to the "
                        "highest-uncertainty factors."
                    ),
                )
            )
        return cruxes[:10]


def claim_signal_from_proposition(
    proposition: Proposition,
    *,
    direction_id: str,
    factor_id: str,
    polarity: ClaimPolarity = ClaimPolarity.SUPPORTS,
    strength: float = 0.5,
    relevance: float = 1.0,
    evidence_quality: float | None = None,
    source_weight: float = 1.0,
    dependency_key: str | None = None,
) -> ClaimSignal:
    """Create a directional signal from a decision-ready proposition."""

    return ClaimSignal(
        id=f"signal_{proposition.id}_{direction_id}_{factor_id}",
        direction_id=direction_id,
        factor_id=factor_id,
        polarity=polarity,
        credence=proposition.credence if proposition.credence is not None else 0.5,
        strength=strength,
        relevance=relevance,
        extraction_confidence=proposition.extraction_confidence,
        evidence_quality=evidence_quality
        if evidence_quality is not None
        else _quality_from_proposition(proposition),
        source_weight=source_weight,
        dependency_key=dependency_key or proposition.source_claim_id or proposition.id,
        proposition_id=proposition.id,
        source_id=proposition.source_node_id,
        text=proposition.text,
        metadata={"claim_type": proposition.claim_type, "roles": proposition.decision_roles},
    )


def iter_claim_signals_jsonl(path: str | Path) -> Iterator[ClaimSignal]:
    """Yield ``ClaimSignal`` records from a JSONL file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                yield ClaimSignal.model_validate(payload)
            except Exception as exc:  # pragma: no cover - defensive CLI helper
                raise ValueError(f"invalid ClaimSignal JSONL at line {line_number}: {exc}") from exc


def write_claim_signals_jsonl(path: str | Path, signals: Iterable[ClaimSignal]) -> None:
    """Write claim signals as JSONL without materializing them all at once."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for signal in signals:
            handle.write(json.dumps(signal.model_dump(mode="json"), ensure_ascii=False) + "\n")


def _quality_from_proposition(proposition: Proposition) -> float:
    support = proposition.support_in_text if proposition.support_in_text is not None else 0.5
    interval_width = 0.5
    if proposition.credence_low is not None and proposition.credence_high is not None:
        interval_width = proposition.credence_high - proposition.credence_low
    return _clamp01(0.35 + 0.35 * support + 0.30 * (1.0 - interval_width))


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
