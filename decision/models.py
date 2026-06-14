"""Typed decision-analysis contracts for Episteme.

The decision layer deliberately keeps epistemic confidence separate from decision
credence/value judgments. Inputs may be hand-authored, generated from extracted
claims, or assembled from templates such as the 80,000 Hours career framework.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class BeneficiaryScope(str, Enum):
    SELF = "self"
    FAMILY = "family"
    LOCAL_GROUP = "local_group"
    ORGANIZATION = "organization"
    COUNTRY = "country"
    HUMANITY = "humanity"
    FUTURE_GENERATIONS = "future_generations"
    ALL_SENTIENT_LIFE = "all_sentient_life"
    CUSTOM = "custom"


class RiskAttitude(str, Enum):
    RISK_AVERSE = "risk_averse"
    RISK_NEUTRAL = "risk_neutral"
    RISK_SEEKING = "risk_seeking"


class CriterionDirection(str, Enum):
    BENEFIT = "benefit"
    COST = "cost"


class ConstraintSeverity(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class OptionStatus(str, Enum):
    RECOMMENDED = "recommended"
    PERMISSIBLE = "permissible"
    DOMINATED = "dominated"
    INVESTIGATE_FIRST = "investigate_first"
    TOO_RISKY = "too_risky"
    IMPERMISSIBLE = "impermissible"


class Estimate(BaseModel):
    """A bounded uncertain estimate.

    Values are usually normalized to 0..1.  ``confidence`` is meta-confidence in
    the estimate, not the probability that the estimate is true.
    """

    expected: float = Field(ge=0.0, le=1.0)
    low: float | None = Field(default=None, ge=0.0, le=1.0)
    high: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    notes: str = ""

    @model_validator(mode="after")
    def _fill_and_order_bounds(self) -> Estimate:
        low = self.expected if self.low is None else self.low
        high = self.expected if self.high is None else self.high
        ordered = sorted([low, self.expected, high])
        self.low = ordered[0]
        self.expected = ordered[1]
        self.high = ordered[2]
        return self

    @property
    def low_value(self) -> float:
        return self.expected if self.low is None else self.low

    @property
    def high_value(self) -> float:
        return self.expected if self.high is None else self.high

    @property
    def width(self) -> float:
        return self.high_value - self.low_value

    def inverted(self) -> Estimate:
        return Estimate(
            low=1.0 - self.high_value,
            expected=1.0 - self.expected,
            high=1.0 - self.low_value,
            confidence=self.confidence,
            notes=self.notes,
        )

    def risk_adjusted(self, attitude: RiskAttitude) -> float:
        low = self.low_value
        high = self.high_value
        width = high - low
        if attitude == RiskAttitude.RISK_AVERSE:
            # Penalize both spread and low meta-confidence.  This is intentionally
            # mild so it surfaces uncertainty without swamping the stated EV.
            penalty = width * (0.15 + 0.35 * (1.0 - self.confidence))
            return _clamp01(self.expected - penalty)
        if attitude == RiskAttitude.RISK_SEEKING:
            bonus = max(0.0, high - self.expected) * (0.10 + 0.20 * self.confidence)
            return _clamp01(self.expected + bonus)
        return self.expected

    @classmethod
    def point(cls, value: float, *, confidence: float = 0.8, notes: str = "") -> Estimate:
        return cls(expected=value, low=value, high=value, confidence=confidence, notes=notes)


class Criterion(BaseModel):
    id: str
    name: str
    description: str = ""
    weight: float = Field(default=1.0, ge=0.0)
    direction: CriterionDirection = CriterionDirection.BENEFIT
    scope_weights: dict[BeneficiaryScope, float] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("criterion id cannot be empty")
        return value

    def effective_weight(self, scope: BeneficiaryScope) -> float:
        return self.weight * self.scope_weights.get(scope, 1.0)


class Constraint(BaseModel):
    description: str
    severity: ConstraintSeverity = ConstraintSeverity.SOFT
    satisfied: bool = True
    penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = ""


class OutcomeBranch(BaseModel):
    description: str
    probability: Estimate
    value: Estimate
    scope: BeneficiaryScope | None = None
    assumptions: list[str] = Field(default_factory=list)


class Option(BaseModel):
    id: str
    name: str
    description: str = ""
    criterion_estimates: dict[str, Estimate] = Field(default_factory=dict)
    outcomes: list[OutcomeBranch] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    exploration_value: Estimate = Field(default_factory=lambda: Estimate.point(0.0, confidence=0.8))
    assumptions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("option id cannot be empty")
        return value


class ObjectiveProfile(BaseModel):
    scope: BeneficiaryScope = BeneficiaryScope.SELF
    risk_attitude: RiskAttitude = RiskAttitude.RISK_NEUTRAL
    moral_weights: dict[str, float] = Field(default_factory=dict)
    beneficiary_weights: dict[BeneficiaryScope, float] = Field(default_factory=dict)
    include_exploration_value: bool = True
    notes: str = ""


class DecisionQuestion(BaseModel):
    prompt: str
    options: list[Option]
    criteria: list[Criterion]
    objective: ObjectiveProfile = Field(default_factory=ObjectiveProfile)
    assumptions: list[str] = Field(default_factory=list)
    horizon: str = "unspecified"
    outcome_mix_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_estimate_default: Estimate = Field(
        default_factory=lambda: Estimate(
            expected=0.5, low=0.25, high=0.75, confidence=0.1, notes="missing estimate default"
        )
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> DecisionQuestion:
        option_ids = [option.id for option in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("option ids must be unique")
        criterion_ids = [criterion.id for criterion in self.criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("criterion ids must be unique")
        return self


class CriterionContribution(BaseModel):
    criterion_id: str
    criterion_name: str
    weight: float
    estimate: Estimate
    score: float = Field(ge=0.0, le=1.0)
    contribution: float
    missing: bool = False


class OptionEvaluation(BaseModel):
    option_id: str
    option_name: str
    status: OptionStatus
    total_score: float
    criteria_score: float
    robust_score: float
    outcome_expected_value: float | None = None
    exploration_bonus: float = 0.0
    soft_penalty: float = 0.0
    regret: float = 0.0
    hard_constraint_failures: list[str] = Field(default_factory=list)
    contributions: list[CriterionContribution] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Crux(BaseModel):
    description: str
    option_id: str | None = None
    criterion_id: str | None = None
    current_gap: float | None = None
    flip_threshold: float | None = None
    value_of_information: float = 0.0
    investigation: str = ""


class DecisionResult(BaseModel):
    prompt: str
    scope: BeneficiaryScope
    risk_attitude: RiskAttitude
    recommended_option_id: str | None
    ranked_options: list[OptionEvaluation]
    cruxes: list[Crux] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def recommended_option(self) -> OptionEvaluation | None:
        if self.recommended_option_id is None:
            return None
        for option in self.ranked_options:
            if option.option_id == self.recommended_option_id:
                return option
        return None


DecisionRole = Literal[
    "empirical_premise",
    "probability_assumption",
    "outcome_model",
    "value_assumption",
    "constraint",
    "decision_criterion",
    "option_description",
    "source_reference",
]


class Proposition(BaseModel):
    id: str
    text: str
    decision_roles: list[DecisionRole] = Field(default_factory=list)
    source_claim_id: str | None = None
    source_node_id: str | None = None
    claim_type: str = "unknown"
    extraction_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    credence: float | None = Field(default=None, ge=0.0, le=1.0)
    credence_low: float | None = Field(default=None, ge=0.0, le=1.0)
    credence_high: float | None = Field(default=None, ge=0.0, le=1.0)
    support_in_text: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: dict[str, Any] = Field(default_factory=dict)


class DecisionReportStyle(str, Enum):
    COMPACT = "compact"
    DETAILED = "detailed"


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
