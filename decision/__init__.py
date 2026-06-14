"""Vendored from episteme (MIT, github.com/parallax1s/Episteme) — decision engine.

Kept in sync manually; see episteme/src/episteme/decision for upstream.
"""
"""Episteme decision-analysis layer."""

from decision.engine import DecisionEngine
from decision.models import (
    BeneficiaryScope,
    Constraint,
    ConstraintSeverity,
    Criterion,
    CriterionDirection,
    Crux,
    DecisionQuestion,
    DecisionResult,
    Estimate,
    ObjectiveProfile,
    Option,
    OptionEvaluation,
    OptionStatus,
    OutcomeBranch,
    Proposition,
    RiskAttitude,
)
from decision.prioritization import (
    ClaimPolarity,
    ClaimSignal,
    Direction,
    DirectionEvaluation,
    DirectionPrioritizer,
    FactorAggregate,
    PrioritizationFactor,
    PrioritizationModel,
    PrioritizationQuestion,
    PrioritizationResult,
    ScaleConfig,
    claim_signal_from_proposition,
    iter_claim_signals_jsonl,
    write_claim_signals_jsonl,
)
from decision.report import decision_to_markdown, prioritization_to_markdown
from decision.templates import (
    career_80k_template,
    global_direction_template,
    sample_career_question,
    sample_global_direction_signals,
)

__all__ = [
    "BeneficiaryScope",
    "Constraint",
    "ConstraintSeverity",
    "Criterion",
    "CriterionDirection",
    "Crux",
    "DecisionEngine",
    "DecisionQuestion",
    "DecisionResult",
    "Estimate",
    "ObjectiveProfile",
    "Option",
    "OptionEvaluation",
    "OptionStatus",
    "OutcomeBranch",
    "Proposition",
    "RiskAttitude",
    "ClaimPolarity",
    "ClaimSignal",
    "Direction",
    "DirectionEvaluation",
    "DirectionPrioritizer",
    "FactorAggregate",
    "PrioritizationFactor",
    "PrioritizationModel",
    "PrioritizationQuestion",
    "PrioritizationResult",
    "ScaleConfig",
    "career_80k_template",
    "decision_to_markdown",
    "prioritization_to_markdown",
    "global_direction_template",
    "sample_career_question",
    "sample_global_direction_signals",
    "claim_signal_from_proposition",
    "iter_claim_signals_jsonl",
    "write_claim_signals_jsonl",
]
