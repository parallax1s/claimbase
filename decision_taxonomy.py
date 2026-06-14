"""AI-grounded direction taxonomy for the claimbase decision layer.

The claimbase corpus is AI discourse, so we rank directions WITHIN AI that the
corpus can actually speak to — not civilizational causes the corpus is silent on.
Factors are the canonical 8 (ITN+) from the episteme global template; we only
swap in AI sub-directions and keep the calibrated factor weights / scope weights.
"""

from __future__ import annotations

from decision import (
    BeneficiaryScope,
    Direction,
    PrioritizationQuestion,
    global_direction_template,
)

# Directions, grounded in the atlas districts the corpus actually populates
# (alignment discourse, AI policy/safety laws, security mandates, AI ethics/
# welfare, capability/model-release churn).
AI_DIRECTIONS: list[Direction] = [
    Direction(
        id="ai_alignment",
        name="AI alignment & safety research",
        description="Technical work making advanced AI systems do what their operators intend: interpretability, evaluations, oversight, control, alignment theory.",
    ),
    Direction(
        id="ai_governance",
        name="AI governance & policy",
        description="Laws, regulation, institutions, standards, and international coordination shaping how AI is built and deployed.",
    ),
    Direction(
        id="ai_security",
        name="AI security & misuse defense",
        description="Defending against misuse of AI: model security, cyber, bio-misuse, jailbreaks, dangerous-capability safeguards, deployment controls.",
    ),
    Direction(
        id="ai_welfare",
        name="AI welfare & moral patienthood",
        description="Whether and how AI systems may have morally relevant interests, and what is owed to them; model welfare.",
    ),
    Direction(
        id="capability_acceleration",
        name="AI capability acceleration",
        description="Pushing the frontier of AI capability and deployment forward faster (treated as a candidate direction one could prioritize, with its own upside and downside).",
    ),
]

DIRECTION_IDS = [d.id for d in AI_DIRECTIONS]

# Factor reference for the LLM judging prompt (ids match the episteme template).
FACTOR_REF: list[tuple[str, str]] = [
    ("scale", "how large the problem / payoff this direction addresses is"),
    ("neglectedness", "how under-resourced it is at the margin (more neglected = higher)"),
    ("tractability", "how solvable / how much traction effort gets"),
    ("personal_fit", "fit for a specific person (corpus rarely speaks to this)"),
    ("leverage", "how much one contribution moves the outcome"),
    ("career_capital", "skills / option value built by working on it"),
    ("downside_risk", "risk of harm or misuse from the direction (a COST factor)"),
    ("epistemic_robustness", "how well-established / robust the case is to being wrong"),
]
FACTOR_IDS = [fid for fid, _ in FACTOR_REF]


def ai_decision_question(scope: BeneficiaryScope) -> PrioritizationQuestion:
    """Canonical 8 factors (calibrated weights + scope weights) with AI directions."""

    base = global_direction_template(scope=scope)
    return base.model_copy(
        update={
            "prompt": (
                "Within AI, where does the current discourse imply effort has the "
                "highest expected return?"
            ),
            "directions": list(AI_DIRECTIONS),
        }
    )
