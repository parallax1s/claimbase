"""Decision-question templates, including an 80,000 Hours-inspired career template."""

from __future__ import annotations

from typing import TYPE_CHECKING

from decision.models import (
    BeneficiaryScope,
    Criterion,
    CriterionDirection,
    DecisionQuestion,
    Estimate,
    ObjectiveProfile,
    Option,
    RiskAttitude,
)

if TYPE_CHECKING:
    from decision.prioritization import ClaimSignal, PrioritizationQuestion


CAREER_80K_CRITERIA: list[Criterion] = [
    Criterion(
        id="personal_fit",
        name="Personal fit",
        description="Expected performance, motivation, and day-to-day fit after realistic testing.",
        weight=1.25,
        scope_weights={
            BeneficiaryScope.SELF: 1.5,
            BeneficiaryScope.FAMILY: 1.2,
            BeneficiaryScope.HUMANITY: 0.8,
            BeneficiaryScope.ALL_SENTIENT_LIFE: 0.8,
        },
        tags=["fit", "sustainability"],
    ),
    Criterion(
        id="problem_scale",
        name="Problem scale",
        description="How large the problem or opportunity is if progress is made.",
        weight=1.30,
        scope_weights={
            BeneficiaryScope.SELF: 0.3,
            BeneficiaryScope.FAMILY: 0.4,
            BeneficiaryScope.HUMANITY: 1.6,
            BeneficiaryScope.FUTURE_GENERATIONS: 1.8,
            BeneficiaryScope.ALL_SENTIENT_LIFE: 1.9,
        },
        tags=["80k", "pressingness", "scale"],
    ),
    Criterion(
        id="neglectedness",
        name="Neglectedness",
        description=(
            "Whether an extra person/resource is likely to matter because the area "
            "is under-supplied."
        ),
        weight=0.85,
        scope_weights={
            BeneficiaryScope.HUMANITY: 1.3,
            BeneficiaryScope.FUTURE_GENERATIONS: 1.4,
            BeneficiaryScope.ALL_SENTIENT_LIFE: 1.4,
        },
        tags=["80k", "pressingness", "neglectedness"],
    ),
    Criterion(
        id="solvability",
        name="Solvability / tractability",
        description="How likely additional effort is to make progress on the problem.",
        weight=1.00,
        scope_weights={
            BeneficiaryScope.HUMANITY: 1.2,
            BeneficiaryScope.FUTURE_GENERATIONS: 1.2,
            BeneficiaryScope.ALL_SENTIENT_LIFE: 1.2,
        },
        tags=["80k", "pressingness", "tractability"],
    ),
    Criterion(
        id="career_capital",
        name="Career capital",
        description="Transferable skills, credentials, network, runway, and option value created.",
        weight=1.00,
        scope_weights={
            BeneficiaryScope.SELF: 1.2,
            BeneficiaryScope.FAMILY: 1.1,
            BeneficiaryScope.HUMANITY: 1.0,
            BeneficiaryScope.ALL_SENTIENT_LIFE: 1.0,
        },
        tags=["80k", "future_impact", "option_value"],
    ),
    Criterion(
        id="supportive_conditions",
        name="Supportive conditions",
        description="Income, health, colleagues, location, energy, and family compatibility.",
        weight=0.90,
        scope_weights={
            BeneficiaryScope.SELF: 1.5,
            BeneficiaryScope.FAMILY: 1.7,
            BeneficiaryScope.HUMANITY: 0.7,
            BeneficiaryScope.ALL_SENTIENT_LIFE: 0.7,
        },
        tags=["sustainability", "constraints"],
    ),
    Criterion(
        id="exploration_value",
        name="Exploration value",
        description="How much the option teaches you about fit, impact, and future choices.",
        weight=0.60,
        scope_weights={
            BeneficiaryScope.SELF: 1.2,
            BeneficiaryScope.HUMANITY: 1.0,
            BeneficiaryScope.ALL_SENTIENT_LIFE: 1.0,
        },
        tags=["voi", "cheap_tests"],
    ),
]


def career_80k_template(
    *,
    scope: BeneficiaryScope = BeneficiaryScope.HUMANITY,
    risk_attitude: RiskAttitude = RiskAttitude.RISK_NEUTRAL,
) -> DecisionQuestion:
    """Return a blank-but-runnable 80k-style career decision question.

    The template intentionally contains example options with neutral estimates so
    users can replace them rather than starting from a blank JSON file.
    """

    neutral_scores = {
        criterion.id: Estimate(expected=0.5, low=0.3, high=0.7, confidence=0.2, notes="replace me")
        for criterion in CAREER_80K_CRITERIA
    }
    options = [
        Option(
            id="option_a",
            name="Option A",
            description="Replace with a concrete career path, e.g. policy fellowship.",
            criterion_estimates=neutral_scores,
            exploration_value=Estimate(expected=0.6, low=0.3, high=0.8, confidence=0.3),
        ),
        Option(
            id="option_b",
            name="Option B",
            description=(
                "Replace with another concrete career path, e.g. software role with later pivot."
            ),
            criterion_estimates=neutral_scores,
            exploration_value=Estimate(expected=0.6, low=0.3, high=0.8, confidence=0.3),
        ),
    ]
    return DecisionQuestion(
        prompt="Which career path should I pursue?",
        criteria=CAREER_80K_CRITERIA,
        options=options,
        objective=ObjectiveProfile(
            scope=scope,
            risk_attitude=risk_attitude,
            moral_weights={
                "near_term_human_welfare": 0.35,
                "future_generations": 0.35,
                "animal_welfare": 0.20,
                "rights_constraints": 0.10,
            },
            include_exploration_value=True,
            notes=(
                "80,000 Hours-inspired career template: fit × pressingness + "
                "career capital + exploration value."
            ),
        ),
        assumptions=[
            "Scores are normalized to 0..1 where higher is better.",
            "Problem pressingness is represented by scale, neglectedness, and solvability.",
            (
                "Personal fit and supportive conditions are treated as both welfare "
                "constraints and impact multipliers."
            ),
            "Exploration value can dominate early when cheap tests could change the decision.",
        ],
        horizon="career / multi-year",
        metadata={"template": "career_80k_v1"},
    )


def sample_career_question(scope: BeneficiaryScope = BeneficiaryScope.HUMANITY) -> DecisionQuestion:
    """A small example useful for smoke tests and demos."""

    def estimates(**values: tuple[float, float, float, float]) -> dict[str, Estimate]:
        return {
            key: Estimate(expected=v[0], low=v[1], high=v[2], confidence=v[3])
            for key, v in values.items()
        }

    return DecisionQuestion(
        prompt="Which path has the highest expected return?",
        criteria=CAREER_80K_CRITERIA,
        objective=ObjectiveProfile(scope=scope, risk_attitude=RiskAttitude.RISK_NEUTRAL),
        options=[
            Option(
                id="ai_policy",
                name="AI governance policy fellowship",
                criterion_estimates=estimates(
                    personal_fit=(0.62, 0.40, 0.80, 0.55),
                    problem_scale=(0.92, 0.75, 0.99, 0.55),
                    neglectedness=(0.78, 0.45, 0.90, 0.45),
                    solvability=(0.55, 0.25, 0.75, 0.35),
                    career_capital=(0.76, 0.45, 0.90, 0.55),
                    supportive_conditions=(0.48, 0.25, 0.65, 0.45),
                    exploration_value=(0.82, 0.60, 0.95, 0.70),
                ),
                exploration_value=Estimate(expected=0.75, low=0.50, high=0.90, confidence=0.65),
            ),
            Option(
                id="software",
                name="Software engineering with later pivot",
                criterion_estimates=estimates(
                    personal_fit=(0.76, 0.60, 0.90, 0.70),
                    problem_scale=(0.58, 0.35, 0.80, 0.45),
                    neglectedness=(0.45, 0.25, 0.70, 0.45),
                    solvability=(0.70, 0.50, 0.85, 0.60),
                    career_capital=(0.86, 0.70, 0.95, 0.75),
                    supportive_conditions=(0.82, 0.65, 0.92, 0.75),
                    exploration_value=(0.55, 0.35, 0.75, 0.55),
                ),
                exploration_value=Estimate(expected=0.45, low=0.25, high=0.65, confidence=0.55),
            ),
        ],
        assumptions=["Example estimates only; replace with user-specific evidence."],
        horizon="5-10 years",
        metadata={"template": "sample_career_80k_v1"},
    )


def global_direction_template(
    *,
    scope: BeneficiaryScope = BeneficiaryScope.HUMANITY,
    risk_attitude: RiskAttitude = RiskAttitude.RISK_NEUTRAL,
) -> PrioritizationQuestion:
    """Template for civilization-scale direction prioritization.

    This is for questions like "work on AI safety or curing cancer?" rather
    than narrow career-option comparison.  It is intentionally claim-signal
    driven: users can stream large JSONL signal files into the prioritizer.
    """

    from decision.prioritization import (
        Direction,
        PrioritizationFactor,
        PrioritizationModel,
        PrioritizationQuestion,
    )

    factors = [
        PrioritizationFactor(
            id="scale",
            name="Scale of the problem",
            description=(
                "How much welfare, risk, or opportunity is at stake if the problem "
                "is solved or mitigated."
            ),
            weight=1.40,
            scope_weights={
                BeneficiaryScope.SELF: 0.4,
                BeneficiaryScope.FAMILY: 0.5,
                BeneficiaryScope.HUMANITY: 1.5,
                BeneficiaryScope.FUTURE_GENERATIONS: 1.8,
                BeneficiaryScope.ALL_SENTIENT_LIFE: 2.0,
            },
            tags=["80k", "importance", "scope"],
        ),
        PrioritizationFactor(
            id="neglectedness",
            name="Neglectedness / marginal room",
            description=(
                "Whether an additional capable person or dollar is likely to matter at the margin."
            ),
            weight=1.05,
            scope_weights={BeneficiaryScope.HUMANITY: 1.2, BeneficiaryScope.ALL_SENTIENT_LIFE: 1.2},
            tags=["80k", "marginal_impact"],
        ),
        PrioritizationFactor(
            id="tractability",
            name="Tractability / solvability",
            description=(
                "How plausible it is that extra effort changes outcomes rather than "
                "merely producing motion."
            ),
            weight=1.10,
            tags=["80k", "solvability"],
        ),
        PrioritizationFactor(
            id="personal_fit",
            name="Personal fit / comparative advantage",
            description=(
                "How unusually well the decision-maker can contribute relative to alternatives."
            ),
            weight=1.00,
            scope_weights={BeneficiaryScope.SELF: 1.6, BeneficiaryScope.FAMILY: 1.3},
            tags=["fit", "execution"],
        ),
        PrioritizationFactor(
            id="leverage",
            name="Contribution leverage",
            description=(
                "Access to levers such as field-building, policy, research, "
                "funding, or scalable institutions."
            ),
            weight=1.00,
            tags=["leverage", "pathway"],
        ),
        PrioritizationFactor(
            id="career_capital",
            name="Career capital / option value",
            description=(
                "Transferable skills, networks, credentials, and future flexibility "
                "generated by the direction."
            ),
            weight=0.75,
            scope_weights={BeneficiaryScope.SELF: 1.3, BeneficiaryScope.FAMILY: 1.2},
            geometric=False,
            tags=["option_value"],
        ),
        PrioritizationFactor(
            id="downside_risk",
            name="Downside and misuse risk",
            description=(
                "Risk that the work backfires, accelerates harm, causes burnout, "
                "or violates constraints."
            ),
            weight=0.85,
            direction=CriterionDirection.COST,
            geometric=True,
            tags=["risk", "constraints"],
        ),
        PrioritizationFactor(
            id="epistemic_robustness",
            name="Epistemic robustness",
            description=(
                "How stable the recommendation is across moral views, evidence "
                "sources, and model uncertainty."
            ),
            weight=0.65,
            geometric=False,
            tags=["robustness", "moral_uncertainty"],
        ),
    ]
    directions = [
        Direction(
            id="ai_safety",
            name="Work on AI safety / AI governance",
            description="Reduce catastrophic and societal risks from advanced AI systems.",
            tags=["x-risk", "future_generations", "technology_governance"],
        ),
        Direction(
            id="biosecurity",
            name="Work on biosecurity and pandemic prevention",
            description="Prevent natural, accidental, and engineered biological catastrophes.",
            tags=["x-risk", "health", "preparedness"],
        ),
        Direction(
            id="cancer_biomedicine",
            name="Work on curing cancer / biomedical R&D",
            description=(
                "Advance prevention, diagnosis, or treatment of cancer and related disease burdens."
            ),
            tags=["health", "science", "near_term_welfare"],
        ),
        Direction(
            id="global_health",
            name="Work on global health and development",
            description=(
                "Improve cost-effective interventions for poverty, infectious "
                "disease, and wellbeing."
            ),
            tags=["near_term_welfare", "global_poverty"],
        ),
        Direction(
            id="animal_welfare",
            name="Work on animal welfare / alternative proteins",
            description=(
                "Reduce large-scale nonhuman animal suffering and improve food-system ethics."
            ),
            tags=["sentient_life", "animal_welfare"],
        ),
        Direction(
            id="institution_building",
            name="Work on institutional decision quality",
            description="Improve governance, epistemics, forecasting, and coordination capacity.",
            tags=["meta", "institutions", "coordination"],
        ),
    ]
    return PrioritizationQuestion(
        prompt="Which large-scale direction has the highest expected return?",
        directions=directions,
        factors=factors,
        objective=ObjectiveProfile(
            scope=scope,
            risk_attitude=risk_attitude,
            moral_weights={
                "near_term_human_welfare": 0.30,
                "future_generations": 0.35,
                "animal_welfare": 0.20,
                "rights_constraints": 0.15,
            },
            notes=(
                "Large-scale prioritization template inspired by 80k-style "
                "importance, neglectedness, tractability, fit, and option value."
            ),
        ),
        model=PrioritizationModel.HYBRID,
        assumptions=[
            "Claim signals are directional pieces of evidence, not final truth certificates.",
            (
                "Repeated claims should share dependency_key/source_id so one "
                "source cannot dominate via duplication."
            ),
            (
                "Scores are normalized; use cruxes and sensitivity rather than "
                "treating the top score as metaphysically precise."
            ),
            (
                "Beneficiary scope changes the weights, so rerun for self, "
                "family, humanity, and all_sentient_life."
            ),
        ],
        horizon="multi-year to civilizational",
        metadata={"template": "global_direction_prioritization_v1"},
    )


def sample_global_direction_signals() -> list[ClaimSignal]:
    """Small illustrative signal set for the global-direction template."""

    from decision.prioritization import ClaimPolarity, ClaimSignal

    return [
        ClaimSignal(
            id="ai_scale_1",
            direction_id="ai_safety",
            factor_id="scale",
            credence=0.86,
            strength=0.92,
            relevance=0.95,
            extraction_confidence=0.85,
            evidence_quality=0.65,
            dependency_key="ai-risk-literature",
            text="Advanced AI could affect very large numbers of present and future people.",
        ),
        ClaimSignal(
            id="ai_neglected_1",
            direction_id="ai_safety",
            factor_id="neglectedness",
            credence=0.72,
            strength=0.70,
            evidence_quality=0.55,
            extraction_confidence=0.80,
            dependency_key="ai-talent-gap",
        ),
        ClaimSignal(
            id="ai_tractability_1",
            direction_id="ai_safety",
            factor_id="tractability",
            credence=0.58,
            strength=0.55,
            evidence_quality=0.45,
            extraction_confidence=0.75,
            dependency_key="ai-tractability-uncertain",
        ),
        ClaimSignal(
            id="ai_risk_1",
            direction_id="ai_safety",
            factor_id="downside_risk",
            polarity=ClaimPolarity.SUPPORTS,
            credence=0.62,
            strength=0.50,
            evidence_quality=0.55,
            extraction_confidence=0.80,
            dependency_key="dual-use-risk",
        ),
        ClaimSignal(
            id="cancer_scale_1",
            direction_id="cancer_biomedicine",
            factor_id="scale",
            credence=0.90,
            strength=0.70,
            evidence_quality=0.80,
            extraction_confidence=0.90,
            dependency_key="gbd-cancer-burden",
            text="Cancer causes a large current disease burden.",
        ),
        ClaimSignal(
            id="cancer_tractability_1",
            direction_id="cancer_biomedicine",
            factor_id="tractability",
            credence=0.78,
            strength=0.65,
            evidence_quality=0.80,
            extraction_confidence=0.90,
            dependency_key="biomed-progress",
        ),
        ClaimSignal(
            id="cancer_neglected_1",
            direction_id="cancer_biomedicine",
            factor_id="neglectedness",
            polarity=ClaimPolarity.WEAKENS,
            credence=0.82,
            strength=0.65,
            evidence_quality=0.80,
            extraction_confidence=0.90,
            dependency_key="cancer-funding-large",
            text="Cancer research already receives large funding relative to many causes.",
        ),
        ClaimSignal(
            id="animal_scale_1",
            direction_id="animal_welfare",
            factor_id="scale",
            credence=0.80,
            strength=0.82,
            evidence_quality=0.65,
            extraction_confidence=0.85,
            dependency_key="farm-animal-counts",
        ),
        ClaimSignal(
            id="animal_neglected_1",
            direction_id="animal_welfare",
            factor_id="neglectedness",
            credence=0.84,
            strength=0.75,
            evidence_quality=0.65,
            extraction_confidence=0.85,
            dependency_key="animal-neglectedness",
        ),
        ClaimSignal(
            id="animal_scope_robustness",
            direction_id="animal_welfare",
            factor_id="epistemic_robustness",
            polarity=ClaimPolarity.WEAKENS,
            credence=0.65,
            strength=0.45,
            evidence_quality=0.55,
            extraction_confidence=0.80,
            dependency_key="moral-patient-uncertainty",
        ),
    ]
