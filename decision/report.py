"""Markdown rendering for decision results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from decision.models import DecisionReportStyle, DecisionResult

if TYPE_CHECKING:
    from decision.prioritization import PrioritizationResult


def decision_to_markdown(
    result: DecisionResult, *, style: DecisionReportStyle = DecisionReportStyle.DETAILED
) -> str:
    lines: list[str] = []
    lines.append(f"# Decision analysis: {result.prompt}")
    lines.append("")
    lines.append(f"Scope: `{result.scope.value}`  ")
    lines.append(f"Risk attitude: `{result.risk_attitude.value}`")
    lines.append("")
    if result.recommended_option:
        rec = result.recommended_option
        lines.append(
            f"**Recommendation:** {rec.option_name} (`{rec.option_id}`), "
            f"score {rec.total_score:.3f}."
        )
    else:
        lines.append("**Recommendation:** no permissible option found.")
    lines.append("")
    lines.append("## Ranked options")
    lines.append("")
    lines.append("| rank | option | status | total | robust | regret | notes |")
    lines.append("|---:|---|---|---:|---:|---:|---|")
    for index, option in enumerate(result.ranked_options, start=1):
        notes = "; ".join(option.notes + option.hard_constraint_failures)
        lines.append(
            f"| {index} | {option.option_name} | {option.status.value} | "
            f"{option.total_score:.3f} | {option.robust_score:.3f} | "
            f"{option.regret:.3f} | {notes} |"
        )
    if style == DecisionReportStyle.DETAILED:
        lines.append("")
        lines.append("## Leading cruxes / value of information")
        lines.append("")
        if result.cruxes:
            for crux in result.cruxes:
                voi = f" VOI≈{crux.value_of_information:.3f}." if crux.value_of_information else ""
                threshold = (
                    f" Flip threshold≈{crux.flip_threshold:.3f}."
                    if crux.flip_threshold is not None
                    else ""
                )
                investigation = (
                    f" Investigation: {crux.investigation}" if crux.investigation else ""
                )
                lines.append(f"- {crux.description}{voi}{threshold}{investigation}")
        else:
            lines.append("No cruxes identified.")
        lines.append("")
        lines.append("## Contributions")
        for option in result.ranked_options:
            lines.append("")
            lines.append(f"### {option.option_name}")
            lines.append("| criterion | weight | score | contribution | interval | confidence |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for c in option.contributions:
                interval = f"{c.estimate.low:.2f}–{c.estimate.high:.2f}"
                missing = " *missing default*" if c.missing else ""
                lines.append(
                    f"| {c.criterion_name}{missing} | {c.weight:.3f} | {c.score:.3f} | "
                    f"{c.contribution:.3f} | {interval} | {c.estimate.confidence:.2f} |"
                )
    if result.warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in result.warnings:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def prioritization_to_markdown(
    result: PrioritizationResult,
    *,
    style: DecisionReportStyle = DecisionReportStyle.DETAILED,
) -> str:
    """Render a high-level direction prioritization result as Markdown."""

    lines: list[str] = []
    lines.append(f"# Direction prioritization: {result.prompt}")
    lines.append("")
    lines.append(f"Scope: `{result.scope.value}`  ")
    lines.append(f"Risk attitude: `{result.risk_attitude.value}`")
    lines.append("")
    if result.recommended_direction:
        rec = result.recommended_direction
        lines.append(
            f"**Top direction:** {rec.direction_name} (`{rec.direction_id}`), "
            f"score {rec.normalized_score:.3f}, relative return index "
            f"{rec.expected_return_index:.2f}."
        )
    else:
        lines.append("**Top direction:** no permissible direction found.")
    lines.append("")
    lines.append("## Ranked directions")
    lines.append("")
    lines.append(
        "| rank | direction | status | score | relative return | robust | evidence | notes |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")
    for index, direction in enumerate(result.ranked_directions, start=1):
        notes = "; ".join(direction.notes + direction.hard_constraint_failures)
        lines.append(
            f"| {index} | {direction.direction_name} | {direction.status.value} | "
            f"{direction.normalized_score:.3f} | {direction.expected_return_index:.2f} | "
            f"{direction.robust_score:.3f} | {direction.evidence_count} | {notes} |"
        )
    if style == DecisionReportStyle.DETAILED:
        lines.append("")
        lines.append("## Leading cruxes / value of information")
        lines.append("")
        if result.cruxes:
            for crux in result.cruxes:
                voi = f" VOI≈{crux.value_of_information:.3f}." if crux.value_of_information else ""
                investigation = (
                    f" Investigation: {crux.investigation}" if crux.investigation else ""
                )
                lines.append(f"- {crux.description}{voi}{investigation}")
        else:
            lines.append("No cruxes identified.")
        lines.append("")
        lines.append("## Factor aggregates")
        for direction in result.ranked_directions:
            lines.append("")
            lines.append(f"### {direction.direction_name}")
            lines.append(
                "| factor | score | estimate | interval | confidence | signals | "
                "eff. weight | groups |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for factor in direction.factor_scores:
                interval = f"{factor.estimate.low:.2f}–{factor.estimate.high:.2f}"
                missing = " *missing/default*" if factor.missing else ""
                lines.append(
                    f"| {factor.factor_name}{missing} | {factor.adjusted_score:.3f} | "
                    f"{factor.estimate.expected:.3f} | {interval} | "
                    f"{factor.estimate.confidence:.2f} | {factor.evidence_count} | "
                    f"{factor.effective_weight:.2f} | {factor.dependency_group_count} |"
                )
    if result.warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in result.warnings:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)
