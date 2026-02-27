"""Reasoning confidence scorer adapter for the AumOS Critical Thinking service.

Quantifies reasoning confidence: evidence strength aggregation, chain validity
scoring, uncertainty propagation, calibrated confidence intervals, confidence
decomposition (evidence vs logic vs assumptions), overconfidence detection,
and structured confidence reports.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain data structures
# ---------------------------------------------------------------------------

@dataclass
class EvidenceConfidence:
    """Confidence contribution from the evidence dimension.

    Attributes:
        raw_score: Raw evidence score before calibration (0.0–1.0).
        calibrated_score: Calibrated evidence confidence (0.0–1.0).
        evidence_count: Number of evidence items evaluated.
        average_source_credibility: Mean source credibility across evidence.
        consistency_ratio: Proportion of evidence consistent with the claim.
    """

    raw_score: float
    calibrated_score: float
    evidence_count: int
    average_source_credibility: float
    consistency_ratio: float


@dataclass
class LogicConfidence:
    """Confidence contribution from the logical reasoning dimension.

    Attributes:
        raw_score: Raw logic score (0.0–1.0).
        calibrated_score: Calibrated logic confidence (0.0–1.0).
        reasoning_chain_length: Number of reasoning steps.
        step_validity_rate: Proportion of reasoning steps that are valid.
        argument_type: Type of argument underlying the reasoning.
        contains_fallacy: True if fallacy was detected in the reasoning chain.
    """

    raw_score: float
    calibrated_score: float
    reasoning_chain_length: int
    step_validity_rate: float
    argument_type: str
    contains_fallacy: bool


@dataclass
class AssumptionConfidence:
    """Confidence contribution from the assumptions dimension.

    Attributes:
        raw_score: Raw assumption score (0.0–1.0).
        calibrated_score: Calibrated assumption confidence (0.0–1.0).
        assumption_count: Number of stated assumptions.
        unstated_assumption_risk: Estimated risk from unstated assumptions (0.0–1.0).
        assumption_plausibility: Average plausibility of listed assumptions (0.0–1.0).
    """

    raw_score: float
    calibrated_score: float
    assumption_count: int
    unstated_assumption_risk: float
    assumption_plausibility: float


@dataclass
class ConfidenceInterval:
    """A calibrated confidence interval for a claim or conclusion.

    Attributes:
        lower: Lower bound of the interval (0.0–1.0).
        upper: Upper bound of the interval (0.0–1.0).
        central: Central estimate (0.0–1.0).
        coverage: Interval coverage level (e.g., 0.90 for 90% CI).
    """

    lower: float
    upper: float
    central: float
    coverage: float


@dataclass
class ConfidenceReport:
    """Structured confidence quantification report.

    Attributes:
        report_id: Unique identifier.
        claim_or_conclusion: The statement being evaluated.
        overall_confidence: Composite confidence score (0.0–1.0).
        confidence_interval: Calibrated interval around the central estimate.
        evidence_component: Evidence dimension breakdown.
        logic_component: Logic dimension breakdown.
        assumption_component: Assumption dimension breakdown.
        uncertainty_sources: Identified sources of uncertainty.
        is_overconfident: True if overconfidence patterns detected.
        overconfidence_signals: Specific overconfidence indicators found.
        generated_at: Timestamp.
    """

    report_id: str
    claim_or_conclusion: str
    overall_confidence: float
    confidence_interval: ConfidenceInterval
    evidence_component: EvidenceConfidence
    logic_component: LogicConfidence
    assumption_component: AssumptionConfidence
    uncertainty_sources: list[str]
    is_overconfident: bool
    overconfidence_signals: list[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ReasoningConfidenceScorer:
    """Computes calibrated confidence scores for reasoning chains and claims.

    Decomposes confidence into evidence, logic, and assumption dimensions,
    propagates uncertainty through reasoning chains, detects overconfidence,
    and generates structured reports with calibrated intervals.
    """

    # Dimension weights for composite confidence computation
    DIMENSION_WEIGHTS: dict[str, float] = {
        "evidence": 0.45,
        "logic": 0.35,
        "assumptions": 0.20,
    }

    # Overconfidence detection threshold: score above this with narrow interval
    OVERCONFIDENCE_SCORE_THRESHOLD: float = 0.90
    OVERCONFIDENCE_INTERVAL_WIDTH_THRESHOLD: float = 0.05

    # Confidence interval coverage levels
    COVERAGE_90: float = 0.90
    CI_90_MULTIPLIER: float = 1.645  # z-score for 90% CI

    def __init__(
        self,
        calibration_factor: float = 0.85,
        overconfidence_correction: float = 0.10,
    ) -> None:
        """Initialise the confidence scorer.

        Args:
            calibration_factor: Shrinkage factor applied to raw scores (0.0–1.0).
                A factor below 1.0 corrects for typical LLM overconfidence.
            overconfidence_correction: Score reduction applied when overconfidence
                is detected.
        """
        self._calibration_factor = min(1.0, max(0.0, calibration_factor))
        self._overconfidence_correction = overconfidence_correction

    def score_from_evidence(
        self,
        evidence_items: list[dict[str, Any]],
        claim_text: str,
    ) -> EvidenceConfidence:
        """Compute the evidence confidence dimension.

        Args:
            evidence_items: List of dicts with 'credibility' (float) and
                'supports_claim' (bool) keys.
            claim_text: The claim being evaluated (used for logging).

        Returns:
            EvidenceConfidence with raw and calibrated scores.
        """
        if not evidence_items:
            return EvidenceConfidence(
                raw_score=0.30,
                calibrated_score=0.30 * self._calibration_factor,
                evidence_count=0,
                average_source_credibility=0.0,
                consistency_ratio=0.0,
            )

        supporting = [e for e in evidence_items if e.get("supports_claim", True)]
        avg_credibility = sum(e.get("credibility", 0.5) for e in evidence_items) / len(evidence_items)
        consistency_ratio = len(supporting) / len(evidence_items) if evidence_items else 0.0

        # More evidence + higher credibility + better consistency = higher raw score
        evidence_depth_bonus = min(len(evidence_items) / 5.0, 0.20)
        raw_score = (avg_credibility * 0.60 + consistency_ratio * 0.40) + evidence_depth_bonus * avg_credibility
        raw_score = min(1.0, raw_score)
        calibrated_score = raw_score * self._calibration_factor

        logger.debug(
            "Evidence confidence computed",
            evidence_count=len(evidence_items),
            avg_credibility=avg_credibility,
            consistency_ratio=consistency_ratio,
            raw_score=raw_score,
        )
        return EvidenceConfidence(
            raw_score=round(raw_score, 4),
            calibrated_score=round(calibrated_score, 4),
            evidence_count=len(evidence_items),
            average_source_credibility=round(avg_credibility, 4),
            consistency_ratio=round(consistency_ratio, 4),
        )

    def score_from_reasoning_chain(
        self,
        steps: list[dict[str, Any]],
        argument_type: str = "inductive",
        contains_fallacy: bool = False,
    ) -> LogicConfidence:
        """Compute the logic confidence dimension from a reasoning chain.

        Args:
            steps: List of reasoning step dicts with 'is_valid' (bool) and
                'confidence' (float) keys.
            argument_type: Type of argument (deductive/inductive/abductive/causal).
            contains_fallacy: True if a fallacy was detected in the chain.

        Returns:
            LogicConfidence with raw and calibrated scores.
        """
        if not steps:
            return LogicConfidence(
                raw_score=0.40,
                calibrated_score=0.40 * self._calibration_factor,
                reasoning_chain_length=0,
                step_validity_rate=0.0,
                argument_type=argument_type,
                contains_fallacy=contains_fallacy,
            )

        valid_steps = [s for s in steps if s.get("is_valid", True)]
        validity_rate = len(valid_steps) / len(steps)
        avg_step_confidence = sum(s.get("confidence", 0.5) for s in steps) / len(steps)

        # Deductive arguments warrant higher logic confidence when valid
        type_multipliers: dict[str, float] = {
            "deductive": 1.10, "causal": 1.00, "inductive": 0.95, "abductive": 0.90, "analogical": 0.85,
        }
        type_multiplier = type_multipliers.get(argument_type, 0.95)
        raw_score = avg_step_confidence * validity_rate * type_multiplier
        if contains_fallacy:
            raw_score *= 0.60  # Significant penalty for fallacious reasoning
        raw_score = min(1.0, raw_score)
        calibrated_score = raw_score * self._calibration_factor

        return LogicConfidence(
            raw_score=round(raw_score, 4),
            calibrated_score=round(calibrated_score, 4),
            reasoning_chain_length=len(steps),
            step_validity_rate=round(validity_rate, 4),
            argument_type=argument_type,
            contains_fallacy=contains_fallacy,
        )

    def score_from_assumptions(
        self,
        assumptions: list[str],
        unstated_risk: float = 0.20,
    ) -> AssumptionConfidence:
        """Compute the assumption confidence dimension.

        Args:
            assumptions: List of stated assumption strings.
            unstated_risk: Estimated risk from unstated assumptions (0.0–1.0).

        Returns:
            AssumptionConfidence with raw and calibrated scores.
        """
        assumption_count = len(assumptions)
        # More assumptions increase explicitness but also increase risk
        explicitness_bonus = min(assumption_count * 0.05, 0.20)
        assumption_plausibility = max(0.0, 0.80 - assumption_count * 0.08)  # Degrades with more assumptions
        raw_score = max(0.0, (1.0 - unstated_risk) * (assumption_plausibility + explicitness_bonus * 0.5))
        raw_score = min(1.0, raw_score)
        calibrated_score = raw_score * self._calibration_factor

        return AssumptionConfidence(
            raw_score=round(raw_score, 4),
            calibrated_score=round(calibrated_score, 4),
            assumption_count=assumption_count,
            unstated_assumption_risk=unstated_risk,
            assumption_plausibility=round(assumption_plausibility, 4),
        )

    def compute_composite_confidence(
        self,
        evidence_component: EvidenceConfidence,
        logic_component: LogicConfidence,
        assumption_component: AssumptionConfidence,
    ) -> float:
        """Compute the overall composite confidence from three dimensions.

        Args:
            evidence_component: Evidence dimension scores.
            logic_component: Logic dimension scores.
            assumption_component: Assumption dimension scores.

        Returns:
            Weighted composite confidence score (0.0–1.0).
        """
        composite = (
            self.DIMENSION_WEIGHTS["evidence"] * evidence_component.calibrated_score
            + self.DIMENSION_WEIGHTS["logic"] * logic_component.calibrated_score
            + self.DIMENSION_WEIGHTS["assumptions"] * assumption_component.calibrated_score
        )
        return round(min(1.0, max(0.0, composite)), 4)

    def compute_confidence_interval(
        self,
        point_estimate: float,
        evidence_count: int,
        step_count: int,
        coverage: float = COVERAGE_90,
    ) -> ConfidenceInterval:
        """Compute a calibrated confidence interval around a point estimate.

        Uses a modified Wilson score-like approach. Wider intervals for fewer
        data points (evidence + reasoning steps).

        Args:
            point_estimate: Central confidence estimate (0.0–1.0).
            evidence_count: Number of evidence items (reduces uncertainty).
            step_count: Number of reasoning steps (reduces uncertainty).
            coverage: Interval coverage level (default 0.90).

        Returns:
            ConfidenceInterval with lower, upper, central, coverage.
        """
        n = max(evidence_count + step_count, 1)
        # Uncertainty shrinks proportionally to sqrt(n)
        base_uncertainty = 0.20 / (n ** 0.5)
        half_width = self.CI_90_MULTIPLIER * base_uncertainty

        lower = max(0.0, point_estimate - half_width)
        upper = min(1.0, point_estimate + half_width)

        return ConfidenceInterval(
            lower=round(lower, 4),
            upper=round(upper, 4),
            central=point_estimate,
            coverage=coverage,
        )

    def detect_overconfidence(
        self,
        overall_confidence: float,
        confidence_interval: ConfidenceInterval,
        evidence_count: int,
        logic_component: LogicConfidence,
    ) -> tuple[bool, list[str]]:
        """Detect overconfidence patterns in the confidence assessment.

        Args:
            overall_confidence: Composite confidence score.
            confidence_interval: Calibrated confidence interval.
            evidence_count: Number of evidence items.
            logic_component: Logic confidence dimension.

        Returns:
            Tuple of (is_overconfident, signals_list).
        """
        signals: list[str] = []
        interval_width = confidence_interval.upper - confidence_interval.lower

        if overall_confidence >= self.OVERCONFIDENCE_SCORE_THRESHOLD and evidence_count < 3:
            signals.append(f"High confidence ({overall_confidence:.2f}) with insufficient evidence ({evidence_count} items).")

        if interval_width < self.OVERCONFIDENCE_INTERVAL_WIDTH_THRESHOLD and evidence_count < 5:
            signals.append(f"Very narrow CI width ({interval_width:.3f}) with limited evidence base.")

        if logic_component.contains_fallacy and overall_confidence > 0.75:
            signals.append("High confidence maintained despite detected fallacy in reasoning chain.")

        if logic_component.step_validity_rate < 0.70 and overall_confidence > 0.80:
            signals.append(
                f"High confidence ({overall_confidence:.2f}) despite low reasoning validity rate "
                f"({logic_component.step_validity_rate:.2f})."
            )

        return bool(signals), signals

    def propagate_uncertainty(
        self, step_confidences: list[float]
    ) -> float:
        """Propagate uncertainty through a reasoning chain.

        Confidence degrades multiplicatively through the chain: each step
        reduces overall confidence proportionally to its own confidence.

        Args:
            step_confidences: Ordered list of per-step confidence scores.

        Returns:
            Propagated chain confidence (0.0–1.0).
        """
        if not step_confidences:
            return 0.0
        propagated = step_confidences[0]
        for step_conf in step_confidences[1:]:
            propagated *= step_conf
        return round(min(1.0, max(0.0, propagated)), 4)

    def generate_report(
        self,
        claim_or_conclusion: str,
        evidence_component: EvidenceConfidence,
        logic_component: LogicConfidence,
        assumption_component: AssumptionConfidence,
        uncertainty_sources: list[str] | None = None,
    ) -> ConfidenceReport:
        """Generate a full structured confidence report.

        Args:
            claim_or_conclusion: The statement being assessed.
            evidence_component: Evidence confidence dimension.
            logic_component: Logic confidence dimension.
            assumption_component: Assumption confidence dimension.
            uncertainty_sources: Optional list of known uncertainty sources.

        Returns:
            Complete ConfidenceReport.
        """
        overall_confidence = self.compute_composite_confidence(
            evidence_component, logic_component, assumption_component
        )
        confidence_interval = self.compute_confidence_interval(
            point_estimate=overall_confidence,
            evidence_count=evidence_component.evidence_count,
            step_count=logic_component.reasoning_chain_length,
        )
        is_overconfident, overconfidence_signals = self.detect_overconfidence(
            overall_confidence=overall_confidence,
            confidence_interval=confidence_interval,
            evidence_count=evidence_component.evidence_count,
            logic_component=logic_component,
        )

        if is_overconfident:
            overall_confidence = max(0.0, overall_confidence - self._overconfidence_correction)

        report = ConfidenceReport(
            report_id=str(uuid.uuid4()),
            claim_or_conclusion=claim_or_conclusion,
            overall_confidence=overall_confidence,
            confidence_interval=confidence_interval,
            evidence_component=evidence_component,
            logic_component=logic_component,
            assumption_component=assumption_component,
            uncertainty_sources=uncertainty_sources or self._infer_uncertainty_sources(
                evidence_component, logic_component, assumption_component
            ),
            is_overconfident=is_overconfident,
            overconfidence_signals=overconfidence_signals,
            generated_at=datetime.now(tz=timezone.utc),
        )

        logger.info(
            "Confidence report generated",
            report_id=report.report_id,
            overall_confidence=overall_confidence,
            is_overconfident=is_overconfident,
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_uncertainty_sources(
        self,
        evidence: EvidenceConfidence,
        logic: LogicConfidence,
        assumptions: AssumptionConfidence,
    ) -> list[str]:
        """Infer primary uncertainty sources from component scores.

        Args:
            evidence: Evidence dimension.
            logic: Logic dimension.
            assumptions: Assumption dimension.

        Returns:
            List of uncertainty source descriptions.
        """
        sources: list[str] = []
        if evidence.evidence_count < 3:
            sources.append("Limited evidence base (fewer than 3 items).")
        if evidence.consistency_ratio < 0.60:
            sources.append(f"Low evidence consistency ({evidence.consistency_ratio:.1%}).")
        if logic.step_validity_rate < 0.80:
            sources.append(f"Reasoning chain has validity gaps ({logic.step_validity_rate:.1%} valid steps).")
        if logic.contains_fallacy:
            sources.append("Logical fallacy detected in reasoning chain.")
        if assumptions.unstated_assumption_risk > 0.40:
            sources.append(f"High unstated assumption risk ({assumptions.unstated_assumption_risk:.1%}).")
        if assumptions.assumption_count > 4:
            sources.append(f"Many explicit assumptions ({assumptions.assumption_count}) increase brittleness.")
        return sources or ["No dominant uncertainty sources identified."]
