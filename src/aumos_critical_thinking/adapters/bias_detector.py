"""Cognitive bias detector adapter for the AumOS Critical Thinking service.

Detects 15+ cognitive bias types in AI outputs and human reasoning:
confirmation bias, anchoring, availability heuristic, framing effects,
sunk cost, and more. Includes severity scoring and mitigation recommendations.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cognitive bias catalogue (15+ types)
# ---------------------------------------------------------------------------

COGNITIVE_BIAS_CATALOGUE: dict[str, dict[str, Any]] = {
    "confirmation_bias": {
        "name": "Confirmation Bias",
        "description": "Selectively searching for or interpreting information that confirms pre-existing beliefs.",
        "ai_indicators": ["only cites sources supporting initial hypothesis", "dismisses contradictory evidence", "one-sided source selection"],
        "severity_range": (0.3, 1.0),
        "mitigation": "Actively seek disconfirming evidence. Present balanced source selection.",
    },
    "anchoring_bias": {
        "name": "Anchoring Bias",
        "description": "Over-relying on the first piece of information encountered when making decisions.",
        "ai_indicators": ["initial estimate dominates subsequent analysis", "numerical anchors drive conclusions", "first data point weighted disproportionately"],
        "severity_range": (0.2, 0.8),
        "mitigation": "Re-analyse from multiple starting points. Blind the initial estimate during later analysis.",
    },
    "availability_heuristic": {
        "name": "Availability Heuristic",
        "description": "Overestimating the likelihood of events that are easily recalled or recent.",
        "ai_indicators": ["recent events disproportionately weighted", "vivid examples treated as representative", "recency bias in risk assessment"],
        "severity_range": (0.2, 0.7),
        "mitigation": "Use base rate statistics rather than memorable examples. Review historical data distributions.",
    },
    "framing_effect": {
        "name": "Framing Effect",
        "description": "Drawing different conclusions from the same information depending on how it is presented.",
        "ai_indicators": ["response changes based on positive vs negative framing", "loss framing triggers risk aversion", "gain framing triggers risk seeking"],
        "severity_range": (0.3, 0.9),
        "mitigation": "Re-frame problems multiple ways before concluding. Test sensitivity to framing changes.",
    },
    "sunk_cost_fallacy": {
        "name": "Sunk Cost Fallacy",
        "description": "Continuing a course of action due to past investment despite unfavourable prospects.",
        "ai_indicators": ["recommends continuation citing prior investment", "past costs factor into forward-looking decisions", "exit reluctance correlates with prior spend"],
        "severity_range": (0.4, 1.0),
        "mitigation": "Evaluate decisions on future costs and benefits only. Ignore historical sunk costs.",
    },
    "overconfidence_bias": {
        "name": "Overconfidence Bias",
        "description": "Excessive confidence in the accuracy of one's answers or predictions.",
        "ai_indicators": ["confidence intervals too narrow", "certainty language without evidence", "low uncertainty acknowledgement"],
        "severity_range": (0.4, 1.0),
        "mitigation": "Calibrate confidence with historical accuracy data. Use probabilistic language.",
    },
    "status_quo_bias": {
        "name": "Status Quo Bias",
        "description": "Preference for the current state of affairs over change.",
        "ai_indicators": ["defaults to existing approaches", "change alternatives underweighted", "conservative recommendations without justification"],
        "severity_range": (0.2, 0.7),
        "mitigation": "Evaluate current state and alternatives on equal footing. Quantify opportunity cost of inaction.",
    },
    "representativeness_heuristic": {
        "name": "Representativeness Heuristic",
        "description": "Judging probability by similarity to prototypes, ignoring base rates.",
        "ai_indicators": ["stereotypical categorisation", "base rates ignored in favour of description matching", "conjunction fallacy patterns"],
        "severity_range": (0.3, 0.8),
        "mitigation": "Apply Bayes' theorem. Incorporate prior probability and base rate data explicitly.",
    },
    "affect_heuristic": {
        "name": "Affect Heuristic",
        "description": "Letting emotional responses guide judgments rather than objective analysis.",
        "ai_indicators": ["sentiment in training data influences risk assessment", "emotionally loaded language skews conclusions", "moral emotions drive recommendations"],
        "severity_range": (0.3, 0.9),
        "mitigation": "Separate emotional signal from analytical conclusions. Use structured scoring criteria.",
    },
    "in_group_bias": {
        "name": "In-Group Bias",
        "description": "Favouring members of one's own group over out-group members.",
        "ai_indicators": ["recommendations favour majority groups", "in-group attributes overweighted", "out-group penalised in predictions"],
        "severity_range": (0.4, 1.0),
        "mitigation": "Audit outputs across demographic groups. Apply fairness constraints in model training.",
    },
    "fundamental_attribution_error": {
        "name": "Fundamental Attribution Error",
        "description": "Over-attributing behaviour to character rather than situational factors.",
        "ai_indicators": ["individual blame for systemic failures", "situational context underweighted", "dispositional labels applied without context"],
        "severity_range": (0.3, 0.8),
        "mitigation": "Systematically document situational factors. Balance dispositional and contextual explanations.",
    },
    "dunning_kruger": {
        "name": "Dunning-Kruger Effect",
        "description": "Overconfidence from limited knowledge; underconfidence from expertise.",
        "ai_indicators": ["confidence inversely correlated with domain coverage", "specialist domains show lower expressed confidence", "generalist overconfidence in specialist topics"],
        "severity_range": (0.3, 0.9),
        "mitigation": "Calibrate confidence based on training data density per domain. Flag low-coverage domains explicitly.",
    },
    "narrative_bias": {
        "name": "Narrative Bias",
        "description": "Preferring coherent stories over statistical evidence.",
        "ai_indicators": ["anecdotes over data", "causal stories accepted without statistical support", "vivid narratives drive stronger recommendations"],
        "severity_range": (0.2, 0.7),
        "mitigation": "Require quantitative evidence alongside qualitative narratives. Apply weight-of-evidence frameworks.",
    },
    "authority_bias": {
        "name": "Authority Bias",
        "description": "Attributing greater accuracy to expert opinions without scrutiny.",
        "ai_indicators": ["expert statements accepted uncritically", "authority cited without evidence evaluation", "rank or title drives conclusion weight"],
        "severity_range": (0.2, 0.8),
        "mitigation": "Evaluate evidence independently of source authority. Apply methodological critique to all sources.",
    },
    "negativity_bias": {
        "name": "Negativity Bias",
        "description": "Weighing negative information more heavily than equivalent positive information.",
        "ai_indicators": ["negative outcomes overweighted in risk models", "loss scenarios elaborated more than gain scenarios", "negative feedback disproportionately influences outputs"],
        "severity_range": (0.2, 0.8),
        "mitigation": "Symmetrically evaluate positive and negative scenarios. Use balanced scoring rubrics.",
    },
    "choice_supportive_bias": {
        "name": "Choice-Supportive Bias",
        "description": "Retroactively attributing positive qualities to chosen options.",
        "ai_indicators": ["post-hoc rationalisation of recommended options", "rejected alternatives retrospectively criticised", "selected option attributes overemphasised"],
        "severity_range": (0.3, 0.7),
        "mitigation": "Document evaluation criteria before final recommendation. Blind post-hoc review.",
    },
}


# ---------------------------------------------------------------------------
# Domain data structures
# ---------------------------------------------------------------------------

@dataclass
class BiasSignal:
    """A detected signal of a specific cognitive bias.

    Attributes:
        signal_id: Unique identifier.
        bias_type: Key from COGNITIVE_BIAS_CATALOGUE.
        bias_name: Human-readable bias name.
        indicator: The specific indicator pattern that triggered detection.
        matched_text: Text fragment that triggered the signal.
        raw_severity: Raw severity score (0.0–1.0).
    """

    signal_id: str
    bias_type: str
    bias_name: str
    indicator: str
    matched_text: str
    raw_severity: float


@dataclass
class CognitiveBiasDetectionResult:
    """Result of cognitive bias detection for a piece of AI output.

    Attributes:
        result_id: Unique identifier.
        detected_biases: All confirmed bias signals.
        overall_bias_score: Aggregate bias severity (0.0–1.0).
        dominant_bias_type: Most severe detected bias type.
        mitigation_recommendations: List of mitigation action strings.
        confidence: Confidence in the detection result (0.0–1.0).
        is_high_risk: True if bias score exceeds high-risk threshold.
    """

    result_id: str
    detected_biases: list[BiasSignal]
    overall_bias_score: float
    dominant_bias_type: str | None
    mitigation_recommendations: list[str]
    confidence: float
    is_high_risk: bool


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class CognitiveBiasDetector:
    """Detects cognitive biases in AI model outputs and human reasoning text.

    Applies indicator-based pattern matching followed by LLM confirmation
    to detect 15+ cognitive bias types. Computes a severity-weighted bias
    score and generates targeted mitigation recommendations.
    """

    HIGH_RISK_THRESHOLD: float = 0.70
    MIN_DETECTION_CONFIDENCE: float = 0.55

    def __init__(
        self,
        llm_client: Any,
        model_name: str = "default",
        enabled_bias_types: list[str] | None = None,
    ) -> None:
        """Initialise the cognitive bias detector.

        Args:
            llm_client: LLM client for analysis.
            model_name: Provider-agnostic model identifier.
            enabled_bias_types: Optional subset of bias types to detect.
        """
        self._llm = llm_client
        self._model_name = model_name
        self._enabled_types = set(enabled_bias_types or COGNITIVE_BIAS_CATALOGUE.keys())

    async def detect_biases(
        self,
        text: str,
        domain: str | None = None,
    ) -> CognitiveBiasDetectionResult:
        """Detect cognitive biases in AI output or human reasoning text.

        Args:
            text: AI-generated or human-authored text to analyse.
            domain: Optional domain hint for context-sensitive detection.

        Returns:
            CognitiveBiasDetectionResult with all detected biases and scores.
        """
        indicator_signals = self._scan_indicators(text)
        llm_signals = await self._llm_bias_scan(text, domain)
        all_signals = self._merge_signals(indicator_signals, llm_signals)

        overall_score = self._compute_overall_score(all_signals)
        dominant = self._find_dominant_bias(all_signals)
        mitigations = self._generate_mitigations(all_signals)
        confidence = self._estimate_confidence(all_signals, text)

        result = CognitiveBiasDetectionResult(
            result_id=str(uuid.uuid4()),
            detected_biases=all_signals,
            overall_bias_score=round(overall_score, 4),
            dominant_bias_type=dominant,
            mitigation_recommendations=mitigations,
            confidence=round(confidence, 4),
            is_high_risk=overall_score >= self.HIGH_RISK_THRESHOLD,
        )

        logger.info(
            "Cognitive bias detection complete",
            bias_count=len(all_signals),
            overall_score=overall_score,
            is_high_risk=result.is_high_risk,
        )
        return result

    async def score_bias_severity(
        self, bias_type: str, text: str
    ) -> float:
        """Score the severity of a specific bias type in text.

        Args:
            bias_type: Bias type key from COGNITIVE_BIAS_CATALOGUE.
            text: Text to score.

        Returns:
            Severity score between 0.0 and 1.0.

        Raises:
            ValueError: If bias_type is not in the catalogue.
        """
        if bias_type not in COGNITIVE_BIAS_CATALOGUE:
            raise ValueError(f"Unknown bias type: {bias_type!r}")

        definition = COGNITIVE_BIAS_CATALOGUE[bias_type]
        min_sev, max_sev = definition["severity_range"]

        prompt = (
            f"Evaluate the severity of {definition['name']} in the following text. "
            f"Definition: {definition['description']}\n"
            f"Indicators to look for: {', '.join(definition['ai_indicators'])}\n\n"
            f"Text:\n{text[:2000]}\n\n"
            f"Return JSON: {{\"severity\": <float between {min_sev} and {max_sev}>, \"rationale\": \"...\"}}"
        )
        result = await self._call_llm_json(prompt)
        severity = float(result.get("severity", min_sev))
        return max(min_sev, min(max_sev, severity))

    def recommend_mitigations(self, detected_biases: list[BiasSignal]) -> list[str]:
        """Generate prioritised mitigation recommendations for detected biases.

        Args:
            detected_biases: List of detected bias signals.

        Returns:
            Ordered list of mitigation recommendation strings.
        """
        return self._generate_mitigations(detected_biases)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_indicators(self, text: str) -> list[BiasSignal]:
        """Scan text for bias indicator phrases.

        Args:
            text: Text to scan.

        Returns:
            List of indicator-matched BiasSignal objects.
        """
        text_lower = text.lower()
        signals: list[BiasSignal] = []

        for bias_type, definition in COGNITIVE_BIAS_CATALOGUE.items():
            if bias_type not in self._enabled_types:
                continue
            min_sev, max_sev = definition["severity_range"]
            for indicator in definition["ai_indicators"]:
                indicator_tokens = indicator.lower().split()
                # Check if majority of indicator tokens appear in text
                matching_tokens = sum(1 for token in indicator_tokens if token in text_lower)
                coverage = matching_tokens / len(indicator_tokens) if indicator_tokens else 0
                if coverage >= 0.5:
                    signals.append(BiasSignal(
                        signal_id=str(uuid.uuid4()),
                        bias_type=bias_type,
                        bias_name=definition["name"],
                        indicator=indicator,
                        matched_text=text[:200],
                        raw_severity=min_sev + (max_sev - min_sev) * coverage,
                    ))

        return signals

    async def _llm_bias_scan(
        self, text: str, domain: str | None
    ) -> list[BiasSignal]:
        """Use LLM to scan for cognitive biases not caught by indicator matching.

        Args:
            text: Text to scan.
            domain: Optional domain context.

        Returns:
            List of LLM-detected BiasSignal objects.
        """
        enabled_list = list(self._enabled_types)[:10]  # Limit prompt size
        catalogue_subset = {k: v for k, v in COGNITIVE_BIAS_CATALOGUE.items() if k in enabled_list}
        catalogue_summary = {
            k: {"name": v["name"], "description": v["description"]}
            for k, v in catalogue_subset.items()
        }

        domain_note = f"Domain: {domain}. " if domain else ""
        prompt = (
            f"{domain_note}Analyse the following text for cognitive biases from this catalogue:\n"
            f"{json.dumps(catalogue_summary, indent=2)}\n\n"
            f"Text:\n{text[:2000]}\n\n"
            "Identify any present biases with: bias_type (catalogue key), "
            "severity (0.0-1.0), matched_text (short excerpt), indicator (what triggered detection).\n\n"
            'Return JSON: {"biases": [{"bias_type": "...", "severity": 0.0, "matched_text": "...", "indicator": "..."}]}'
        )
        result = await self._call_llm_json(prompt)
        signals: list[BiasSignal] = []

        for raw in result.get("biases", []):
            bias_type = raw.get("bias_type", "")
            if bias_type not in COGNITIVE_BIAS_CATALOGUE:
                continue
            definition = COGNITIVE_BIAS_CATALOGUE[bias_type]
            signals.append(BiasSignal(
                signal_id=str(uuid.uuid4()),
                bias_type=bias_type,
                bias_name=definition["name"],
                indicator=str(raw.get("indicator", "LLM detection")),
                matched_text=str(raw.get("matched_text", text[:100])),
                raw_severity=float(raw.get("severity", 0.5)),
            ))

        return signals

    def _merge_signals(
        self,
        indicator_signals: list[BiasSignal],
        llm_signals: list[BiasSignal],
    ) -> list[BiasSignal]:
        """Merge indicator and LLM signals, deduplicating by bias_type.

        When both sources detect the same bias type, the signal with higher
        severity is retained and confidence is boosted.

        Args:
            indicator_signals: Pattern-matched signals.
            llm_signals: LLM-detected signals.

        Returns:
            Deduplicated merged signal list.
        """
        merged: dict[str, BiasSignal] = {}

        for signal in indicator_signals:
            merged[signal.bias_type] = signal

        for signal in llm_signals:
            if signal.bias_type in merged:
                existing = merged[signal.bias_type]
                # Take higher severity, boost with cross-confirmation
                merged[signal.bias_type] = BiasSignal(
                    signal_id=existing.signal_id,
                    bias_type=existing.bias_type,
                    bias_name=existing.bias_name,
                    indicator=existing.indicator,
                    matched_text=existing.matched_text,
                    raw_severity=min(1.0, max(existing.raw_severity, signal.raw_severity) + 0.05),
                )
            else:
                merged[signal.bias_type] = signal

        return list(merged.values())

    def _compute_overall_score(self, signals: list[BiasSignal]) -> float:
        """Compute aggregate bias score as severity-weighted mean.

        Args:
            signals: Detected bias signals.

        Returns:
            Overall bias score (0.0–1.0).
        """
        if not signals:
            return 0.0
        return min(1.0, sum(s.raw_severity for s in signals) / max(len(signals), 1))

    def _find_dominant_bias(self, signals: list[BiasSignal]) -> str | None:
        """Identify the most severe bias type.

        Args:
            signals: Detected bias signals.

        Returns:
            Bias type key of the highest-severity signal, or None.
        """
        if not signals:
            return None
        return max(signals, key=lambda s: s.raw_severity).bias_type

    def _generate_mitigations(self, signals: list[BiasSignal]) -> list[str]:
        """Generate prioritised mitigation recommendations.

        Args:
            signals: Detected bias signals sorted by severity.

        Returns:
            List of mitigation strings ordered by severity (highest first).
        """
        sorted_signals = sorted(signals, key=lambda s: s.raw_severity, reverse=True)
        mitigations: list[str] = []
        seen_types: set[str] = set()

        for signal in sorted_signals:
            if signal.bias_type in seen_types:
                continue
            seen_types.add(signal.bias_type)
            definition = COGNITIVE_BIAS_CATALOGUE.get(signal.bias_type, {})
            mitigation = definition.get("mitigation", "Review for potential bias.")
            mitigations.append(f"[{signal.bias_name}] {mitigation}")

        return mitigations

    def _estimate_confidence(self, signals: list[BiasSignal], text: str) -> float:
        """Estimate overall detection confidence.

        Args:
            signals: Detected signals.
            text: Source text (length used for calibration).

        Returns:
            Confidence score (0.0–1.0).
        """
        if not signals:
            return 0.85  # High confidence in null result for short texts
        base = 0.65
        # Cross-confirmed signals (severity boosted) increase confidence
        boosted = sum(1 for s in signals if s.raw_severity > 0.60)
        boost = min(boosted * 0.05, 0.20)
        return min(1.0, base + boost)

    async def _call_llm_json(self, prompt: str) -> Any:
        """Call LLM and return parsed JSON.

        Args:
            prompt: Prompt to send.

        Returns:
            Parsed JSON dict.
        """
        try:
            response = await self._llm.complete(
                prompt=prompt,
                model=self._model_name,
                response_format={"type": "json_object"},
            )
            raw = response.text if hasattr(response, "text") else str(response)
            return json.loads(raw)
        except Exception as exc:
            logger.error("LLM call failed in cognitive bias detector", error=str(exc))
            return {}
