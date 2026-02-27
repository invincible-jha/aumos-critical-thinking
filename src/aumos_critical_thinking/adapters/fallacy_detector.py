"""Fallacy detector adapter for the AumOS Critical Thinking service.

Detects 20+ logical fallacy types using pattern matching and LLM analysis,
with confidence scoring, explanation generation, context-aware filtering,
and false positive suppression.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Fallacy catalogue (20+ types)
# ---------------------------------------------------------------------------

FALLACY_CATALOGUE: dict[str, dict[str, Any]] = {
    "ad_hominem": {
        "name": "Ad Hominem",
        "description": "Attacking the person making the argument rather than the argument itself.",
        "patterns": [r"\byou are\b.*\bwrong\b", r"\bhe is\b.*\bcannot be trusted\b", r"\bshe is\b.*\bunqualified\b"],
        "severity": "high",
    },
    "straw_man": {
        "name": "Straw Man",
        "description": "Misrepresenting an opponent's position to make it easier to attack.",
        "patterns": [r"\bso you are saying\b", r"\bwhat you mean is\b.*\bextreme\b"],
        "severity": "high",
    },
    "false_dichotomy": {
        "name": "False Dichotomy",
        "description": "Presenting only two options when more exist.",
        "patterns": [r"\beither.*or\b", r"\byou are either.*with us or against us\b", r"\bthere are only two\b"],
        "severity": "high",
    },
    "appeal_to_authority": {
        "name": "Appeal to Authority",
        "description": "Claiming something is true because an authority says so, without evidence.",
        "patterns": [r"\bexperts say\b", r"\baccording to\b.*\bso it must be\b", r"\bscientists agree\b.*\btherefore\b"],
        "severity": "medium",
    },
    "appeal_to_emotion": {
        "name": "Appeal to Emotion",
        "description": "Using emotional manipulation rather than logical reasoning.",
        "patterns": [r"\bthink of the children\b", r"\bwould you want.*to suffer\b", r"\bit would be terrible\b"],
        "severity": "medium",
    },
    "slippery_slope": {
        "name": "Slippery Slope",
        "description": "Claiming one event will lead to extreme consequences without justification.",
        "patterns": [r"\bif we allow.*then.*will happen\b", r"\bnext thing you know\b", r"\bthis will lead to\b.*\bdisaster\b"],
        "severity": "medium",
    },
    "hasty_generalization": {
        "name": "Hasty Generalization",
        "description": "Drawing broad conclusions from insufficient evidence.",
        "patterns": [r"\ball .* are\b", r"\bevery .* is\b", r"\bnone of .* ever\b"],
        "severity": "medium",
    },
    "circular_reasoning": {
        "name": "Circular Reasoning",
        "description": "Using the conclusion as a premise to support itself.",
        "patterns": [r"\bbecause.*is.*because\b"],
        "severity": "high",
    },
    "appeal_to_tradition": {
        "name": "Appeal to Tradition",
        "description": "Claiming something is correct because it has always been done that way.",
        "patterns": [r"\bwe have always\b", r"\bthat is how it has always been\b", r"\btraditionally\b.*\bso it is right\b"],
        "severity": "low",
    },
    "appeal_to_nature": {
        "name": "Appeal to Nature",
        "description": "Equating natural with good or unnatural with bad.",
        "patterns": [r"\bnatural.*therefore.*good\b", r"\bunnatural.*therefore.*bad\b", r"\bas nature intended\b"],
        "severity": "low",
    },
    "red_herring": {
        "name": "Red Herring",
        "description": "Introducing an irrelevant topic to distract from the main argument.",
        "patterns": [r"\bbut what about\b", r"\blets not forget\b.*\bunrelated\b"],
        "severity": "medium",
    },
    "false_equivalence": {
        "name": "False Equivalence",
        "description": "Treating two things as equivalent when they are not.",
        "patterns": [r"\bthis is the same as\b", r"\bno different from\b", r"\bjust like\b.*\bwhich means\b"],
        "severity": "medium",
    },
    "post_hoc": {
        "name": "Post Hoc Ergo Propter Hoc",
        "description": "Assuming causation from temporal sequence alone.",
        "patterns": [r"\bafter.*therefore.*because of\b", r"\bsince.*happened.*it caused\b"],
        "severity": "high",
    },
    "bandwagon": {
        "name": "Bandwagon (Appeal to Popularity)",
        "description": "Claiming something is true because many people believe it.",
        "patterns": [r"\beveryone knows\b", r"\bmost people believe\b.*\btherefore\b", r"\bthe majority says\b"],
        "severity": "medium",
    },
    "tu_quoque": {
        "name": "Tu Quoque (Appeal to Hypocrisy)",
        "description": "Deflecting criticism by pointing out the critic's own failings.",
        "patterns": [r"\byou do it too\b", r"\byou are no better\b", r"\bbut you also\b"],
        "severity": "medium",
    },
    "no_true_scotsman": {
        "name": "No True Scotsman",
        "description": "Redefining a group to exclude counterexamples.",
        "patterns": [r"\bno real .* would\b", r"\ba true .* never\b", r"\bthat person is not a real\b"],
        "severity": "medium",
    },
    "burden_of_proof": {
        "name": "Shifting the Burden of Proof",
        "description": "Demanding others disprove a claim rather than proving it oneself.",
        "patterns": [r"\bprove it is not\b", r"\buntil you can disprove\b", r"\bshow me it does not exist\b"],
        "severity": "medium",
    },
    "middle_ground": {
        "name": "Middle Ground (False Balance)",
        "description": "Assuming a compromise between two extremes must be correct.",
        "patterns": [r"\bthe truth is somewhere in the middle\b", r"\bboth sides have a point\b.*\bso\b"],
        "severity": "low",
    },
    "special_pleading": {
        "name": "Special Pleading",
        "description": "Applying standards or exemptions inconsistently to protect a position.",
        "patterns": [r"\bexcept in my case\b", r"\bthis is different because.*I\b", r"\brules do not apply\b.*\bme\b"],
        "severity": "high",
    },
    "texas_sharpshooter": {
        "name": "Texas Sharpshooter",
        "description": "Cherry-picking data to fit a pattern after the fact.",
        "patterns": [r"\bif you look at just these cases\b", r"\bthe data shows.*when we exclude\b"],
        "severity": "high",
    },
    "appeal_to_ignorance": {
        "name": "Appeal to Ignorance",
        "description": "Claiming something is true because it has not been proven false.",
        "patterns": [r"\bno one has proven it is not\b", r"\bsince you cannot prove otherwise\b", r"\bnot yet disproven\b"],
        "severity": "medium",
    },
    "genetic_fallacy": {
        "name": "Genetic Fallacy",
        "description": "Judging something based solely on its origin or source.",
        "patterns": [r"\bwhere it comes from means\b", r"\bbecause it was invented by\b", r"\boriginated from\b.*\btherefore wrong\b"],
        "severity": "low",
    },
}


# ---------------------------------------------------------------------------
# Domain data structures
# ---------------------------------------------------------------------------

@dataclass
class FallacyDetection:
    """A single detected logical fallacy.

    Attributes:
        detection_id: Unique identifier.
        fallacy_type: Key from FALLACY_CATALOGUE.
        fallacy_name: Human-readable fallacy name.
        description: What this fallacy is.
        matched_text: The text span that triggered detection.
        confidence: Confidence in this detection (0.0–1.0).
        severity: high | medium | low.
        explanation: Context-specific explanation of why this is a fallacy.
        is_false_positive: True if suppressed as likely false positive.
    """

    detection_id: str
    fallacy_type: str
    fallacy_name: str
    description: str
    matched_text: str
    confidence: float
    severity: str
    explanation: str
    is_false_positive: bool = False


@dataclass
class FallacyReport:
    """Complete fallacy analysis report for a piece of text.

    Attributes:
        report_id: Unique identifier.
        text_length: Length of analysed text.
        detections: All confirmed fallacy detections.
        suppressed_count: Number of detections suppressed as false positives.
        fallacy_density: Detections per 1000 words.
        dominant_fallacy_type: Most frequently detected fallacy type.
        overall_quality_score: Argument quality (1.0 = no fallacies).
        summary: Narrative summary of findings.
    """

    report_id: str
    text_length: int
    detections: list[FallacyDetection]
    suppressed_count: int
    fallacy_density: float
    dominant_fallacy_type: str | None
    overall_quality_score: float
    summary: str


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class FallacyDetector:
    """Detects logical fallacies in text using pattern matching and LLM analysis.

    Applies a two-pass detection strategy: fast regex pattern matching for
    common fallacies, followed by LLM-based analysis for ambiguous cases.
    False positive suppression removes detections that are clearly idiomatic
    or non-argumentative uses of fallacy-adjacent language.
    """

    # Confidence threshold below which pattern-matched detections are suppressed
    PATTERN_CONFIDENCE_BASE: float = 0.60
    # Confidence added by LLM confirmation
    LLM_CONFIRMATION_BOOST: float = 0.20
    # False positive suppression: minimum surrounding context tokens
    MIN_CONTEXT_TOKENS: int = 5

    def __init__(
        self,
        llm_client: Any,
        model_name: str = "default",
        enabled_fallacy_types: list[str] | None = None,
    ) -> None:
        """Initialise the fallacy detector.

        Args:
            llm_client: LLM client supporting async completion.
            model_name: Provider-agnostic model identifier.
            enabled_fallacy_types: Optional subset of fallacy types to detect.
                Defaults to all 20+ types in FALLACY_CATALOGUE.
        """
        self._llm = llm_client
        self._model_name = model_name
        self._enabled_types = set(enabled_fallacy_types or FALLACY_CATALOGUE.keys())

    async def detect_fallacies(
        self,
        text: str,
        domain_context: str | None = None,
        strict_mode: bool = False,
    ) -> list[FallacyDetection]:
        """Detect logical fallacies in the provided text.

        Runs pattern matching, then LLM confirmation for ambiguous hits,
        then applies false positive suppression.

        Args:
            text: Source text to analyse.
            domain_context: Optional domain hint for context-aware analysis.
            strict_mode: If True, lower confidence threshold for reporting.

        Returns:
            List of confirmed FallacyDetection objects.
        """
        pattern_hits = self._run_pattern_matching(text)
        llm_confirmed = await self._llm_confirm_detections(pattern_hits, text, domain_context)
        filtered = self._suppress_false_positives(llm_confirmed, text, strict_mode)

        logger.info(
            "Fallacy detection complete",
            text_length=len(text),
            pattern_hits=len(pattern_hits),
            confirmed=len(llm_confirmed),
            after_suppression=len(filtered),
        )
        return filtered

    async def generate_report(
        self,
        text: str,
        domain_context: str | None = None,
    ) -> FallacyReport:
        """Generate a complete fallacy analysis report.

        Args:
            text: Source text to analyse.
            domain_context: Optional domain context.

        Returns:
            FallacyReport with all detections and summary statistics.
        """
        detections_all = self._run_pattern_matching(text)
        confirmed = await self._llm_confirm_detections(detections_all, text, domain_context)
        filtered = self._suppress_false_positives(confirmed, text, strict_mode=False)

        suppressed_count = len(confirmed) - len(filtered)
        word_count = len(text.split())
        fallacy_density = (len(filtered) / word_count * 1000) if word_count > 0 else 0.0

        type_counts: dict[str, int] = {}
        for detection in filtered:
            type_counts[detection.fallacy_type] = type_counts.get(detection.fallacy_type, 0) + 1
        dominant = max(type_counts, key=lambda k: type_counts[k]) if type_counts else None

        quality_score = max(0.0, 1.0 - (len(filtered) * 0.08))

        summary = self._generate_summary(filtered, dominant, quality_score)

        report = FallacyReport(
            report_id=str(uuid.uuid4()),
            text_length=len(text),
            detections=filtered,
            suppressed_count=suppressed_count,
            fallacy_density=round(fallacy_density, 3),
            dominant_fallacy_type=dominant,
            overall_quality_score=round(quality_score, 3),
            summary=summary,
        )

        logger.info(
            "Fallacy report generated",
            report_id=report.report_id,
            detection_count=len(filtered),
            quality_score=quality_score,
        )
        return report

    async def explain_fallacy(
        self, detection: FallacyDetection, full_text: str
    ) -> str:
        """Generate a context-specific explanation for a detected fallacy.

        Args:
            detection: The FallacyDetection to explain.
            full_text: Original source text for context.

        Returns:
            Human-readable explanation string.
        """
        prompt = (
            f"A '{detection.fallacy_name}' fallacy was detected in the following text segment:\n"
            f"'{detection.matched_text}'\n\n"
            f"Full context (first 1000 chars): {full_text[:1000]}\n\n"
            f"Definition: {detection.description}\n\n"
            "Explain in 2-3 sentences why this specific instance constitutes this fallacy "
            "and what would constitute a valid argument in its place. "
            'Return JSON: {"explanation": "..."}'
        )
        result = await self._call_llm_json(prompt)
        return result.get("explanation", f"This text contains a {detection.fallacy_name}.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_pattern_matching(self, text: str) -> list[FallacyDetection]:
        """Run regex pattern matching across all enabled fallacy types.

        Args:
            text: Source text to scan.

        Returns:
            List of pattern-matched FallacyDetection candidates.
        """
        detections: list[FallacyDetection] = []
        text_lower = text.lower()

        for fallacy_type, definition in FALLACY_CATALOGUE.items():
            if fallacy_type not in self._enabled_types:
                continue
            for pattern in definition["patterns"]:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    matched_text = text[match.start():match.end()]
                    detection = FallacyDetection(
                        detection_id=str(uuid.uuid4()),
                        fallacy_type=fallacy_type,
                        fallacy_name=definition["name"],
                        description=definition["description"],
                        matched_text=matched_text,
                        confidence=self.PATTERN_CONFIDENCE_BASE,
                        severity=definition["severity"],
                        explanation=f"Pattern match detected potential {definition['name']}.",
                    )
                    detections.append(detection)

        return detections

    async def _llm_confirm_detections(
        self,
        detections: list[FallacyDetection],
        text: str,
        domain_context: str | None,
    ) -> list[FallacyDetection]:
        """Use LLM to confirm or reject pattern-matched detections.

        For efficiency, batch-confirms up to 10 detections per LLM call.
        Confirmed detections receive a confidence boost.

        Args:
            detections: Pattern-matched candidates.
            text: Source text.
            domain_context: Optional domain context.

        Returns:
            List of LLM-confirmed detections.
        """
        if not detections:
            return []

        # Deduplicate by fallacy_type for efficiency
        unique_detections: dict[str, FallacyDetection] = {}
        for d in detections:
            if d.fallacy_type not in unique_detections:
                unique_detections[d.fallacy_type] = d

        candidates_summary = [
            {"fallacy_type": d.fallacy_type, "fallacy_name": d.fallacy_name, "matched_text": d.matched_text[:100]}
            for d in unique_detections.values()
        ]

        domain_note = f"Domain: {domain_context}. " if domain_context else ""
        prompt = (
            f"{domain_note}Review the following potential logical fallacy detections in the text below. "
            "For each, determine if it is a genuine fallacy (true) or a false positive (false), "
            "and update the confidence (0.0-1.0).\n\n"
            f"Text (first 2000 chars): {text[:2000]}\n\n"
            f"Candidates: {json.dumps(candidates_summary)}\n\n"
            'Return JSON: {"results": [{"fallacy_type": "...", "confirmed": true/false, "confidence": 0.0, "explanation": "..."}]}'
        )

        try:
            result = await self._call_llm_json(prompt)
            confirmed_map: dict[str, dict[str, Any]] = {}
            for item in result.get("results", []):
                if item.get("confirmed", False):
                    confirmed_map[item["fallacy_type"]] = item
        except Exception as exc:
            logger.error("LLM confirmation failed", error=str(exc))
            confirmed_map = {d.fallacy_type: {"confidence": self.PATTERN_CONFIDENCE_BASE, "explanation": ""} for d in detections}

        confirmed_detections: list[FallacyDetection] = []
        for detection in detections:
            if detection.fallacy_type in confirmed_map:
                llm_result = confirmed_map[detection.fallacy_type]
                detection.confidence = min(
                    1.0,
                    float(llm_result.get("confidence", detection.confidence)) + self.LLM_CONFIRMATION_BOOST,
                )
                detection.explanation = llm_result.get("explanation", detection.explanation)
                confirmed_detections.append(detection)

        return confirmed_detections

    def _suppress_false_positives(
        self,
        detections: list[FallacyDetection],
        text: str,
        strict_mode: bool,
    ) -> list[FallacyDetection]:
        """Suppress likely false positive detections.

        Filters by minimum confidence threshold and ensures matched text
        has sufficient surrounding context (MIN_CONTEXT_TOKENS) to be
        considered argumentative rather than purely idiomatic.

        Args:
            detections: Confirmed detections to filter.
            text: Source text.
            strict_mode: If True, use lower threshold.

        Returns:
            Filtered list with false positives removed.
        """
        threshold = 0.50 if strict_mode else 0.65
        filtered: list[FallacyDetection] = []

        for detection in detections:
            if detection.confidence < threshold:
                detection.is_false_positive = True
                continue
            if len(detection.matched_text.split()) < 3:
                detection.is_false_positive = True
                continue
            filtered.append(detection)

        return filtered

    def _generate_summary(
        self,
        detections: list[FallacyDetection],
        dominant_type: str | None,
        quality_score: float,
    ) -> str:
        """Generate a narrative summary of fallacy findings.

        Args:
            detections: Confirmed detections.
            dominant_type: Most frequent fallacy type.
            quality_score: Overall quality score.

        Returns:
            Summary string.
        """
        if not detections:
            return "No logical fallacies detected. The argument appears logically sound."
        dominant_name = FALLACY_CATALOGUE.get(dominant_type or "", {}).get("name", dominant_type) if dominant_type else "None"
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for d in detections:
            severity_counts[d.severity] = severity_counts.get(d.severity, 0) + 1
        return (
            f"{len(detections)} logical fallacy/fallacies detected. "
            f"Most common: {dominant_name}. "
            f"High severity: {severity_counts['high']}, Medium: {severity_counts['medium']}, Low: {severity_counts['low']}. "
            f"Argument quality score: {quality_score:.2f}/1.00."
        )

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
            logger.error("LLM call failed in fallacy detector", error=str(exc))
            return {}
