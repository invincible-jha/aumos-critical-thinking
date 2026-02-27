"""Argument extractor adapter for the AumOS Critical Thinking service.

Identifies logical argument structure from text: premises, conclusions,
argument types (deductive, inductive, abductive), strength scoring,
supporting evidence linking, counter-argument detection, and argument graphs.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain data structures
# ---------------------------------------------------------------------------

ArgumentType = str  # deductive | inductive | abductive | analogical | causal


@dataclass
class Premise:
    """A single premise in a logical argument.

    Attributes:
        premise_id: Unique identifier.
        text: The premise statement.
        source_span: Character span in the original text (start, end).
        credibility: Estimated credibility of the premise (0.0–1.0).
        supporting_evidence: Associated evidence fragments.
    """

    premise_id: str
    text: str
    source_span: tuple[int, int]
    credibility: float
    supporting_evidence: list[str] = field(default_factory=list)


@dataclass
class Conclusion:
    """A logical conclusion extracted from text.

    Attributes:
        conclusion_id: Unique identifier.
        text: The conclusion statement.
        source_span: Character span in the original text.
        follows_from: Premise IDs that logically lead to this conclusion.
        confidence: Confidence that this is the intended conclusion (0.0–1.0).
    """

    conclusion_id: str
    text: str
    source_span: tuple[int, int]
    follows_from: list[str]
    confidence: float


@dataclass
class Argument:
    """A complete logical argument with premises, conclusion, and metadata.

    Attributes:
        argument_id: Unique identifier.
        premises: Ordered list of Premise objects.
        conclusion: The Conclusion object.
        argument_type: deductive | inductive | abductive | analogical | causal.
        strength_score: Overall argument strength (0.0–1.0).
        counter_arguments: Detected counter-argument texts.
        is_valid: Whether the argument structure is logically sound.
        notes: Analyst notes on argument quality.
    """

    argument_id: str
    premises: list[Premise]
    conclusion: Conclusion
    argument_type: ArgumentType
    strength_score: float
    counter_arguments: list[str] = field(default_factory=list)
    is_valid: bool = True
    notes: str = ""


@dataclass
class ArgumentGraph:
    """Graph representation of arguments and their relationships.

    Attributes:
        graph_id: Unique identifier.
        arguments: All extracted arguments.
        edges: List of (from_argument_id, to_argument_id, relationship_type) tuples.
        central_claim: The dominant claim in the argument cluster.
        cohesion_score: How well arguments support each other (0.0–1.0).
    """

    graph_id: str
    arguments: list[Argument]
    edges: list[tuple[str, str, str]]
    central_claim: str
    cohesion_score: float


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ArgumentExtractor:
    """Extracts and analyses logical arguments from natural language text.

    Combines LLM-based extraction with rule-based validation to identify
    premises, conclusions, argument types, strength scores, and construct
    argument dependency graphs for complex multi-argument texts.
    """

    # Argument type indicator phrases
    DEDUCTIVE_MARKERS: tuple[str, ...] = ("therefore", "thus", "hence", "it follows that", "necessarily")
    INDUCTIVE_MARKERS: tuple[str, ...] = ("probably", "likely", "suggests", "tends to", "in most cases")
    ABDUCTIVE_MARKERS: tuple[str, ...] = ("best explains", "most plausible", "the simplest explanation", "accounts for")
    CAUSAL_MARKERS: tuple[str, ...] = ("causes", "leads to", "results in", "because of", "due to")

    def __init__(self, llm_client: Any, model_name: str = "default") -> None:
        """Initialise the argument extractor.

        Args:
            llm_client: LLM client supporting async completion.
            model_name: Provider-agnostic model identifier.
        """
        self._llm = llm_client
        self._model_name = model_name

    async def extract_arguments(
        self,
        text: str,
        domain_context: str | None = None,
    ) -> list[Argument]:
        """Extract all logical arguments from a body of text.

        Uses LLM to identify argument segments, then validates and scores
        each argument's structure for logical soundness.

        Args:
            text: Source text to analyse.
            domain_context: Optional domain hint (e.g., "legal", "scientific").

        Returns:
            List of Argument objects extracted from the text.
        """
        prompt = self._build_extraction_prompt(text, domain_context)
        raw_args = await self._call_llm_json(prompt)

        arguments: list[Argument] = []
        for raw in (raw_args if isinstance(raw_args, list) else raw_args.get("arguments", [])):
            argument = self._parse_raw_argument(raw, text)
            argument.strength_score = self._score_argument_strength(argument)
            argument.is_valid = self._validate_argument_structure(argument)
            arguments.append(argument)

        logger.info(
            "Arguments extracted",
            argument_count=len(arguments),
            text_length=len(text),
        )
        return arguments

    async def identify_argument_type(self, argument: Argument) -> ArgumentType:
        """Classify an argument as deductive, inductive, abductive, or causal.

        Uses marker phrase detection first; falls back to LLM classification
        for ambiguous cases.

        Args:
            argument: The Argument to classify.

        Returns:
            ArgumentType string classification.
        """
        combined_text = " ".join(p.text for p in argument.premises) + " " + argument.conclusion.text
        combined_lower = combined_text.lower()

        for marker in self.DEDUCTIVE_MARKERS:
            if marker in combined_lower:
                return "deductive"
        for marker in self.CAUSAL_MARKERS:
            if marker in combined_lower:
                return "causal"
        for marker in self.INDUCTIVE_MARKERS:
            if marker in combined_lower:
                return "inductive"
        for marker in self.ABDUCTIVE_MARKERS:
            if marker in combined_lower:
                return "abductive"

        # LLM fallback classification
        prompt = (
            f"Classify the following argument type as exactly one of: "
            "deductive, inductive, abductive, analogical, causal.\n\n"
            f"Premises: {'; '.join(p.text for p in argument.premises)}\n"
            f"Conclusion: {argument.conclusion.text}\n\n"
            'Return JSON: {"argument_type": "<type>"}'
        )
        result = await self._call_llm_json(prompt)
        argument_type = result.get("argument_type", "inductive")
        if argument_type not in ("deductive", "inductive", "abductive", "analogical", "causal"):
            argument_type = "inductive"
        return argument_type

    async def detect_counter_arguments(
        self, argument: Argument, source_text: str
    ) -> list[str]:
        """Detect counter-arguments to a given argument within the source text.

        Args:
            argument: The Argument to find counter-arguments for.
            source_text: The full text that may contain counter-arguments.

        Returns:
            List of counter-argument statement strings.
        """
        prompt = (
            f"Given the following argument:\n"
            f"Premises: {'; '.join(p.text for p in argument.premises)}\n"
            f"Conclusion: {argument.conclusion.text}\n\n"
            f"Identify any counter-arguments present in the text below. "
            "Counter-arguments may rebut premises or challenge the conclusion.\n\n"
            f"Text: {source_text[:3000]}\n\n"
            'Return JSON: {"counter_arguments": ["...", ...]}'
        )
        result = await self._call_llm_json(prompt)
        counter_arguments: list[str] = result.get("counter_arguments", [])
        logger.info("Counter-arguments detected", count=len(counter_arguments))
        return counter_arguments

    async def build_argument_graph(
        self, arguments: list[Argument], source_text: str
    ) -> ArgumentGraph:
        """Construct an argument graph from a list of arguments.

        Identifies argument dependencies (one argument's conclusion serving
        as another's premise), labels edges with relationship types, and
        identifies the central claim.

        Args:
            arguments: List of Argument objects to graph.
            source_text: Original source text for context.

        Returns:
            ArgumentGraph with nodes, edges, and cohesion score.
        """
        graph_id = str(uuid.uuid4())

        # Find edges: conclusion of arg A supports premise of arg B
        edges: list[tuple[str, str, str]] = []
        for arg_a in arguments:
            for arg_b in arguments:
                if arg_a.argument_id == arg_b.argument_id:
                    continue
                for premise in arg_b.premises:
                    similarity = self._text_similarity(arg_a.conclusion.text, premise.text)
                    if similarity > 0.65:
                        edges.append((arg_a.argument_id, arg_b.argument_id, "supports"))

        # Identify central claim: argument with most outgoing support edges
        out_degree: dict[str, int] = {arg.argument_id: 0 for arg in arguments}
        for from_id, _, _ in edges:
            out_degree[from_id] = out_degree.get(from_id, 0) + 1

        central_arg_id = max(out_degree, key=lambda k: out_degree[k]) if out_degree else None
        central_claim = ""
        if central_arg_id:
            central_arg = next((a for a in arguments if a.argument_id == central_arg_id), None)
            if central_arg:
                central_claim = central_arg.conclusion.text

        cohesion_score = self._compute_cohesion(arguments, edges)

        graph = ArgumentGraph(
            graph_id=graph_id,
            arguments=arguments,
            edges=edges,
            central_claim=central_claim,
            cohesion_score=cohesion_score,
        )

        logger.info(
            "Argument graph constructed",
            graph_id=graph_id,
            node_count=len(arguments),
            edge_count=len(edges),
            cohesion=cohesion_score,
        )
        return graph

    def score_argument_strength(self, argument: Argument) -> float:
        """Compute argument strength score from premise quality and structure.

        Args:
            argument: The Argument to score.

        Returns:
            Strength score between 0.0 and 1.0.
        """
        return self._score_argument_strength(argument)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_extraction_prompt(self, text: str, domain_context: str | None) -> str:
        """Build the LLM prompt for argument extraction.

        Args:
            text: Source text.
            domain_context: Optional domain hint.

        Returns:
            Formatted prompt string.
        """
        domain_note = f"Domain context: {domain_context}.\n" if domain_context else ""
        return (
            f"{domain_note}Extract all logical arguments from the following text. "
            "For each argument, identify: premises (list of statements), "
            "conclusion (main claim), argument_type (deductive/inductive/abductive/causal), "
            "and confidence score (0.0-1.0) for the conclusion.\n\n"
            f"Text:\n{text[:4000]}\n\n"
            'Return JSON: {"arguments": [{"premises": [...], "conclusion": "...", '
            '"argument_type": "...", "confidence": 0.0}]}'
        )

    def _parse_raw_argument(self, raw: dict[str, Any], source_text: str) -> Argument:
        """Parse a raw LLM argument dict into an Argument object.

        Args:
            raw: Raw argument dict from LLM.
            source_text: Original source text for span computation.

        Returns:
            Structured Argument object.
        """
        arg_id = str(uuid.uuid4())
        premises: list[Premise] = []

        for i, premise_text in enumerate(raw.get("premises", [])):
            text_str = str(premise_text)
            start = source_text.find(text_str[:40]) if len(text_str) >= 40 else source_text.find(text_str)
            span = (max(0, start), max(0, start) + len(text_str)) if start >= 0 else (0, 0)
            premises.append(Premise(
                premise_id=f"{arg_id}-p{i}",
                text=text_str,
                source_span=span,
                credibility=0.7,
                supporting_evidence=[],
            ))

        conclusion_text = str(raw.get("conclusion", ""))
        conc_start = source_text.find(conclusion_text[:40]) if len(conclusion_text) >= 40 else source_text.find(conclusion_text)
        conc_span = (max(0, conc_start), max(0, conc_start) + len(conclusion_text)) if conc_start >= 0 else (0, 0)

        conclusion = Conclusion(
            conclusion_id=f"{arg_id}-c",
            text=conclusion_text,
            source_span=conc_span,
            follows_from=[p.premise_id for p in premises],
            confidence=float(raw.get("confidence", 0.5)),
        )

        return Argument(
            argument_id=arg_id,
            premises=premises,
            conclusion=conclusion,
            argument_type=raw.get("argument_type", "inductive"),
            strength_score=0.5,
        )

    def _score_argument_strength(self, argument: Argument) -> float:
        """Compute argument strength from premise credibility and conclusion confidence.

        Args:
            argument: The Argument to score.

        Returns:
            Strength score (0.0–1.0).
        """
        if not argument.premises:
            return argument.conclusion.confidence * 0.5

        avg_premise_credibility = sum(p.credibility for p in argument.premises) / len(argument.premises)
        premise_coverage = min(len(argument.premises) / 3.0, 1.0)
        counter_penalty = min(len(argument.counter_arguments) * 0.1, 0.30)

        strength = (
            0.45 * argument.conclusion.confidence
            + 0.35 * avg_premise_credibility
            + 0.20 * premise_coverage
            - counter_penalty
        )
        return max(0.0, min(1.0, strength))

    def _validate_argument_structure(self, argument: Argument) -> bool:
        """Check that an argument has at least one premise and a conclusion.

        Args:
            argument: Argument to validate.

        Returns:
            True if structure is minimally valid.
        """
        return bool(argument.premises) and bool(argument.conclusion.text.strip())

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Compute simple token overlap similarity between two strings.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Jaccard similarity score (0.0–1.0).
        """
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def _compute_cohesion(
        self, arguments: list[Argument], edges: list[tuple[str, str, str]]
    ) -> float:
        """Compute graph cohesion as ratio of actual to possible support edges.

        Args:
            arguments: List of arguments.
            edges: List of (from_id, to_id, relationship) tuples.

        Returns:
            Cohesion score (0.0–1.0).
        """
        n = len(arguments)
        if n < 2:
            return 1.0
        max_edges = n * (n - 1)
        return min(len(edges) / max_edges, 1.0) if max_edges > 0 else 0.0

    async def _call_llm_json(self, prompt: str) -> Any:
        """Call LLM and parse JSON response.

        Args:
            prompt: Prompt to send.

        Returns:
            Parsed JSON (dict or list).
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
            logger.error("LLM JSON call failed in argument extractor", error=str(exc))
            return {}
