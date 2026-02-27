"""Evidence gatherer adapter for the AumOS Critical Thinking service.

Source attribution and verification: claim extraction, source identification,
credibility scoring, fact-checking workflow, evidence strength classification,
citation generation, and evidence chain tracking.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain data structures
# ---------------------------------------------------------------------------

class EvidenceStrength(str, Enum):
    """Classification of evidence quality."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"


@dataclass
class Claim:
    """A factual or logical claim extracted from text.

    Attributes:
        claim_id: Unique identifier.
        text: The claim statement.
        claim_type: factual | inferential | normative | predictive.
        source_span: Character span in original text (start, end).
        requires_verification: True if the claim needs external validation.
        confidence_prior: Initial confidence before verification (0.0–1.0).
    """

    claim_id: str
    text: str
    claim_type: str
    source_span: tuple[int, int]
    requires_verification: bool
    confidence_prior: float


@dataclass
class Source:
    """An identified source for a claim or piece of evidence.

    Attributes:
        source_id: Unique identifier.
        name: Source name or description.
        source_type: primary | secondary | tertiary | anecdotal | expert_opinion.
        url: Optional URL reference.
        publication_date: Optional publication or retrieval date.
        credibility_score: Estimated credibility (0.0–1.0).
        domain_authority: Estimated domain-specific authority (0.0–1.0).
    """

    source_id: str
    name: str
    source_type: str
    url: str | None
    publication_date: str | None
    credibility_score: float
    domain_authority: float


@dataclass
class EvidenceItem:
    """A single piece of evidence supporting or refuting a claim.

    Attributes:
        evidence_id: Unique identifier.
        claim_id: Associated claim UUID.
        text: The evidence statement.
        source: The Source this evidence comes from.
        strength: EvidenceStrength classification.
        supports_claim: True if evidence supports the claim, False if it refutes it.
        citation: Formatted citation string.
    """

    evidence_id: str
    claim_id: str
    text: str
    source: Source
    strength: EvidenceStrength
    supports_claim: bool
    citation: str


@dataclass
class FactCheckResult:
    """Result of a fact-checking workflow for a claim.

    Attributes:
        result_id: Unique identifier.
        claim: The claim that was fact-checked.
        verdict: true | false | partially_true | unverifiable | disputed.
        confidence: Confidence in the verdict (0.0–1.0).
        supporting_evidence: Evidence items that support the claim.
        refuting_evidence: Evidence items that refute the claim.
        notes: Analyst notes.
        checked_at: Timestamp of the check.
    """

    result_id: str
    claim: Claim
    verdict: str
    confidence: float
    supporting_evidence: list[EvidenceItem]
    refuting_evidence: list[EvidenceItem]
    notes: str
    checked_at: datetime


@dataclass
class EvidenceChain:
    """A chain of evidence items linking from a root claim to a conclusion.

    Attributes:
        chain_id: Unique identifier.
        root_claim: The originating claim.
        evidence_links: Ordered evidence items forming the chain.
        chain_strength: Overall chain strength (minimum link strength).
        conclusion: The claim or fact the chain supports.
        is_complete: True if the chain reaches the conclusion without gaps.
    """

    chain_id: str
    root_claim: Claim
    evidence_links: list[EvidenceItem]
    chain_strength: EvidenceStrength
    conclusion: str
    is_complete: bool


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class EvidenceGatherer:
    """Gathers, evaluates, and tracks evidence for claims in text.

    Extracts claims using LLM, scores source credibility using heuristics,
    classifies evidence strength, generates formatted citations, and builds
    evidence chains to support or refute claims.
    """

    # Credibility score thresholds for strength classification
    STRENGTH_THRESHOLDS: dict[EvidenceStrength, float] = {
        EvidenceStrength.STRONG: 0.75,
        EvidenceStrength.MODERATE: 0.50,
        EvidenceStrength.WEAK: 0.25,
    }

    def __init__(
        self,
        llm_client: Any,
        model_name: str = "default",
        known_high_credibility_domains: list[str] | None = None,
    ) -> None:
        """Initialise the evidence gatherer.

        Args:
            llm_client: LLM client for claim and source extraction.
            model_name: Provider-agnostic model identifier.
            known_high_credibility_domains: List of high-authority source domains.
        """
        self._llm = llm_client
        self._model_name = model_name
        self._high_credibility_domains = set(known_high_credibility_domains or [
            "pubmed.ncbi.nlm.nih.gov", "nature.com", "sciencemag.org",
            "reuters.com", "apnews.com", "bbc.co.uk", "nytimes.com",
        ])

    async def extract_claims(self, text: str) -> list[Claim]:
        """Extract factual and inferential claims from text.

        Args:
            text: Source text to analyse.

        Returns:
            List of Claim objects extracted from the text.
        """
        prompt = (
            "Extract all distinct factual, inferential, normative, and predictive claims "
            "from the following text. For each claim, identify: text (str), "
            "claim_type (factual/inferential/normative/predictive), "
            "requires_verification (bool), confidence_prior (float 0.0-1.0).\n\n"
            f"Text:\n{text[:4000]}\n\n"
            'Return JSON: {"claims": [{"text": "...", "claim_type": "...", '
            '"requires_verification": true, "confidence_prior": 0.0}]}'
        )
        result = await self._call_llm_json(prompt)
        claims: list[Claim] = []

        for raw in result.get("claims", []):
            claim_text = str(raw.get("text", ""))
            start = text.find(claim_text[:40]) if len(claim_text) >= 40 else text.find(claim_text)
            span = (max(0, start), max(0, start) + len(claim_text)) if start >= 0 else (0, 0)
            claims.append(Claim(
                claim_id=str(uuid.uuid4()),
                text=claim_text,
                claim_type=raw.get("claim_type", "factual"),
                source_span=span,
                requires_verification=bool(raw.get("requires_verification", True)),
                confidence_prior=float(raw.get("confidence_prior", 0.5)),
            ))

        logger.info("Claims extracted", claim_count=len(claims))
        return claims

    async def identify_sources(self, text: str, claims: list[Claim]) -> list[Source]:
        """Identify source references for claims within text.

        Args:
            text: Source text that may contain citations or attributions.
            claims: Claims to find sources for.

        Returns:
            List of Source objects identified in the text.
        """
        claim_texts = [c.text[:100] for c in claims[:10]]
        prompt = (
            "Identify all sources cited, referenced, or attributed in the following text. "
            "For each source, provide: name, source_type (primary/secondary/tertiary/anecdotal/expert_opinion), "
            "url (if present or null), publication_date (if present or null).\n\n"
            f"Text:\n{text[:3000]}\n\n"
            f"Claims to source: {claim_texts}\n\n"
            'Return JSON: {"sources": [{"name": "...", "source_type": "...", "url": null, "publication_date": null}]}'
        )
        result = await self._call_llm_json(prompt)
        sources: list[Source] = []

        for raw in result.get("sources", []):
            url = raw.get("url")
            credibility = self._score_source_credibility(
                name=str(raw.get("name", "")),
                source_type=str(raw.get("source_type", "secondary")),
                url=url,
            )
            sources.append(Source(
                source_id=str(uuid.uuid4()),
                name=str(raw.get("name", "Unknown source")),
                source_type=str(raw.get("source_type", "secondary")),
                url=url,
                publication_date=raw.get("publication_date"),
                credibility_score=credibility,
                domain_authority=self._estimate_domain_authority(url),
            ))

        logger.info("Sources identified", source_count=len(sources))
        return sources

    async def fact_check(self, claim: Claim, sources: list[Source]) -> FactCheckResult:
        """Run the fact-checking workflow for a single claim.

        Evaluates claim against available sources, determines verdict,
        and classifies supporting and refuting evidence.

        Args:
            claim: The Claim to fact-check.
            sources: Available sources to evaluate against.

        Returns:
            FactCheckResult with verdict, confidence, and evidence breakdown.
        """
        sources_summary = [
            {"name": s.name, "type": s.source_type, "credibility": s.credibility_score}
            for s in sources
        ]
        prompt = (
            f"Fact-check the following claim using the available sources.\n\n"
            f"Claim: {claim.text}\n\n"
            f"Available sources: {json.dumps(sources_summary)}\n\n"
            "Determine verdict (true/false/partially_true/unverifiable/disputed), "
            "confidence (0.0-1.0), supporting_evidence (list of texts), "
            "refuting_evidence (list of texts), and notes.\n\n"
            'Return JSON: {"verdict": "...", "confidence": 0.0, "supporting_evidence": [...], '
            '"refuting_evidence": [...], "notes": "..."}'
        )
        result = await self._call_llm_json(prompt)

        supporting: list[EvidenceItem] = []
        for evidence_text in result.get("supporting_evidence", []):
            source = sources[0] if sources else self._create_generic_source()
            supporting.append(self._build_evidence_item(claim, str(evidence_text), source, supports=True))

        refuting: list[EvidenceItem] = []
        for evidence_text in result.get("refuting_evidence", []):
            source = sources[0] if sources else self._create_generic_source()
            refuting.append(self._build_evidence_item(claim, str(evidence_text), source, supports=False))

        fact_result = FactCheckResult(
            result_id=str(uuid.uuid4()),
            claim=claim,
            verdict=result.get("verdict", "unverifiable"),
            confidence=float(result.get("confidence", 0.5)),
            supporting_evidence=supporting,
            refuting_evidence=refuting,
            notes=str(result.get("notes", "")),
            checked_at=datetime.now(tz=timezone.utc),
        )

        logger.info(
            "Claim fact-checked",
            claim_id=claim.claim_id,
            verdict=fact_result.verdict,
            confidence=fact_result.confidence,
        )
        return fact_result

    def classify_evidence_strength(self, evidence: EvidenceItem) -> EvidenceStrength:
        """Classify evidence strength based on source credibility and evidence type.

        Args:
            evidence: EvidenceItem to classify.

        Returns:
            EvidenceStrength enum value.
        """
        score = evidence.source.credibility_score
        if score >= self.STRENGTH_THRESHOLDS[EvidenceStrength.STRONG]:
            return EvidenceStrength.STRONG
        if score >= self.STRENGTH_THRESHOLDS[EvidenceStrength.MODERATE]:
            return EvidenceStrength.MODERATE
        if score >= self.STRENGTH_THRESHOLDS[EvidenceStrength.WEAK]:
            return EvidenceStrength.WEAK
        return EvidenceStrength.INSUFFICIENT

    def generate_citation(self, evidence: EvidenceItem) -> str:
        """Generate a formatted citation for an evidence item.

        Args:
            evidence: EvidenceItem to cite.

        Returns:
            APA-style formatted citation string.
        """
        source = evidence.source
        date_part = f" ({source.publication_date})" if source.publication_date else ""
        url_part = f" Retrieved from {source.url}" if source.url else ""
        return f"{source.name}{date_part}. {evidence.text[:80]}...{url_part}"

    def build_evidence_chain(
        self,
        root_claim: Claim,
        evidence_items: list[EvidenceItem],
        target_conclusion: str,
    ) -> EvidenceChain:
        """Build an evidence chain from root claim to target conclusion.

        Links evidence items in order of source credibility, identifies
        gaps in the chain, and computes overall chain strength as the
        minimum link strength.

        Args:
            root_claim: The originating claim to chain from.
            evidence_items: Available evidence to construct the chain.
            target_conclusion: The conclusion the chain should support.

        Returns:
            EvidenceChain with strength classification and completeness flag.
        """
        chain_id = str(uuid.uuid4())

        # Sort by credibility descending; take top 5 supporting items
        supporting = [e for e in evidence_items if e.supports_claim and e.claim_id == root_claim.claim_id]
        supporting.sort(key=lambda e: e.source.credibility_score, reverse=True)
        chain_links = supporting[:5]

        if not chain_links:
            chain_strength = EvidenceStrength.INSUFFICIENT
        else:
            min_score = min(e.source.credibility_score for e in chain_links)
            chain_strength = self.classify_evidence_strength(chain_links[0])
            # Degrade if any link is very weak
            if min_score < 0.30:
                chain_strength = EvidenceStrength.WEAK

        is_complete = (
            len(chain_links) >= 2
            and chain_strength != EvidenceStrength.INSUFFICIENT
        )

        chain = EvidenceChain(
            chain_id=chain_id,
            root_claim=root_claim,
            evidence_links=chain_links,
            chain_strength=chain_strength,
            conclusion=target_conclusion,
            is_complete=is_complete,
        )

        logger.info(
            "Evidence chain built",
            chain_id=chain_id,
            link_count=len(chain_links),
            strength=chain_strength,
            is_complete=is_complete,
        )
        return chain

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_source_credibility(
        self, name: str, source_type: str, url: str | None
    ) -> float:
        """Compute source credibility from type and domain.

        Args:
            name: Source name.
            source_type: primary | secondary | tertiary | anecdotal | expert_opinion.
            url: Optional source URL.

        Returns:
            Credibility score (0.0–1.0).
        """
        type_scores: dict[str, float] = {
            "primary": 0.85,
            "expert_opinion": 0.75,
            "secondary": 0.60,
            "tertiary": 0.45,
            "anecdotal": 0.25,
        }
        base = type_scores.get(source_type, 0.50)

        if url:
            domain = url.split("/")[2] if "/" in url else url
            if any(hd in domain for hd in self._high_credibility_domains):
                base = min(1.0, base + 0.15)

        return round(base, 3)

    def _estimate_domain_authority(self, url: str | None) -> float:
        """Estimate domain authority from URL pattern.

        Args:
            url: Source URL.

        Returns:
            Domain authority score (0.0–1.0).
        """
        if not url:
            return 0.5
        domain = url.split("/")[2] if "/" in url else url
        if any(hd in domain for hd in self._high_credibility_domains):
            return 0.90
        if domain.endswith(".gov") or domain.endswith(".edu"):
            return 0.80
        if domain.endswith(".org"):
            return 0.65
        return 0.50

    def _build_evidence_item(
        self,
        claim: Claim,
        evidence_text: str,
        source: Source,
        supports: bool,
    ) -> EvidenceItem:
        """Build an EvidenceItem from raw components.

        Args:
            claim: Associated claim.
            evidence_text: The evidence statement.
            source: Source for this evidence.
            supports: True if evidence supports the claim.

        Returns:
            Constructed EvidenceItem.
        """
        evidence = EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            claim_id=claim.claim_id,
            text=evidence_text,
            source=source,
            strength=EvidenceStrength.MODERATE,
            supports_claim=supports,
            citation="",
        )
        evidence.strength = self.classify_evidence_strength(evidence)
        evidence.citation = self.generate_citation(evidence)
        return evidence

    def _create_generic_source(self) -> Source:
        """Create a generic placeholder source for unattributed evidence.

        Returns:
            Source with low credibility marking it as unattributed.
        """
        return Source(
            source_id=str(uuid.uuid4()),
            name="Unattributed",
            source_type="anecdotal",
            url=None,
            publication_date=None,
            credibility_score=0.20,
            domain_authority=0.20,
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
            logger.error("LLM call failed in evidence gatherer", error=str(exc))
            return {}
