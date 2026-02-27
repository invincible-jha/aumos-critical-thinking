"""Debate simulator adapter for the AumOS Critical Thinking service.

Adversarial argumentation: pro/con argument generation, rebuttal creation,
argument strength comparison, debate round management, judge scoring,
Socratic questioning, and debate transcript generation.
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

class DebatePosition(str, Enum):
    """Position in a debate."""

    PRO = "pro"
    CON = "con"
    NEUTRAL = "neutral"


@dataclass
class DebateArgument:
    """A single argument made in a debate round.

    Attributes:
        argument_id: Unique identifier.
        position: PRO or CON position.
        speaker_label: Speaker identifier (e.g., "Pro-1", "Con-1", "Judge").
        content: The argument text.
        argument_type: opening | rebuttal | closing | socratic_question.
        strength_score: Judge-evaluated strength (0.0–1.0).
        round_number: Which debate round this belongs to.
        rebutting_argument_id: ID of the argument being rebutted (for rebuttals).
    """

    argument_id: str
    position: DebatePosition
    speaker_label: str
    content: str
    argument_type: str
    strength_score: float
    round_number: int
    rebutting_argument_id: str | None = None


@dataclass
class JudgeScore:
    """Score assigned by the debate judge.

    Attributes:
        score_id: Unique identifier.
        round_number: Round being scored.
        pro_score: Points awarded to PRO position.
        con_score: Points awarded to CON position.
        criteria_scores: Dict mapping criterion name to (pro_score, con_score) tuples.
        rationale: Judge's scoring rationale.
    """

    score_id: str
    round_number: int
    pro_score: float
    con_score: float
    criteria_scores: dict[str, tuple[float, float]]
    rationale: str


@dataclass
class DebateTranscript:
    """Complete transcript of a simulated debate.

    Attributes:
        transcript_id: Unique identifier.
        motion: The debate motion or proposition.
        rounds: All debate rounds (list of DebateArgument lists).
        judge_scores: Per-round judge scores.
        socratic_questions: Socratic questions generated during the debate.
        final_verdict: pro_wins | con_wins | draw.
        final_pro_score: Cumulative PRO score.
        final_con_score: Cumulative CON score.
        debate_summary: Narrative summary of the debate.
        started_at: Debate start timestamp.
        completed_at: Debate completion timestamp.
    """

    transcript_id: str
    motion: str
    rounds: list[list[DebateArgument]]
    judge_scores: list[JudgeScore]
    socratic_questions: list[str]
    final_verdict: str
    final_pro_score: float
    final_con_score: float
    debate_summary: str
    started_at: datetime
    completed_at: datetime


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class DebateSimulator:
    """Simulates structured adversarial debates on propositions.

    Generates pro and con arguments, rebuttals, and Socratic questions via LLM.
    Implements debate round management with judge scoring across multiple
    criteria. Produces a complete transcript with final verdict.
    """

    # Default number of debate rounds
    DEFAULT_ROUNDS: int = 3
    # Judge scoring criteria and their weights
    SCORING_CRITERIA: dict[str, float] = {
        "logical_coherence": 0.30,
        "evidence_quality": 0.25,
        "rebuttal_effectiveness": 0.25,
        "clarity": 0.20,
    }

    def __init__(
        self,
        llm_client: Any,
        model_name: str = "default",
        max_rounds: int = 5,
    ) -> None:
        """Initialise the debate simulator.

        Args:
            llm_client: LLM client for argument generation.
            model_name: Provider-agnostic model identifier.
            max_rounds: Maximum debate rounds allowed.
        """
        self._llm = llm_client
        self._model_name = model_name
        self._max_rounds = min(max_rounds, 8)

    async def generate_opening_arguments(
        self,
        motion: str,
        context: str | None = None,
    ) -> tuple[DebateArgument, DebateArgument]:
        """Generate opening arguments for both PRO and CON positions.

        Args:
            motion: The debate proposition.
            context: Optional background context.

        Returns:
            Tuple of (pro_argument, con_argument) for the opening round.
        """
        context_note = f"\nContext: {context}" if context else ""
        prompt = (
            f"Generate opening debate arguments for the following motion:\n"
            f"Motion: {motion}{context_note}\n\n"
            "Produce a strong opening argument for PRO (supporting the motion) and "
            "a strong opening argument for CON (opposing the motion). "
            "Each argument should be 3-5 sentences with clear reasoning.\n\n"
            'Return JSON: {"pro": {"content": "...", "strength_score": 0.0}, '
            '"con": {"content": "...", "strength_score": 0.0}}'
        )
        result = await self._call_llm_json(prompt)

        pro_raw = result.get("pro", {})
        con_raw = result.get("con", {})

        pro_arg = DebateArgument(
            argument_id=str(uuid.uuid4()),
            position=DebatePosition.PRO,
            speaker_label="Pro-1",
            content=str(pro_raw.get("content", "The motion is supported by strong evidence.")),
            argument_type="opening",
            strength_score=float(pro_raw.get("strength_score", 0.6)),
            round_number=1,
        )
        con_arg = DebateArgument(
            argument_id=str(uuid.uuid4()),
            position=DebatePosition.CON,
            speaker_label="Con-1",
            content=str(con_raw.get("content", "The motion is opposed by compelling counter-evidence.")),
            argument_type="opening",
            strength_score=float(con_raw.get("strength_score", 0.6)),
            round_number=1,
        )

        logger.info("Opening arguments generated", motion=motion[:80])
        return pro_arg, con_arg

    async def generate_rebuttal(
        self,
        argument_to_rebut: DebateArgument,
        rebutting_position: DebatePosition,
        motion: str,
        round_number: int,
    ) -> DebateArgument:
        """Generate a rebuttal to an opponent's argument.

        Args:
            argument_to_rebut: The argument being rebutted.
            rebutting_position: The position generating the rebuttal.
            motion: The debate motion.
            round_number: Current debate round number.

        Returns:
            A rebuttal DebateArgument.
        """
        prompt = (
            f"Generate a rebuttal to the following {argument_to_rebut.position.value} argument "
            f"in a debate on: '{motion}'\n\n"
            f"Argument to rebut: {argument_to_rebut.content}\n\n"
            f"Generate a {rebutting_position.value} rebuttal that directly addresses the opponent's "
            "main points and presents counter-evidence or logical challenges. "
            "3-4 sentences.\n\n"
            'Return JSON: {"content": "...", "strength_score": 0.0}'
        )
        result = await self._call_llm_json(prompt)

        rebuttal = DebateArgument(
            argument_id=str(uuid.uuid4()),
            position=rebutting_position,
            speaker_label=f"{rebutting_position.value.capitalize()}-{round_number}",
            content=str(result.get("content", "The previous argument overlooks key counter-evidence.")),
            argument_type="rebuttal",
            strength_score=float(result.get("strength_score", 0.55)),
            round_number=round_number,
            rebutting_argument_id=argument_to_rebut.argument_id,
        )

        logger.debug(
            "Rebuttal generated",
            position=rebutting_position.value,
            round=round_number,
        )
        return rebuttal

    async def generate_socratic_questions(
        self,
        motion: str,
        argument: DebateArgument,
        count: int = 3,
    ) -> list[str]:
        """Generate Socratic questions to probe an argument's assumptions.

        Args:
            motion: The debate motion.
            argument: The argument to probe.
            count: Number of Socratic questions to generate.

        Returns:
            List of Socratic question strings.
        """
        prompt = (
            f"Generate {count} Socratic questions to probe the assumptions and reasoning in "
            f"the following argument about: '{motion}'\n\n"
            f"Argument: {argument.content}\n\n"
            "Questions should expose hidden assumptions, challenge evidence quality, "
            "and invite deeper reasoning. Each should be a single clear question.\n\n"
            f'Return JSON: {{"questions": [<{count} question strings>]}}'
        )
        result = await self._call_llm_json(prompt)
        questions = result.get("questions", [])
        return [str(q) for q in questions[:count]]

    async def judge_round(
        self,
        pro_argument: DebateArgument,
        con_argument: DebateArgument,
        round_number: int,
    ) -> JudgeScore:
        """Score a debate round using multiple criteria.

        Args:
            pro_argument: PRO side argument for this round.
            con_argument: CON side argument for this round.
            round_number: Round being scored.

        Returns:
            JudgeScore with per-criterion and aggregate scores.
        """
        criteria_list = list(self.SCORING_CRITERIA.keys())
        prompt = (
            f"As an impartial judge, score the following debate round on these criteria: "
            f"{', '.join(criteria_list)}.\n\n"
            f"PRO argument: {pro_argument.content}\n\n"
            f"CON argument: {con_argument.content}\n\n"
            "For each criterion, assign a score to PRO (0.0-10.0) and CON (0.0-10.0). "
            "Provide a brief rationale.\n\n"
            "Return JSON: {\"criteria\": {"
            + ", ".join(f'"{c}": {{"pro": 0.0, "con": 0.0}}' for c in criteria_list)
            + "}, \"rationale\": \"...\"}"
        )
        result = await self._call_llm_json(prompt)

        criteria_scores: dict[str, tuple[float, float]] = {}
        raw_criteria = result.get("criteria", {})
        for criterion, weight in self.SCORING_CRITERIA.items():
            raw = raw_criteria.get(criterion, {"pro": 5.0, "con": 5.0})
            criteria_scores[criterion] = (float(raw.get("pro", 5.0)), float(raw.get("con", 5.0)))

        pro_total = sum(
            score[0] * self.SCORING_CRITERIA[criterion]
            for criterion, score in criteria_scores.items()
        )
        con_total = sum(
            score[1] * self.SCORING_CRITERIA[criterion]
            for criterion, score in criteria_scores.items()
        )

        judge_score = JudgeScore(
            score_id=str(uuid.uuid4()),
            round_number=round_number,
            pro_score=round(pro_total, 3),
            con_score=round(con_total, 3),
            criteria_scores=criteria_scores,
            rationale=str(result.get("rationale", "Balanced round.")),
        )

        logger.info(
            "Round judged",
            round=round_number,
            pro_score=pro_total,
            con_score=con_total,
        )
        return judge_score

    async def run_debate(
        self,
        motion: str,
        rounds: int | None = None,
        context: str | None = None,
    ) -> DebateTranscript:
        """Run a complete multi-round debate simulation.

        Args:
            motion: The debate proposition.
            rounds: Number of debate rounds (default: DEFAULT_ROUNDS).
            context: Optional background context.

        Returns:
            Complete DebateTranscript with all rounds, scores, and verdict.
        """
        num_rounds = min(rounds or self.DEFAULT_ROUNDS, self._max_rounds)
        started_at = datetime.now(tz=timezone.utc)
        transcript_id = str(uuid.uuid4())

        all_rounds: list[list[DebateArgument]] = []
        judge_scores: list[JudgeScore] = []
        socratic_questions: list[str] = []

        logger.info("Debate simulation started", motion=motion[:80], rounds=num_rounds)

        # Round 1: Opening arguments
        pro_arg, con_arg = await self.generate_opening_arguments(motion, context)
        all_rounds.append([pro_arg, con_arg])

        # Judge round 1
        round_score = await self.judge_round(pro_arg, con_arg, round_number=1)
        judge_scores.append(round_score)

        # Socratic questions after round 1
        questions = await self.generate_socratic_questions(motion, pro_arg, count=2)
        socratic_questions.extend(questions)

        # Subsequent rebuttal rounds
        last_pro = pro_arg
        last_con = con_arg

        for round_num in range(2, num_rounds + 1):
            con_rebuttal = await self.generate_rebuttal(last_pro, DebatePosition.CON, motion, round_num)
            pro_rebuttal = await self.generate_rebuttal(last_con, DebatePosition.PRO, motion, round_num)
            all_rounds.append([pro_rebuttal, con_rebuttal])

            round_score = await self.judge_round(pro_rebuttal, con_rebuttal, round_num)
            judge_scores.append(round_score)

            last_pro = pro_rebuttal
            last_con = con_rebuttal

        # Compute final scores
        final_pro = sum(s.pro_score for s in judge_scores)
        final_con = sum(s.con_score for s in judge_scores)

        if final_pro > final_con + 0.5:
            verdict = "pro_wins"
        elif final_con > final_pro + 0.5:
            verdict = "con_wins"
        else:
            verdict = "draw"

        summary = self._generate_debate_summary(motion, judge_scores, final_pro, final_con, verdict)
        completed_at = datetime.now(tz=timezone.utc)

        transcript = DebateTranscript(
            transcript_id=transcript_id,
            motion=motion,
            rounds=all_rounds,
            judge_scores=judge_scores,
            socratic_questions=socratic_questions,
            final_verdict=verdict,
            final_pro_score=round(final_pro, 3),
            final_con_score=round(final_con, 3),
            debate_summary=summary,
            started_at=started_at,
            completed_at=completed_at,
        )

        logger.info(
            "Debate complete",
            transcript_id=transcript_id,
            verdict=verdict,
            pro_score=final_pro,
            con_score=final_con,
        )
        return transcript

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_debate_summary(
        self,
        motion: str,
        judge_scores: list[JudgeScore],
        final_pro: float,
        final_con: float,
        verdict: str,
    ) -> str:
        """Generate a narrative debate summary.

        Args:
            motion: The debate motion.
            judge_scores: All round scores.
            final_pro: Total PRO score.
            final_con: Total CON score.
            verdict: Final verdict string.

        Returns:
            Summary narrative string.
        """
        verdict_text = {
            "pro_wins": "The PRO position prevailed",
            "con_wins": "The CON position prevailed",
            "draw": "The debate ended in a draw",
        }.get(verdict, "The debate concluded")

        return (
            f"Debate on motion: '{motion}'. {len(judge_scores)} rounds completed. "
            f"{verdict_text} with a final score of PRO {final_pro:.2f} vs CON {final_con:.2f}. "
            f"The deciding factors were argument strength and evidence quality in the later rounds."
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
            logger.error("LLM call failed in debate simulator", error=str(exc))
            return {}
