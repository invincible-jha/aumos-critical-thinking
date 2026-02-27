"""Structured reasoning framework adapter for the AumOS Critical Thinking service.

Implements chain-of-thought step decomposition, tree-of-thought branching,
reasoning step validation, backtracking, multi-path comparison, and best-path
selection via LLM-backed reasoning traces.
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
class ReasoningStep:
    """A single step in a reasoning chain.

    Attributes:
        step_id: Unique identifier for this step.
        description: Human-readable description of the reasoning action.
        evidence: Supporting evidence or premises for this step.
        conclusion: The conclusion derived in this step.
        confidence: Confidence score for this step (0.0–1.0).
        is_valid: Whether this step passes validation checks.
        parent_step_id: Parent step ID for tree-of-thought branching.
    """

    step_id: str
    description: str
    evidence: list[str]
    conclusion: str
    confidence: float
    is_valid: bool = True
    parent_step_id: str | None = None


@dataclass
class ReasoningPath:
    """A complete reasoning path through a problem.

    Attributes:
        path_id: Unique path identifier.
        steps: Ordered sequence of reasoning steps.
        final_conclusion: Terminal conclusion for this path.
        path_score: Aggregate quality score (0.0–1.0).
        is_valid: Whether all steps passed validation.
        depth: Number of reasoning steps.
        backtrack_count: Number of backtrack events during construction.
    """

    path_id: str
    steps: list[ReasoningStep]
    final_conclusion: str
    path_score: float
    is_valid: bool
    depth: int
    backtrack_count: int = 0


@dataclass
class ReasoningTrace:
    """Serialized trace of a full reasoning session.

    Attributes:
        trace_id: Unique trace identifier.
        problem_statement: The original problem or question.
        reasoning_mode: chain_of_thought | tree_of_thought.
        paths: All explored reasoning paths.
        best_path: The selected best path after comparison.
        total_steps_explored: Count of all steps across all paths.
        created_at: Trace creation timestamp.
        metadata: Supplementary metadata.
    """

    trace_id: str
    problem_statement: str
    reasoning_mode: str
    paths: list[ReasoningPath]
    best_path: ReasoningPath | None
    total_steps_explored: int
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ReasoningFramework:
    """Structured reasoning engine implementing chain-of-thought and tree-of-thought.

    Provides step-by-step reasoning decomposition, validates each reasoning step
    against logical consistency rules, supports backtracking when invalid conclusions
    are detected, and selects the best reasoning path from multiple candidates.
    """

    # Minimum confidence threshold below which a step is marked invalid
    STEP_CONFIDENCE_THRESHOLD: float = 0.35
    # Maximum tree branches to explore in tree-of-thought mode
    MAX_BRANCHES: int = 5
    # Maximum depth per branch to prevent runaway exploration
    MAX_DEPTH: int = 12

    def __init__(self, llm_client: Any, model_name: str = "default") -> None:
        """Initialise the reasoning framework.

        Args:
            llm_client: LLM client that supports async completion calls.
            model_name: Provider-agnostic model identifier.
        """
        self._llm = llm_client
        self._model_name = model_name

    async def decompose_chain_of_thought(
        self,
        problem_statement: str,
        context: dict[str, Any] | None = None,
        max_steps: int = 8,
    ) -> ReasoningPath:
        """Decompose a problem using linear chain-of-thought reasoning.

        Generates sequential reasoning steps where each step builds on the
        previous conclusion, validates each step, and backtracks if a step
        is found to be invalid.

        Args:
            problem_statement: The problem or question to reason through.
            context: Optional domain context to prime the reasoning.
            max_steps: Maximum number of chain steps to generate.

        Returns:
            A validated ReasoningPath representing the reasoning chain.
        """
        path_id = str(uuid.uuid4())
        steps: list[ReasoningStep] = []
        backtrack_count = 0
        prior_conclusions: list[str] = []

        logger.info(
            "Starting chain-of-thought decomposition",
            path_id=path_id,
            max_steps=max_steps,
        )

        for step_index in range(min(max_steps, self.MAX_DEPTH)):
            prompt = self._build_cot_step_prompt(
                problem_statement=problem_statement,
                step_index=step_index,
                prior_conclusions=prior_conclusions,
                context=context or {},
            )

            raw_step = await self._call_llm_structured(prompt, expected_keys=["description", "evidence", "conclusion", "confidence"])

            step = ReasoningStep(
                step_id=f"{path_id}-step-{step_index}",
                description=raw_step.get("description", f"Step {step_index + 1}"),
                evidence=raw_step.get("evidence", []),
                conclusion=raw_step.get("conclusion", ""),
                confidence=float(raw_step.get("confidence", 0.5)),
                parent_step_id=steps[-1].step_id if steps else None,
            )

            is_valid, validation_note = self._validate_step(step, prior_conclusions)
            step.is_valid = is_valid

            if not is_valid:
                logger.warning(
                    "Reasoning step failed validation — backtracking",
                    step_id=step.step_id,
                    note=validation_note,
                )
                backtrack_count += 1
                if steps:
                    prior_conclusions.pop()
                    steps.pop()
                continue

            steps.append(step)
            prior_conclusions.append(step.conclusion)

            if self._is_terminal_conclusion(step.conclusion):
                logger.info("Terminal conclusion reached", step_id=step.step_id)
                break

        final_conclusion = steps[-1].conclusion if steps else "No valid conclusion reached."
        path_score = self._compute_path_score(steps)

        path = ReasoningPath(
            path_id=path_id,
            steps=steps,
            final_conclusion=final_conclusion,
            path_score=path_score,
            is_valid=all(s.is_valid for s in steps),
            depth=len(steps),
            backtrack_count=backtrack_count,
        )

        logger.info(
            "Chain-of-thought complete",
            path_id=path_id,
            depth=len(steps),
            score=path_score,
            backtracks=backtrack_count,
        )
        return path

    async def explore_tree_of_thought(
        self,
        problem_statement: str,
        context: dict[str, Any] | None = None,
        branch_factor: int = 3,
        max_depth: int = 6,
    ) -> list[ReasoningPath]:
        """Explore multiple reasoning branches using tree-of-thought.

        Generates branching reasoning paths from a root step, evaluates each
        branch independently, and returns all valid paths for comparison.

        Args:
            problem_statement: The problem or question to reason through.
            context: Optional domain context.
            branch_factor: Number of alternative branches at each node (max 5).
            max_depth: Maximum depth for each branch.

        Returns:
            List of ReasoningPath instances representing all explored branches.
        """
        branch_factor = min(branch_factor, self.MAX_BRANCHES)
        max_depth = min(max_depth, self.MAX_DEPTH)

        logger.info(
            "Starting tree-of-thought exploration",
            branches=branch_factor,
            max_depth=max_depth,
        )

        root_prompt = self._build_tot_branch_prompt(
            problem_statement=problem_statement,
            context=context or {},
            branch_count=branch_factor,
        )
        root_branches = await self._call_llm_branches(root_prompt, count=branch_factor)

        all_paths: list[ReasoningPath] = []

        for branch_index, branch_seed in enumerate(root_branches):
            path_id = str(uuid.uuid4())
            steps: list[ReasoningStep] = []
            backtrack_count = 0
            prior_conclusions: list[str] = [branch_seed.get("conclusion", "")]

            root_step = ReasoningStep(
                step_id=f"{path_id}-step-0",
                description=branch_seed.get("description", f"Branch {branch_index} root"),
                evidence=branch_seed.get("evidence", []),
                conclusion=branch_seed.get("conclusion", ""),
                confidence=float(branch_seed.get("confidence", 0.5)),
                is_valid=True,
                parent_step_id=None,
            )
            steps.append(root_step)

            for depth in range(1, max_depth):
                continuation_prompt = self._build_cot_step_prompt(
                    problem_statement=problem_statement,
                    step_index=depth,
                    prior_conclusions=prior_conclusions,
                    context=context or {},
                )
                raw_step = await self._call_llm_structured(
                    continuation_prompt,
                    expected_keys=["description", "evidence", "conclusion", "confidence"],
                )

                step = ReasoningStep(
                    step_id=f"{path_id}-step-{depth}",
                    description=raw_step.get("description", f"Step {depth}"),
                    evidence=raw_step.get("evidence", []),
                    conclusion=raw_step.get("conclusion", ""),
                    confidence=float(raw_step.get("confidence", 0.5)),
                    parent_step_id=steps[-1].step_id,
                )

                is_valid, _ = self._validate_step(step, prior_conclusions)
                step.is_valid = is_valid

                if not is_valid:
                    backtrack_count += 1
                    break

                steps.append(step)
                prior_conclusions.append(step.conclusion)

                if self._is_terminal_conclusion(step.conclusion):
                    break

            final_conclusion = steps[-1].conclusion if steps else "Branch terminated without conclusion."
            path_score = self._compute_path_score(steps)

            path = ReasoningPath(
                path_id=path_id,
                steps=steps,
                final_conclusion=final_conclusion,
                path_score=path_score,
                is_valid=all(s.is_valid for s in steps),
                depth=len(steps),
                backtrack_count=backtrack_count,
            )
            all_paths.append(path)

        logger.info("Tree-of-thought exploration complete", paths_explored=len(all_paths))
        return all_paths

    def select_best_path(self, paths: list[ReasoningPath]) -> ReasoningPath | None:
        """Select the best reasoning path from multiple candidates.

        Scores paths by: validity (disqualifies invalid paths), path_score
        (weighted average step confidence), depth (prefers more thorough paths
        up to a point), and backtrack_count (penalises heavy backtracking).

        Args:
            paths: List of ReasoningPath candidates.

        Returns:
            The highest-scoring valid ReasoningPath, or None if all paths are invalid.
        """
        valid_paths = [p for p in paths if p.is_valid and p.depth > 0]
        if not valid_paths:
            logger.warning("No valid paths to select from", total_paths=len(paths))
            return None

        def path_rank(path: ReasoningPath) -> float:
            depth_bonus = min(path.depth / 8.0, 0.15)
            backtrack_penalty = min(path.backtrack_count * 0.05, 0.20)
            return path.path_score + depth_bonus - backtrack_penalty

        best = max(valid_paths, key=path_rank)
        logger.info("Best path selected", path_id=best.path_id, score=best.path_score)
        return best

    def compare_paths(self, paths: list[ReasoningPath]) -> list[dict[str, Any]]:
        """Generate a comparison matrix across multiple reasoning paths.

        Args:
            paths: List of ReasoningPath candidates to compare.

        Returns:
            List of comparison dicts with path_id, score, depth, backtrack_count,
            is_valid, final_conclusion, and rank.
        """
        comparison = []
        for rank, path in enumerate(sorted(paths, key=lambda p: p.path_score, reverse=True), start=1):
            comparison.append({
                "rank": rank,
                "path_id": path.path_id,
                "score": round(path.path_score, 4),
                "depth": path.depth,
                "backtrack_count": path.backtrack_count,
                "is_valid": path.is_valid,
                "final_conclusion": path.final_conclusion,
            })
        return comparison

    async def create_reasoning_trace(
        self,
        problem_statement: str,
        reasoning_mode: str = "chain_of_thought",
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReasoningTrace:
        """Execute full reasoning and return a serialized trace.

        Args:
            problem_statement: The problem or question to reason through.
            reasoning_mode: chain_of_thought | tree_of_thought.
            context: Optional domain context.
            metadata: Optional supplementary metadata.

        Returns:
            ReasoningTrace with all paths and selected best path.

        Raises:
            ValueError: If reasoning_mode is not recognized.
        """
        if reasoning_mode not in ("chain_of_thought", "tree_of_thought"):
            raise ValueError(f"Unknown reasoning_mode: {reasoning_mode!r}")

        trace_id = str(uuid.uuid4())

        if reasoning_mode == "chain_of_thought":
            single_path = await self.decompose_chain_of_thought(problem_statement, context)
            paths = [single_path]
        else:
            paths = await self.explore_tree_of_thought(problem_statement, context)

        best_path = self.select_best_path(paths)
        total_steps = sum(p.depth for p in paths)

        trace = ReasoningTrace(
            trace_id=trace_id,
            problem_statement=problem_statement,
            reasoning_mode=reasoning_mode,
            paths=paths,
            best_path=best_path,
            total_steps_explored=total_steps,
            created_at=datetime.now(tz=timezone.utc),
            metadata=metadata or {},
        )

        logger.info(
            "Reasoning trace created",
            trace_id=trace_id,
            mode=reasoning_mode,
            total_paths=len(paths),
            total_steps=total_steps,
        )
        return trace

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_step(
        self, step: ReasoningStep, prior_conclusions: list[str]
    ) -> tuple[bool, str]:
        """Validate a reasoning step for logical consistency.

        Checks: non-empty conclusion, minimum confidence threshold, and
        that the step does not directly contradict prior conclusions via
        negation heuristics.

        Args:
            step: The ReasoningStep to validate.
            prior_conclusions: List of prior conclusions in the chain.

        Returns:
            Tuple of (is_valid, validation_note).
        """
        if not step.conclusion.strip():
            return False, "Empty conclusion"
        if step.confidence < self.STEP_CONFIDENCE_THRESHOLD:
            return False, f"Confidence {step.confidence:.2f} below threshold {self.STEP_CONFIDENCE_THRESHOLD}"
        conclusion_lower = step.conclusion.lower()
        for prior in prior_conclusions:
            prior_lower = prior.lower()
            negated = f"not {prior_lower}" in conclusion_lower or f"no {prior_lower}" in conclusion_lower
            if negated and len(prior_lower) > 20:
                return False, f"Conclusion appears to contradict prior: {prior[:60]}"
        return True, "Valid"

    def _is_terminal_conclusion(self, conclusion: str) -> bool:
        """Detect terminal conclusion markers in the conclusion text.

        Args:
            conclusion: The conclusion text to check.

        Returns:
            True if the conclusion signals the end of the reasoning chain.
        """
        terminal_markers = [
            "therefore,", "thus,", "in conclusion,", "final answer:",
            "conclusion:", "we conclude", "the answer is",
        ]
        conclusion_lower = conclusion.lower()
        return any(marker in conclusion_lower for marker in terminal_markers)

    def _compute_path_score(self, steps: list[ReasoningStep]) -> float:
        """Compute aggregate quality score for a reasoning path.

        Args:
            steps: Ordered list of ReasoningStep instances.

        Returns:
            Weighted average confidence score (0.0–1.0).
        """
        if not steps:
            return 0.0
        valid_steps = [s for s in steps if s.is_valid]
        if not valid_steps:
            return 0.0
        return sum(s.confidence for s in valid_steps) / len(valid_steps)

    def _build_cot_step_prompt(
        self,
        problem_statement: str,
        step_index: int,
        prior_conclusions: list[str],
        context: dict[str, Any],
    ) -> str:
        """Build the prompt for a single chain-of-thought step.

        Args:
            problem_statement: The original problem.
            step_index: Current step index.
            prior_conclusions: Conclusions established so far.
            context: Domain context dict.

        Returns:
            Formatted prompt string.
        """
        prior_text = "\n".join(f"- {c}" for c in prior_conclusions) if prior_conclusions else "None yet."
        context_text = "\n".join(f"{k}: {v}" for k, v in context.items()) if context else "No additional context."
        return (
            f"Problem: {problem_statement}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Prior conclusions established:\n{prior_text}\n\n"
            f"Generate reasoning step {step_index + 1}. Return JSON with keys: "
            "description (str), evidence (list[str]), conclusion (str), confidence (float 0.0-1.0)."
        )

    def _build_tot_branch_prompt(
        self,
        problem_statement: str,
        context: dict[str, Any],
        branch_count: int,
    ) -> str:
        """Build prompt for tree-of-thought initial branching.

        Args:
            problem_statement: The original problem.
            context: Domain context dict.
            branch_count: Number of distinct branches to generate.

        Returns:
            Formatted prompt string.
        """
        context_text = "\n".join(f"{k}: {v}" for k, v in context.items()) if context else "No additional context."
        return (
            f"Problem: {problem_statement}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Generate {branch_count} distinct initial reasoning approaches. "
            f"Return JSON array of {branch_count} objects, each with keys: "
            "description (str), evidence (list[str]), conclusion (str), confidence (float 0.0-1.0)."
        )

    async def _call_llm_structured(
        self, prompt: str, expected_keys: list[str]
    ) -> dict[str, Any]:
        """Call the LLM and parse structured JSON output.

        Args:
            prompt: The prompt to send to the LLM.
            expected_keys: Expected top-level keys to validate presence.

        Returns:
            Parsed dict with at minimum the expected keys present.
        """
        try:
            response = await self._llm.complete(
                prompt=prompt,
                model=self._model_name,
                response_format={"type": "json_object"},
            )
            import json
            result = json.loads(response.text if hasattr(response, "text") else str(response))
            for key in expected_keys:
                if key not in result:
                    result[key] = "" if key in ("description", "conclusion") else ([] if key == "evidence" else 0.5)
            return result
        except Exception as exc:
            logger.error("LLM call failed in reasoning framework", error=str(exc))
            return {key: ("" if key in ("description", "conclusion") else ([] if key == "evidence" else 0.5)) for key in expected_keys}

    async def _call_llm_branches(
        self, prompt: str, count: int
    ) -> list[dict[str, Any]]:
        """Call the LLM and parse a JSON array of branch seeds.

        Args:
            prompt: The prompt requesting multiple branches.
            count: Expected number of branches.

        Returns:
            List of branch seed dicts.
        """
        try:
            response = await self._llm.complete(
                prompt=prompt,
                model=self._model_name,
                response_format={"type": "json_object"},
            )
            import json
            raw = response.text if hasattr(response, "text") else str(response)
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed[:count]
            if isinstance(parsed, dict) and "branches" in parsed:
                return parsed["branches"][:count]
            return [parsed]
        except Exception as exc:
            logger.error("LLM branch call failed", error=str(exc))
            return [{"description": f"Branch {i}", "evidence": [], "conclusion": "", "confidence": 0.5} for i in range(count)]
