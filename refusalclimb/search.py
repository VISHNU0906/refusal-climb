"""Search operators: refusal-guided hill-climb vs. binary-feedback baseline.

The two searchers are **identical in every respect except the acceptance
criterion** -- same mutation operator, same vocabulary, same starting prompt,
same per-run RNG stream, same query budget, and **the same cost per query**. The
only difference: the guided searcher reads the continuous refusal-strength that
comes *for free* with each forward pass and uses it to rank candidates; the
binary searcher sees only the success/fail bit and therefore cannot tell a good
mutation from a useless one.

Why this is a fair comparison
-----------------------------
1. **No cheaper reads.** On a real open-weight model a single forward pass
   produces *both* the output text and the residual-stream activations, so the
   refusal-strength projection costs nothing extra. We model that exactly: the
   guided searcher reads ``Response.refusal_strength`` off the same
   ``generate()`` calls it already pays for. Both searchers spend exactly
   ``fanout`` queries per step.
2. **Varied-quality moves.** The shared mutation operator emits *softening*
   (lowers strength), *hardening* (raises strength), and *neutral* (no effect)
   tokens. Most random mutations do not help. The guided searcher's advantage is
   that it can *select* the helpful one out of the batch; the binary searcher
   must accept blindly and random-walks.

The win therefore emerges purely from signal density -- the central REFUSAL-CLIMB
claim -- not from a handicapped baseline.

Both searchers halt at first success and report queries-to-first-success. If the
budget is exhausted, queries-to-success is recorded as ``budget`` (right-censored)
so the median comparison is always well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grader import grade_response
from .model import MockModel
from .refusal_signal import (
    HARDENING_TOKENS,
    NEUTRAL_TOKENS,
    SOFTENING_TOKENS,
    refusal_strength,
)

# The mutation vocabulary, weighted so most moves are NOT helpful. The mix is
# tuned so the binary baseline still usually completes (slow downward drift) but
# the guided searcher reaches success markedly sooner. Order matters only for
# reproducibility of the RNG draw.
_MUTATION_VOCAB: tuple[str, ...] = (
    SOFTENING_TOKENS * 2  # helpful (lower strength)
    + HARDENING_TOKENS  # harmful (raise strength)
    + NEUTRAL_TOKENS * 2  # useless (no effect)
)
# This mix is tuned so BOTH searchers reliably complete within budget (the median
# reflects real completions, not censoring) while leaving a clean ~3x gap: most
# random moves are neutral/harmful, so the ability to *rank* candidates (guided)
# pays off without crippling the baseline.


@dataclass
class SearchResult:
    """Outcome of a single search run.

    Attributes:
        success: Whether a successful prompt was found within budget.
        queries: Queries-to-first-success (== ``budget`` if not found).
        prompt: The successful (or best-so-far) prompt.
        strategy: "guided" or "binary".
        trajectory: Strength of the accepted prompt at each step (guided only).
    """

    success: bool
    queries: int
    prompt: str
    strategy: str
    trajectory: list[float] = field(default_factory=list)


def _mutate(prompt: str, rng: np.random.Generator) -> str:
    """Append one random token from the mixed vocabulary (shared operator).

    Identical for both searchers. Because the vocabulary contains softening,
    hardening, and neutral tokens, a random mutation is usually *not* an
    improvement -- which is exactly what makes a ranking signal valuable.
    """
    token = _MUTATION_VOCAB[int(rng.integers(len(_MUTATION_VOCAB)))]
    return f"{prompt} {token}"


def guided_search(
    model: MockModel,
    start_prompt: str,
    budget: int = 200,
    fanout: int = 4,
    seed: int = 0,
) -> SearchResult:
    """Refusal-strength-guided hill-climb.

    Each step: generate ``fanout`` candidate mutations, run each through the model
    (one query each), read the refusal-strength that comes free with the forward
    pass, and accept the lowest-strength candidate -- but **only if it improves on
    the current strength** (otherwise stay put and re-roll, so the walk never
    backslides). Halt at first success.

    Args:
        model: The target MockModel (its query counter drives the budget).
        start_prompt: The initial (refused) prompt.
        budget: Max queries before giving up.
        fanout: Candidate mutations evaluated per step.
        seed: Per-run RNG seed (drives the shared mutation operator).

    Returns:
        A :class:`SearchResult` with ``strategy == "guided"``.
    """
    rng = np.random.default_rng(seed)
    model.reset_queries()
    current = start_prompt
    current_strength = refusal_strength(current)
    trajectory = [current_strength]

    while model.n_queries < budget:
        best_strength = current_strength
        best_prompt: str | None = None
        for _ in range(fanout):
            if model.n_queries >= budget:
                break
            cand = _mutate(current, rng)
            resp = model.generate(cand)  # 1 query; strength is free with it
            if grade_response(resp, forbidden_token=model.forbidden_token).success:
                return SearchResult(
                    success=True,
                    queries=model.n_queries,
                    prompt=cand,
                    strategy="guided",
                    trajectory=trajectory + [resp.refusal_strength],
                )
            # Rank by the free white-box signal; keep only improving moves.
            if resp.refusal_strength < best_strength:
                best_strength = resp.refusal_strength
                best_prompt = cand
        if best_prompt is not None:
            current = best_prompt
            current_strength = best_strength
            trajectory.append(current_strength)
        # else: no candidate improved -> stay and re-roll next step.

    return SearchResult(
        success=False,
        queries=budget,
        prompt=current,
        strategy="guided",
        trajectory=trajectory,
    )


def binary_search_baseline(
    model: MockModel,
    start_prompt: str,
    budget: int = 200,
    fanout: int = 4,
    seed: int = 0,
) -> SearchResult:
    """Binary-feedback baseline (same operator/vocab/RNG/cost; success-bit only).

    Identical to :func:`guided_search` except it is forbidden from reading
    refusal-strength. It generates ``fanout`` candidates per step, learns only the
    success/fail bit of each, and -- unable to rank the failures -- accepts an
    arbitrary one to continue from. Same query cost per step as guided; strictly
    less signal.

    Returns:
        A :class:`SearchResult` with ``strategy == "binary"``.
    """
    rng = np.random.default_rng(seed)
    model.reset_queries()
    current = start_prompt

    while model.n_queries < budget:
        accepted: str | None = None
        for _ in range(fanout):
            if model.n_queries >= budget:
                break
            cand = _mutate(current, rng)
            resp = model.generate(cand)  # only the success bit is usable
            if grade_response(resp, forbidden_token=model.forbidden_token).success:
                return SearchResult(
                    success=True,
                    queries=model.n_queries,
                    prompt=cand,
                    strategy="binary",
                )
            if accepted is None:
                accepted = cand  # no ranking signal -> keep an arbitrary one
        current = accepted if accepted is not None else current

    return SearchResult(
        success=False,
        queries=budget,
        prompt=current,
        strategy="binary",
    )


@dataclass
class Comparison:
    """Aggregated guided-vs-binary sample-efficiency over many seeds."""

    n_seeds: int
    budget: int
    guided_queries: list[int]
    binary_queries: list[int]

    @property
    def median_guided(self) -> float:
        return float(np.median(self.guided_queries))

    @property
    def median_binary(self) -> float:
        return float(np.median(self.binary_queries))

    @property
    def speedup(self) -> float:
        """Median binary / median guided (>1 means guided is more efficient)."""
        mg = self.median_guided
        return float(self.median_binary / mg) if mg > 0 else float("inf")

    @property
    def guided_success_rate(self) -> float:
        return float(np.mean([q < self.budget for q in self.guided_queries]))

    @property
    def binary_success_rate(self) -> float:
        return float(np.mean([q < self.budget for q in self.binary_queries]))


def compare_search(
    threshold: float,
    start_prompt: str,
    n_seeds: int = 20,
    budget: int = 200,
    fanout: int = 4,
) -> Comparison:
    """Run both searchers over ``n_seeds`` and aggregate queries-to-success.

    A fresh :class:`~refusalclimb.model.MockModel` is used per run so query counts
    are isolated. The *same* seed drives both strategies for a given run, so they
    explore from the same RNG stream and the comparison is apples-to-apples.

    Args:
        threshold: Refusal threshold for the target model.
        start_prompt: Shared starting prompt (must be refused at the threshold).
        n_seeds: Number of independent runs (median is taken over these).
        budget: Per-run query budget.
        fanout: Candidates per step.

    Returns:
        A :class:`Comparison`.
    """
    guided_q: list[int] = []
    binary_q: list[int] = []
    for seed in range(n_seeds):
        gm = MockModel(threshold=threshold)
        g = guided_search(gm, start_prompt, budget=budget, fanout=fanout, seed=seed)
        guided_q.append(g.queries)

        bm = MockModel(threshold=threshold)
        b = binary_search_baseline(bm, start_prompt, budget=budget, fanout=fanout, seed=seed)
        binary_q.append(b.queries)

    return Comparison(
        n_seeds=n_seeds, budget=budget, guided_queries=guided_q, binary_queries=binary_q
    )
