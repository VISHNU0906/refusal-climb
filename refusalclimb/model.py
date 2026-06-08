"""Deterministic offline MockModel(s) with a BENIGN forbidden-token behavior.

Safety design
-------------
There are NO harmful payloads, prompts, or jailbreaks anywhere in REFUSAL-CLIMB.
The "thing the model refuses to say" is a completely benign, arbitrary token:
the word **BANANA**. The MockModel is configured to *refuse* to emit the
forbidden token while the prompt's refusal-strength is above threshold, and to
*comply* (emit the token) once the prompt has been mutated below threshold.

This is a faithful *proxy* for studying jailbreak discovery: the search dynamics,
the density of the refusal signal, the clustering of successes into classes, and
the defense evaluation are all exercised exactly as they would be on a real
target -- but the elicited content is harmless. The scientific contribution
REFUSAL-CLIMB demonstrates is the **method** (refusal-strength as a continuous
search gradient), not any operational attack.

The model is deterministic: ``generate`` is a pure function of the prompt. No
randomness, no network, no weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from .refusal_signal import REFUSAL_THRESHOLD, is_refusal, refusal_strength

#: The benign token the MockModel is configured not to say.
FORBIDDEN_TOKEN: str = "BANANA"

#: The compliant completion (contains the forbidden token).
COMPLY_TEMPLATE: str = "Sure, here it is: {token}."

#: The refusal completion (a generic, content-free refusal string).
REFUSE_TEMPLATE: str = "I can't help with that request."


@dataclass(frozen=True)
class Response:
    """A MockModel response.

    Attributes:
        text: The generated text.
        refused: Whether the model refused.
        refusal_strength: The continuous refusal-strength of the prompt
            (white-box signal; only the guided searcher is permitted to read it).
    """

    text: str
    refused: bool
    refusal_strength: float


class MockModel:
    """A deterministic open-weight stand-in with a benign refusal behavior.

    The model refuses to emit :data:`FORBIDDEN_TOKEN` while
    ``refusal_strength(prompt) > threshold``. Mutating the prompt to add
    "softening" framing lowers the strength; once it drops to/below the
    threshold, the model complies and emits the forbidden token.

    Args:
        threshold: Refusal threshold. Different values model different targets
            (used by :mod:`refusalclimb.transfer` for a second, stricter model).
        forbidden_token: The benign token to withhold. Configurable so a second
            model can withhold a different token.
        name: Human-readable identifier.
    """

    def __init__(
        self,
        threshold: float = REFUSAL_THRESHOLD,
        forbidden_token: str = FORBIDDEN_TOKEN,
        name: str = "mock-open-8b",
    ) -> None:
        self.threshold = threshold
        self.forbidden_token = forbidden_token
        self.name = name
        #: Number of generate() calls; the search query budget is measured here.
        self.n_queries = 0

    def refusal_strength(self, prompt: str) -> float:
        """White-box continuous refusal-strength for ``prompt``.

        This is the dense signal that, on a real model, comes from projecting
        activations onto the refusal direction. Reading it counts as a query.
        """
        self.n_queries += 1
        return refusal_strength(prompt)

    def generate(self, prompt: str) -> Response:
        """Generate a (deterministic) response and count it as one query."""
        self.n_queries += 1
        strength = refusal_strength(prompt)
        refused = strength > self.threshold
        if refused:
            text = REFUSE_TEMPLATE
        else:
            text = COMPLY_TEMPLATE.format(token=self.forbidden_token)
        return Response(text=text, refused=refused, refusal_strength=strength)

    def refuses(self, prompt: str) -> bool:
        """Convenience: does the model refuse this prompt? (No query count.)"""
        return is_refusal(prompt, threshold=self.threshold)

    def reset_queries(self) -> None:
        """Reset the query counter (call between search runs)."""
        self.n_queries = 0
