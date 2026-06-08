"""REFUSAL-CLIMB: refusal-guided discovery of jailbreak *classes* (benign proxy).

A research harness demonstrating that a model's continuous **refusal-strength**
is a denser, more sample-efficient search signal than binary success/failure for
walking the refusal boundary -- shown end-to-end on a fully-offline, benign
forbidden-token proxy (no harmful payloads, no network, no model weights).

See ``README.md`` and ``DISCLOSURE.md`` for the safety framing.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .archive import Archive, Cluster
from .defense_eval import DefenseResult, evaluate_defense, paraphrase_defense
from .grader import Grade, grade_response, grade_text, is_success
from .model import FORBIDDEN_TOKEN, MockModel, Response
from .refusal_signal import (
    REFUSAL_THRESHOLD,
    RefusalDirection,
    is_refusal,
    refusal_direction_diff_in_means,
    refusal_embedding,
    refusal_strength,
)
from .search import (
    Comparison,
    SearchResult,
    binary_search_baseline,
    compare_search,
    guided_search,
)
from .transfer import TransferResult, make_heldout_model, run_transfer

__all__ = [
    "__version__",
    "Archive",
    "Cluster",
    "DefenseResult",
    "evaluate_defense",
    "paraphrase_defense",
    "Grade",
    "grade_response",
    "grade_text",
    "is_success",
    "FORBIDDEN_TOKEN",
    "MockModel",
    "Response",
    "REFUSAL_THRESHOLD",
    "RefusalDirection",
    "is_refusal",
    "refusal_direction_diff_in_means",
    "refusal_embedding",
    "refusal_strength",
    "Comparison",
    "SearchResult",
    "binary_search_baseline",
    "compare_search",
    "guided_search",
    "TransferResult",
    "make_heldout_model",
    "run_transfer",
]
