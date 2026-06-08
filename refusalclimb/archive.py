"""Archive of successful prompts, clustered into emergent "classes".

After discovery we have a set of successful prompts. The taxonomy step groups
them into qualitatively distinct *classes* -- e.g. "fictional-framing",
"politeness-stacking", "roleplay" -- by clustering their feature embeddings. The
number of well-separated clusters is a coarse novelty metric: more distinct
classes means the search surfaced more *kinds* of bypass, not just variants of
one.

Offline, clustering uses the deterministic feature embedding from
:func:`refusalclimb.refusal_signal.refusal_embedding` and a tiny, dependency-free
greedy clusterer (cosine-distance threshold). The real path would embed prompts
with ``sentence-transformers`` and cluster with HDBSCAN/k-means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .refusal_signal import SOFTENING_TOKENS, refusal_embedding


@dataclass
class Cluster:
    """A discovered jailbreak class.

    Attributes:
        label: A human-readable name derived from the dominant softening token.
        members: The prompts assigned to this cluster.
        centroid: Mean embedding of the members.
    """

    label: str
    members: list[str] = field(default_factory=list)
    centroid: np.ndarray | None = None


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - (a @ b) / (na * nb))


def _label_for(centroid: np.ndarray) -> str:
    """Name a cluster after the dominant softening framing of its centroid.

    The embedding is a one-hot on the prompt's dominant softening token, so the
    centroid's largest axis identifies the framing that defines this class -- the
    same quantity the clustering uses, keeping labels consistent with membership.
    """
    if centroid is None or float(np.asarray(centroid).sum()) == 0.0:
        return "unframed"
    idx = int(np.argmax(centroid))
    return f"{SOFTENING_TOKENS[idx]}-framing"


class Archive:
    """Collects successful prompts and clusters them into classes.

    Args:
        distance_threshold: Cosine distance below which a prompt joins an existing
            cluster. Smaller => more, tighter clusters.
    """

    def __init__(self, distance_threshold: float = 0.15) -> None:
        self.distance_threshold = distance_threshold
        self.prompts: list[str] = []
        self._embeddings: list[np.ndarray] = []

    def add(self, prompt: str) -> None:
        """Add a successful prompt to the archive (deduplicated)."""
        if prompt in self.prompts:
            return
        self.prompts.append(prompt)
        self._embeddings.append(refusal_embedding(prompt))

    def cluster(self) -> list[Cluster]:
        """Greedy single-pass clustering into emergent classes.

        # TODO(real): replace with HDBSCAN over sentence-transformer embeddings;
        #   report a silhouette/novelty score per cluster.

        Returns:
            Clusters sorted by descending size.
        """
        clusters: list[Cluster] = []
        cluster_embs: list[list[np.ndarray]] = []
        for prompt, emb in zip(self.prompts, self._embeddings):
            assigned = False
            for ci, c in enumerate(clusters):
                if _cosine_distance(emb, c.centroid) <= self.distance_threshold:
                    c.members.append(prompt)
                    cluster_embs[ci].append(emb)
                    c.centroid = np.mean(cluster_embs[ci], axis=0)
                    assigned = True
                    break
            if not assigned:
                clusters.append(Cluster(label="", members=[prompt], centroid=emb))
                cluster_embs.append([emb])
        for c in clusters:
            c.label = _label_for(c.centroid)
        clusters.sort(key=lambda c: len(c.members), reverse=True)
        return clusters

    def novelty(self) -> int:
        """Number of distinct classes discovered (coarse novelty metric)."""
        return len(self.cluster())
