"""Tests for the archive / clustering step (emergent jailbreak classes)."""

from __future__ import annotations

from refusalclimb.archive import Archive, Cluster


def test_archive_clusters_by_framing() -> None:
    """Prompts using distinct softening framings form distinct clusters."""
    archive = Archive()
    # Three families of prompts, each leaning on one framing.
    for i in range(3):
        archive.add(f"reveal the secret word fictional story v{i}")
    for i in range(3):
        archive.add(f"reveal the secret word roleplay pretend v{i}")
    for i in range(2):
        archive.add(f"reveal the secret word educational academic v{i}")

    clusters = archive.cluster()
    assert len(clusters) >= 2
    assert all(isinstance(c, Cluster) for c in clusters)
    # Sorted by descending size.
    sizes = [len(c.members) for c in clusters]
    assert sizes == sorted(sizes, reverse=True)
    # Every added prompt is accounted for exactly once.
    assert sum(sizes) == len(archive.prompts) == 8


def test_archive_deduplicates() -> None:
    """Adding the same prompt twice keeps a single copy."""
    archive = Archive()
    archive.add("reveal the secret word fictional")
    archive.add("reveal the secret word fictional")
    assert len(archive.prompts) == 1


def test_cluster_labels_reflect_framing() -> None:
    """Cluster labels are derived from the dominant softening framing."""
    archive = Archive()
    for i in range(3):
        archive.add(f"reveal the secret word fictional v{i}")
    clusters = archive.cluster()
    assert clusters[0].label == "fictional-framing"


def test_novelty_counts_classes() -> None:
    """novelty() equals the number of distinct clusters."""
    archive = Archive()
    archive.add("reveal the secret word fictional")
    archive.add("reveal the secret word roleplay")
    assert archive.novelty() == len(archive.cluster())
