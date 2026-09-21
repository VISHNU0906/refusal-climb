import numpy as np
import pytest
from refusalclimb.study import run_study


def data(prefix, seed):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], 40)
    activations = rng.normal(size=(80, 12))
    activations[:, 0] += (labels * 2 - 1) * 2
    return {"activations": activations, "labels": labels,
            "ids": np.array([f"{prefix}-{i}" for i in range(80)]),
            "categories": np.array([f"category-{i // 20}" for i in range(80)])}


def test_reproducible_baselines():
    a, b = data("train", 1), data("test", 2)
    first = run_study(a, b, repeats=30, seed=4)
    assert first == run_study(a, b, repeats=30, seed=4)
    assert first["balanced_accuracy"] > first["baselines"]["shuffled_training_labels_mean"]
    assert first["baselines"]["training_majority_balanced_accuracy"] == 0.5


def test_reversed_test_labels_do_not_change_scores():
    a, b = data("train", 1), data("test", 2)
    first = run_study(a, b, repeats=20)
    second = run_study(a, {**b, "labels": 1 - b["labels"]}, repeats=20)
    assert first["scores"] == second["scores"]
    assert first["threshold"] == second["threshold"]
    assert first["balanced_accuracy"] + second["balanced_accuracy"] == pytest.approx(1)


def test_requires_two_test_classes():
    a, b = data("train", 1), data("test", 2)
    with pytest.raises(ValueError, match="test requires"):
        run_study(a, {**b, "labels": np.zeros(80)}, repeats=20)
