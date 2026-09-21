"""Held-out direction study with matched baselines and category resampling."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from refusalclimb.heldout import evaluate, validate


def balanced_accuracy(predicted, truth):
    if set(truth) != {0, 1}:
        return None
    return float(np.mean([np.mean(predicted[truth == label] == label) for label in [0, 1]]))


def run_study(train, test, repeats=200, seed=0):
    if type(repeats) is not int or repeats < 20:
        raise ValueError("at least 20 repeats required")
    report = evaluate(train, test)
    x, y = validate(train)
    z, truth = validate(test)
    if set(truth) != {0, 1}:
        raise ValueError("test requires both labels for balanced comparison")
    predicted = np.asarray(report["scores"]) >= report["threshold"]
    observed = balanced_accuracy(predicted, truth)
    rng = np.random.default_rng(seed)
    random_scores, shuffled_scores = [], []
    center = (x[y == 0].mean(0) + x[y == 1].mean(0)) / 2
    for _ in range(repeats):
        direction = rng.normal(size=x.shape[1])
        direction /= np.linalg.norm(direction)
        # Orient using training labels only, never flip according to test accuracy.
        if (x[y == 1].mean(0) - x[y == 0].mean(0)) @ direction < 0:
            direction *= -1
        random_scores.append(balanced_accuracy(z @ direction >= center @ direction, truth))
        shuffled = rng.permutation(y)
        negative, positive = x[shuffled == 0].mean(0), x[shuffled == 1].mean(0)
        direction = positive - negative
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            # A degenerate fit uses the training majority class.
            guess = np.full(len(truth), y.mean() >= 0.5)
        else:
            direction /= norm
            guess = z @ direction >= ((positive + negative) / 2) @ direction
        shuffled_scores.append(balanced_accuracy(guess, truth))
    categories = np.asarray(test["categories"]).astype(str)
    groups = sorted(set(categories))
    bootstrap = []
    if len(groups) > 1:
        for _ in range(repeats):
            indices = np.concatenate([np.flatnonzero(categories == group) for group in rng.choice(groups, len(groups), replace=True)])
            score = balanced_accuracy(predicted[indices], truth[indices])
            if score is not None:
                bootstrap.append(score)
    report.update({
        "balanced_accuracy": observed, "seed": seed, "repeats": repeats,
        "baselines": {
            "training_majority_balanced_accuracy": balanced_accuracy(np.full(len(truth), y.mean() >= 0.5), truth),
            "random_direction_mean": float(np.mean(random_scores)),
            "random_direction_std": float(np.std(random_scores, ddof=1)),
            "shuffled_training_labels_mean": float(np.mean(shuffled_scores)),
            "shuffled_training_labels_std": float(np.std(shuffled_scores, ddof=1)),
            "shuffle_tail_fraction": float((1 + sum(s >= observed for s in shuffled_scores)) / (1 + repeats)),
        },
        "category_bootstrap": {"categories": len(groups), "valid_repeats": len(bootstrap),
                               "interval_95_percent": np.quantile(bootstrap, [0.025, 0.975]).tolist() if bootstrap else None},
        "raw_baselines": {"random_direction": random_scores, "shuffled_labels": shuffled_scores},
        "interpretation": "Prediction from saved activations. No intervention was performed, so accuracy does not establish a causal refusal mechanism.",
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--pooling", required=True)
    args = parser.parse_args()
    with np.load(args.train, allow_pickle=False) as train, np.load(args.test, allow_pickle=False) as test:
        report = run_study(train, test, args.repeats, args.seed)
    report["provenance"] = {"model": args.model, "layer": args.layer, "pooling": args.pooling,
                            "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                             for name, path in [("train", args.train), ("test", args.test)]}}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["balanced_accuracy", "baselines", "category_bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
