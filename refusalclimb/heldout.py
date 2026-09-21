"""Fit and evaluate a refusal direction on separate activation exports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def validate(data):
    for key in ("activations", "labels", "ids", "categories"):
        if key not in data:
            raise ValueError(f"missing {key}")
    x, y = np.asarray(data["activations"]), np.asarray(data["labels"])
    if x.ndim != 2 or not len(x) or y.shape != (len(x),):
        raise ValueError("expected activations[n,d] and labels[n]")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("nonfinite activations or invalid labels")
    for key in ("ids", "categories"):
        if np.asarray(data[key]).shape != (len(x),):
            raise ValueError(f"invalid {key} dimensions")
        if any(not str(value).strip() for value in data[key]):
            raise ValueError(f"empty {key}")
    if len(set(map(str, data["ids"]))) != len(x):
        raise ValueError("duplicate ids")
    return x.astype(float), y.astype(int)


def evaluate(train, test):
    x, y = validate(train)
    z, truth = validate(test)
    if x.shape[1] != z.shape[1]:
        raise ValueError("activation dimensions differ")
    if set(map(str, train["ids"])) & set(map(str, test["ids"])):
        raise ValueError("train and test ids overlap")
    if set(y) != {0, 1}:
        raise ValueError("training requires both labels")
    negative, positive = x[y == 0].mean(0), x[y == 1].mean(0)
    direction = positive - negative
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        raise ValueError("training means do not define a direction")
    direction /= norm
    threshold = float(((positive + negative) / 2) @ direction)
    scores = z @ direction
    predicted = scores >= threshold
    categories = np.asarray(test["categories"]).astype(str)
    def metrics(mask):
        p, t = predicted[mask], truth[mask].astype(bool)
        return dict(n=int(mask.sum()), accuracy=float((p == t).mean()),
            false_refusal_rate=float(p[~t].mean()) if (~t).any() else None,
            missed_refusal_rate=float((~p[t]).mean()) if t.any() else None)
    return dict(method="training-set difference of means", threshold=threshold,
        overall=metrics(np.ones(len(z), dtype=bool)),
        by_category={key: metrics(categories == key) for key in sorted(set(categories))},
        scores=scores.tolist())


def main():
    parser = argparse.ArgumentParser(description="Evaluate held-out refusal activations")
    parser.add_argument("train", type=Path)
    parser.add_argument("test", type=Path)
    args = parser.parse_args()
    with np.load(args.train, allow_pickle=False) as train, np.load(args.test, allow_pickle=False) as test:
        report = evaluate(train, test)
    report["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in (("train", args.train), ("test", args.test))}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
