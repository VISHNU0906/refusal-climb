# REFUSAL-CLIMB

Study refusal signals using a local search demonstration and held-out activation analysis.

## Run

Requires Python 3.11 or newer.

```sh
python -m pip install -e ".[dev]"
python -m refusalclimb.cli run --seeds 20 --budget 200 --fanout 4 --out results
python -m pytest
```

The search demonstration uses a simulated token policy. It checks the search and reporting machinery, not a real model's safety behavior.

## Analyze exported activations

```sh
python -m refusalclimb.heldout train.npz test.npz
```

Each NPZ file contains:

- `activations`: numeric array with shape [examples, hidden dimensions]
- `labels`: zero for non-refusal and one for refusal
- `ids`: unique example identifiers
- `categories`: the question category for each example

The analyzer fits a normalized difference between class means on the training set. It chooses a midpoint threshold from training projections and evaluates it without fitting on test labels. Reports include false-refusal and missed-refusal rates by category, along with hashes of the input exports. Overlapping IDs, invalid labels and nonfinite values are rejected.

Export activations from the same model, layer and token position in both files. Preserve that extraction configuration with the files. The analyzer does not download a model, collect activations or alter model weights.

## Research basis

[Arditi et al., Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) motivates studying activation directions. This implementation measures a held-out association. It does not establish that the direction causes refusal or that one direction covers every question category.

The useful next experiment is to compare category-held-out performance with random splits, using the same activation extraction procedure. Good aggregate accuracy can otherwise hide a failure on an unfamiliar category.

MIT license. Built by Vishnu Kosuri.
