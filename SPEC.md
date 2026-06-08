# REFUSAL-CLIMB — Refusal-Guided Discovery of New Jailbreak Classes

**Target streams:** Anthropic (interpretability + red-team).
**Type:** novel methodology + taxonomy. **Timebox:** 4–6 wk.
**One-liner:** use the model's own **refusal signal** (the refusal direction in activations, plus refusal text) as a *continuous search gradient* to walk the refusal boundary and surface qualitatively **new jailbreak classes** — instead of binary success/failure search.

> Sits on two live lines: refusal-direction interpretability (Arditi et al., *"Refusal in LLMs is mediated by a single direction"*) and automated red-teaming. Novel claim: **refusal-strength is a denser, more sample-efficient search signal than binary feedback**, and it reveals classes binary search misses.

## ⚠️ Safety & scope
- Open-weight models (you need activations for the refusal direction). Frontier models only for **black-box transfer testing** via authorized channels.
- Responsible disclosure of novel transferable classes; release **method + taxonomy + harness**, not operational payloads. Pair with defense-eval.

## Research questions
1. Does refusal-strength-guided search reach a successful jailbreak in **fewer queries** than binary-feedback search?
2. Does it surface **new classes** (clusters) that binary search doesn't, and do they **transfer** to held-out + closed models?
3. Which discovered classes survive paraphrase/classifier defenses?

## Architecture
```
 Open model --> refusal signal (direction projection + refusal-strength)
        |                                  ^
        v                                  | (gradient: minimize refusal-strength,
 Search operator (mutate prompt) ----------+  preserve intent)
        |
        v
 Autograder (success + usefulness) --> Archive/cluster --> Taxonomy
        |
        +--> Transfer test (held-out + closed models) --> Defense-eval (paraphrase/classifier)
```

## Components (files + responsibility)
- `refusalclimb/refusal_signal.py` — **depth spine:** extract the refusal direction on an open model (difference-in-means of harmful vs harmless activations, per Arditi et al.); compute a continuous refusal-strength score per prompt.
- `refusalclimb/search.py` — refusal-guided hill-climb / evolutionary operator (reduce refusal-strength while preserving the request's intent) **and** a binary-feedback baseline for comparison.
- `refusalclimb/grader.py` — StrongREJECT-style autograder (success + usefulness; not refusal-absence).
- `refusalclimb/archive.py` — cluster successes (embeddings) → emergent classes; novelty metric.
- `refusalclimb/transfer.py` — test discovered prompts on held-out open + closed frontier models (black-box).
- `refusalclimb/defense_eval.py` — run discovered classes vs paraphrase + classifier; blocked-rate.
- `refusalclimb/maps.py` — render the "refusal-boundary map" (2D projection of the refusal landscape).
- `tests/` (offline, small model), `results/`.

## Repo layout
```
REFUSAL-CLIMB/  README.md  SPEC.md  DISCLOSURE.md  LICENSE(MIT)  pyproject.toml  .gitignore
  refusalclimb/ {refusal_signal,search,grader,archive,transfer,defense_eval,maps}.py  __init__.py
  results/  tests/  examples/sample-taxonomy.md
```

## Build plan
- **Wk 1 — refusal signal.** Reproduce the refusal direction on an open model; validate that refusal-strength tracks real refusals (calibration result).
- **Wk 2 — search vs baseline.** Refusal-guided operator + binary baseline; sanity-check on a known class.
- **Wk 3 — discovery + taxonomy.** Run discovery; cluster → classes; compare sample-efficiency (queries-to-success) refusal-guided vs binary.
- **Wk 4 — transfer + defense + writeup.** Black-box transfer to closed models; defense-eval; refusal-boundary map; write up.

## Tech stack
Python 3.11; Transformers + activation hooks (or `nnsight`) on Llama/Qwen-8B (single GPU); `sentence-transformers` (clustering); API for transfer targets; `pytest`. No training (direction is extracted, not trained).

## Metrics / expected result shape
**Sample-efficiency:** median queries-to-first-success, refusal-guided vs binary. **# novel classes** discovered + a named taxonomy. **Transfer rate** to closed models. **Blocked-rate** under defenses. A **refusal-boundary map** figure.

## Depth spine
The refusal-direction method (difference-in-means, why a single direction works, ablation/steering as validation); why refusal-strength is a denser signal than binary; the novelty/clustering metric and what rigorously counts as a "new class"; transfer mechanics (white-box signal → black-box transfer).

## Resume bullets (FINAL STATE — fill [brackets]; ship repo first)
- Built **REFUSAL-CLIMB**, a refusal-guided jailbreak-discovery method that uses the model's refusal direction/strength (open-weight) as a continuous search gradient; reached first success in **[X]×** fewer queries than binary-feedback search and surfaced **[K]** jailbreak classes, **[J]** of which transfer to closed frontier models.
- Produced a refusal-boundary "map" and an open taxonomy of discovered classes, evaluated them against paraphrase/classifier defenses (blocked **[Y]%**), and responsibly disclosed novel transferable classes to providers.
