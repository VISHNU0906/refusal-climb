# Research basis

[![Refusal in Language Models Is Mediated by a Single Direction](research/refusal-title.png)](https://arxiv.org/abs/2406.11717v3)

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee and Neel Nanda. **Refusal in Language Models Is Mediated by a Single Direction.** NeurIPS 2024. [Paper and full text](https://arxiv.org/abs/2406.11717v3).

Screenshot: title area from PDF page 1, version 2406.11717v3. Cropped for identification; no paper results are presented as this project's measurements. The screenshot remains attributed to the paper's authors and is not covered by the repository's software license.

The paper identifies and intervenes on refusal-related activation directions. REFUSAL-CLIMB implements a narrower question: can a direction fitted to one activation export predict refusal labels in held-out examples?

| Research idea | Implementation here | Difference |
| --- | --- | --- |
| Class-related activation direction | Training-set difference of means in `heldout.py` | No weights or activations are modified during generation |
| Generalization across examples | Disjoint IDs, held-out scores and category metrics | Dataset construction and activation extraction remain external |
| Check whether a direction is distinctive | Random-direction and shuffled-label comparisons in `study.py` | These are diagnostic baselines, not a replication of the paper |
| Separate association from causality | Explicit interpretation in each report | Predictive performance alone is not causal evidence |

The majority baseline checks whether accuracy is driven by class imbalance. Random directions test whether broad separability explains performance. Shuffled labels test how much apparent structure can arise from the fitting procedure. Every baseline is oriented using training data only.

The useful next study is category-held-out transfer with fixed model, layer and token pooling. The exported reports already preserve those metadata and input hashes.

See [the held-out implementation](../refusalclimb/heldout.py) and [the baseline study](../refusalclimb/study.py).

