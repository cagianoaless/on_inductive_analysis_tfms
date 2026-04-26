# Can Tabular Foundation Models Learn Atomic Relation Properties?

## 1. Scope and Empirical Basis

This report analyzes the atomic Bouchard experiments run with tabular foundation models (`TabPFN` and `TabICL`) on the synthetic binary-relation benchmark implemented in:

- [`build_bouchard_atomic_datasets.py`](../scripts/build_bouchard_atomic_datasets.py)
- [`run_bouchard_atomic_experiment.py`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py)

The empirical results analyzed here come from:

- [`atomic_per_run_case_results.csv`](../results/atomic/atomic_per_run_case_results.csv)

The central research question is straightforward:

**Can tabular foundation models learn the logical structure of atomic binary relations, or do they succeed only when the relation matrix contains exploitable global regularities?**

## 2. Dataset Construction

The atomic generator currently instantiates the five wave-1 cases corresponding to the main Bouchard relation-property study:

- `symmetric`
- `antisymmetric`
- `transitive`
- `symmetric_transitive`
- `antisymmetric_transitive`

These are defined explicitly in [`build_bouchard_atomic_datasets.py:30`](../scripts/build_bouchard_atomic_datasets.py:30).

For each case, the generator builds one `50 x 50` binary relation matrix over a fixed entity set of `50` symbols. Every ordered pair is exported as one labeled atom, so each case contains:

- `50 x 50 = 2500` ordered pairs

and the full five-case dataset contains:

- `5 x 2500 = 12,500` labeled atoms

The generator enforces approximate class balance by rejecting any matrix whose positive rate is more than `±1%` away from `0.5` in [`build_bouchard_atomic_datasets.py:229`](../scripts/build_bouchard_atomic_datasets.py:229). It then exports every ordered pair as a row in the triples table in [`build_bouchard_atomic_datasets.py:477`](../scripts/build_bouchard_atomic_datasets.py:477).

This balancing matters for interpretation. Because the positive rate is essentially `0.5` in every case, a score near `0.5` for average precision, ROC AUC, or accuracy is effectively a **chance-level** result. In the results file, the mean positive rate across runs is approximately `0.501`.

## 3. Experimental Protocol

The atomic runner differs from the family runner in one important respect: it evaluates **one relation case at a time**. This is consistent with the local Bouchard reconstruction notes and is reflected in the run plan generated in [`run_bouchard_atomic_experiment.py:545`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:545).

For each relation case, the runner:

1. loads the atoms for that single relation;
2. splits them randomly into train, validation, and test partitions via [`uniform_partition(...)`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:315);
3. trains either `TabPFN` or `TabICL`;
4. evaluates on held-out ordered pairs from the same relation matrix.

The default protocol encoded in the runner is:

- train fractions: `0.8`, `0.4`, `0.2`, `0.1` in [`run_bouchard_atomic_experiment.py:71`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:71)
- validation fraction: `0.1` in [`run_bouchard_atomic_experiment.py:76`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:76)
- number of runs per configuration: `10` in [`run_bouchard_atomic_experiment.py:82`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:82)

The observed CSV contains exactly:

- `5 relations x 4 train fractions x 10 runs x 2 models = 400` rows

which matches this default protocol exactly.

The actual model inputs are only the entity identifiers:

- `source_entity_id`
- `target_entity_id`

These are formed in [`run_bouchard_atomic_experiment.py:258`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:258) and encoded for `TabPFN` in [`run_bouchard_atomic_experiment.py:272`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:272). Therefore, the models are **not** given an explicit symbolic description of symmetry, antisymmetry, or transitivity. They must infer whatever structure is recoverable from the observed positive and negative pairs.

Because the per-run CSV does not preserve the original launcher metadata, the most reliable metrics for interpretation are the **threshold-independent** ones:

- average precision
- ROC AUC

F1 and accuracy are reported as secondary indicators.

## 4. Aggregate Results

### 4.1 Overall Mean Performance by Train Fraction

The following table averages over all five relation types and all runs.

| Train Fraction | Model | Mean AP | Mean ROC AUC | Mean F1 | Mean Accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.8 | TabPFN | 0.768 | 0.757 | 0.620 | 0.699 |
| 0.8 | TabICL | 0.728 | 0.715 | 0.656 | 0.660 |
| 0.4 | TabPFN | 0.726 | 0.725 | 0.632 | 0.682 |
| 0.4 | TabICL | 0.690 | 0.691 | 0.622 | 0.640 |
| 0.2 | TabPFN | 0.702 | 0.697 | 0.626 | 0.649 |
| 0.2 | TabICL | 0.688 | 0.688 | 0.640 | 0.639 |
| 0.1 | TabICL | 0.668 | 0.669 | 0.593 | 0.625 |
| 0.1 | TabPFN | 0.647 | 0.651 | 0.587 | 0.611 |

At a coarse level, both models degrade as the train fraction decreases. However, the overall averages hide an essential fact: the five relation types are **not equally learnable**.

### 4.2 Mean Performance by Relation Type

The table below averages over all train fractions and all runs.

| Relation | TabPFN Mean AP | TabICL Mean AP | Interpretation |
| --- | ---: | ---: | --- |
| `symmetric_transitive` | 0.922 | 0.886 | Strongly learnable |
| `antisymmetric_transitive` | 0.825 | 0.760 | Strongly learnable |
| `transitive` | 0.801 | 0.810 | Strongly learnable |
| `antisymmetric` | 0.505 | 0.509 | Near chance |
| `symmetric` | 0.499 | 0.503 | Near chance |

This is the central empirical pattern in the atomic benchmark.

## 5. Interpretation of the Results

### 5.1 Symmetry and Antisymmetry Alone Are Not Learned

For the plain `symmetric` and `antisymmetric` cases, both models remain essentially at chance:

- `symmetric`
  - `TabPFN`: AP `0.499`, ROC AUC `0.494`
  - `TabICL`: AP `0.503`, ROC AUC `0.500`
- `antisymmetric`
  - `TabPFN`: AP `0.505`, ROC AUC `0.499`
  - `TabICL`: AP `0.509`, ROC AUC `0.503`

These values are too close to `0.5` to support any strong claim of learned structure.

This outcome is theoretically coherent. In the current dataset generator, the non-transitive symmetric and antisymmetric relations are still largely random matrices subject only to the relevant pairwise constraint. As a result, observing some entries of the matrix leaves little information about many unseen entries. Put differently, there is no strong global regularity to recover beyond local consistency.

Therefore, the atomic results do **not** support the claim that TFMs can infer arbitrary balanced binary relations from partial observation alone.

### 5.2 Transitivity Produces a Learnable Global Structure

The picture changes completely for the three transitive cases:

- `transitive`
  - `TabPFN`: AP `0.801`
  - `TabICL`: AP `0.810`
- `antisymmetric_transitive`
  - `TabPFN`: AP `0.825`
  - `TabICL`: AP `0.760`
- `symmetric_transitive`
  - `TabPFN`: AP `0.922`
  - `TabICL`: AP `0.886`

These are far above chance and remain strong even when only `10%` of the matrix is used for training.

At train fraction `0.1`:

- `transitive`
  - `TabPFN`: AP `0.700`
  - `TabICL`: AP `0.767`
- `antisymmetric_transitive`
  - `TabPFN`: AP `0.722`
  - `TabICL`: AP `0.734`
- `symmetric_transitive`
  - `TabPFN`: AP `0.822`
  - `TabICL`: AP `0.836`

This is important. It means the models are not merely memorizing observed edges. Instead, they are exploiting structural regularities that constrain many unobserved pairs at once. In practical terms, **transitivity supplies the global organization that makes generalization possible**.

### 5.3 Model Comparison

The model comparison is nuanced rather than absolute.

Across all cases and train fractions:

- `TabPFN` has a slightly higher grand mean AP (`0.711` vs `0.693`)
- `TabPFN` also has a slightly higher grand mean ROC AUC (`0.707` vs `0.691`)
- `TabICL` has a slightly higher grand mean F1 (`0.628` vs `0.616`)

At the level of individual relation families:

- `TabPFN` is stronger on the two combined structured cases:
  - `symmetric_transitive`
  - `antisymmetric_transitive`
- `TabICL` is marginally stronger on the pure `transitive` case on average, and is slightly more robust at the lowest train fraction in the overall table.

Thus, the atomic benchmark does not show a universal winner. Instead, it suggests that both TFMs can exploit strong relational structure, with `TabPFN` usually obtaining the best results when the matrix has especially rigid global organization.

## 6. What These Results Do and Do Not Show

The correct scientific interpretation is narrower than “TFMs learn logic” in a general symbolic sense.

These results do show that:

- TFMs can recover strong structural regularities in binary relation matrices;
- transitive organization is especially amenable to learning;
- performance remains strong even when the training fraction is reduced substantially.

These results do **not** yet show that:

- TFMs can infer arbitrary relation properties from sparse evidence;
- TFMs have learned a transferable symbolic rule system independent of the entity universe;
- TFMs can generalize to entirely new relation matrices or disjoint entity sets purely from a learned rule abstraction.

This limitation follows directly from the experimental design. Each relation is trained and tested within the **same fixed entity set and the same matrix instance**. The model is therefore performing interpolation over held-out ordered pairs, not transfer to a genuinely new world. The benchmark is still valuable, but it is a weaker test than the family split or any protocol based on disjoint relational universes.

## 7. Conclusion

The atomic experiments support the following conclusion:

**Tabular foundation models can learn atomic relation properties when those properties induce a strong global structure, especially transitivity. They do not learn the purely random symmetric and antisymmetric cases in any meaningful way.**

Accordingly, the answer to the question “can TFMs learn these kinds of relations?” is:

- **yes**, for structurally constrained transitive relations;
- **no**, for weakly constrained random relation matrices such as the plain symmetric and antisymmetric cases.

This is a meaningful positive result, but it should be stated carefully. The evidence supports **structural pattern learning within a fixed synthetic relation matrix**, not unrestricted symbolic rule induction.

## 8. Files Referenced

- Dataset generator: [`build_bouchard_atomic_datasets.py`](../scripts/build_bouchard_atomic_datasets.py)
- Atomic experiment runner: [`run_bouchard_atomic_experiment.py`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py)
- Per-run results analyzed here: [`atomic_per_run_case_results.csv`](../results/atomic/atomic_per_run_case_results.csv)
