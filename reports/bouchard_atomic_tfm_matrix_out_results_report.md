# Can Tabular Foundation Models Transfer Atomic Relation Properties to New Matrices?

## 1. Scope and Empirical Basis

This report analyzes the strongest atomic Bouchard protocol currently implemented for tabular foundation models (`TabPFN` and `TabICL`):

- dataset generator: [`build_bouchard_atomic_datasets.py`](../scripts/build_bouchard_atomic_datasets.py)
- experiment runner: [`run_bouchard_atomic_experiment.py`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py)
- result file analyzed here: [`atomic_per_run_case_matrix_out.csv`](../results/atomic/atomic_per_run_case_matrix_out.csv)

This experiment uses `split_mode = leave_matrix_out_support`, which is substantially stronger than the earlier within-matrix protocol analyzed in:

- [`atomic_per_run_case_results.csv`](../results/atomic/atomic_per_run_case_results.csv)

The central question is:

**Can TFMs transfer atomic relation structure to a genuinely new relation matrix when given only sparse support from that new world?**

That question is stronger than the earlier one-matrix interpolation benchmark, because the model is no longer trained and tested only inside the same matrix instance.

## 2. Dataset Construction

The atomic generator instantiates the five main Bouchard cases in [`build_bouchard_atomic_datasets.py:31`](../scripts/build_bouchard_atomic_datasets.py:31):

- `symmetric`
- `antisymmetric`
- `transitive`
- `symmetric_transitive`
- `antisymmetric_transitive`

Multiple independent matrices per case are enabled through `--matrices-per-case` in [`build_bouchard_atomic_datasets.py:133`](../scripts/build_bouchard_atomic_datasets.py:133). The generator writes:

- `atomic_relation_atoms.csv` in [`build_bouchard_atomic_datasets.py:644`](../scripts/build_bouchard_atomic_datasets.py:644)
- `metadata.json` in [`build_bouchard_atomic_datasets.py:647`](../scripts/build_bouchard_atomic_datasets.py:647)
- `relation_types.csv` with `matrix_count` in [`build_bouchard_atomic_datasets.py:655`](../scripts/build_bouchard_atomic_datasets.py:655)

Each atomic matrix is a binary relation over `50` entities, so each matrix yields:

- `50 x 50 = 2500` ordered-pair atoms

The generator enforces approximate class balance, so the positive rate remains very close to `0.5`. In the analyzed result file, the mean `positive_rate_true` is `0.5002`. Therefore:

- `average_precision ≈ 0.5` means chance
- `roc_auc ≈ 0.5` means chance
- `accuracy ≈ 0.5` also means chance

This balancing is crucial for interpretation. Unlike the family benchmark, the atomic benchmark does not hide performance behind class imbalance.

## 3. Experimental Protocol

The relevant split logic is implemented in [`build_split_frames_for_case(...)`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:414). For `leave_matrix_out_support`, the runner:

1. chooses one `matrix_id` as the held-out matrix;
2. puts **all rows from the other matrices of the same relation type** into training;
3. samples a support subset from the held-out matrix according to `train_fraction`;
4. splits the remaining held-out rows into validation and test sets.

This logic is defined in [`run_bouchard_atomic_experiment.py:482`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:482) through [`run_bouchard_atomic_experiment.py:510`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:510).

To avoid entity-ID collisions across matrices, the runner uses matrix-scoped entity identifiers in [`make_feature_frame(...)`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:300) and activates this behavior specifically for the leave-matrix-out regime in [`run_bouchard_atomic_experiment.py:912`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py:912).

The analyzed CSV contains exactly `4000` rows, which corresponds to:

- `5 relation types`
- `10 held-out matrices per relation`
- `4 support fractions`
- `10 random runs`
- `2 models`

That is:

- `5 x 10 x 4 x 10 x 2 = 4000`

The structural metadata inside the CSV further confirms the protocol:

- `transfer_matrix_count = 9`
- `transfer_train_rows = 22500`
- `support_pool_rows = 2500`
- `heldout_support_rows = 250, 500, 1000, 2000`

So, for each relation case, the model trains on:

- all `9 x 2500 = 22500` rows from the non-held-out matrices;
- plus a sparse support subset from the new matrix.

This means the experiment is **not zero-support transfer**. It is better described as:

- **cross-matrix transfer with sparse adaptation evidence**

That distinction matters for the scientific claim.

## 4. Aggregate Results

### 4.1 Overall Mean Performance by Support Fraction

The table below averages over all relations, all held-out matrices, and all runs.

| Support Fraction | Model | Mean AP | Mean ROC AUC | Mean F1 | Mean Accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.1 | TabICL | 0.559 | 0.563 | 0.524 | 0.543 |
| 0.2 | TabICL | 0.577 | 0.580 | 0.541 | 0.555 |
| 0.4 | TabICL | 0.595 | 0.596 | 0.572 | 0.566 |
| 0.8 | TabICL | 0.620 | 0.618 | 0.623 | 0.581 |
| 0.1 | TabPFN | 0.537 | 0.541 | 0.526 | 0.526 |
| 0.2 | TabPFN | 0.547 | 0.550 | 0.522 | 0.533 |
| 0.4 | TabPFN | 0.558 | 0.561 | 0.518 | 0.541 |
| 0.8 | TabPFN | 0.568 | 0.570 | 0.561 | 0.548 |

Two patterns are immediate:

- both models improve monotonically as more support rows from the held-out matrix are provided;
- `TabICL` is consistently stronger than `TabPFN` under this cross-matrix transfer regime.

Across the whole file, the grand means are:

| Model | Mean AP | Mean ROC AUC | Mean F1 | Mean Accuracy |
| --- | ---: | ---: | ---: | ---: |
| TabICL | 0.588 | 0.589 | 0.565 | 0.561 |
| TabPFN | 0.553 | 0.556 | 0.532 | 0.537 |

### 4.2 Mean Performance by Relation Type

The table below averages over all support fractions and all runs.

| Relation | TabICL Mean AP | TabPFN Mean AP | Interpretation |
| --- | ---: | ---: | --- |
| `symmetric` | 0.505 | 0.503 | Chance |
| `antisymmetric` | 0.505 | 0.504 | Chance |
| `transitive` | 0.645 | 0.581 | Clearly above chance |
| `symmetric_transitive` | 0.647 | 0.586 | Clearly above chance |
| `antisymmetric_transitive` | 0.637 | 0.589 | Clearly above chance |

This is the central result of the stronger protocol.

The plain `symmetric` and `antisymmetric` cases remain essentially unlearned even after training on nine other matrices of the same relation family. By contrast, all three transitive-family cases transfer substantially above chance.

### 4.3 Support-Fraction Trends by Relation

The support fraction is especially informative because it measures how strongly the model benefits from sparse evidence from the new matrix.

For `TabICL`:

| Relation | AP at 0.1 | AP at 0.2 | AP at 0.4 | AP at 0.8 |
| --- | ---: | ---: | ---: | ---: |
| `antisymmetric` | 0.501 | 0.503 | 0.505 | 0.512 |
| `antisymmetric_transitive` | 0.596 | 0.622 | 0.648 | 0.681 |
| `symmetric` | 0.507 | 0.504 | 0.505 | 0.502 |
| `symmetric_transitive` | 0.603 | 0.637 | 0.661 | 0.688 |
| `transitive` | 0.586 | 0.617 | 0.656 | 0.719 |

For `TabPFN`:

| Relation | AP at 0.1 | AP at 0.2 | AP at 0.4 | AP at 0.8 |
| --- | ---: | ---: | ---: | ---: |
| `antisymmetric` | 0.502 | 0.502 | 0.505 | 0.505 |
| `antisymmetric_transitive` | 0.566 | 0.582 | 0.594 | 0.615 |
| `symmetric` | 0.500 | 0.503 | 0.509 | 0.502 |
| `symmetric_transitive` | 0.560 | 0.576 | 0.595 | 0.612 |
| `transitive` | 0.558 | 0.572 | 0.588 | 0.608 |

The interpretation is clear:

- for the non-transitive cases, increasing support does not help in any meaningful way;
- for the transitive-family cases, increasing support helps consistently and substantially.

This strongly suggests that the models are not simply memorizing training matrices. They are extracting a relation-specific prior that becomes useful when even a modest amount of held-out-matrix evidence is provided.

## 5. Comparison with the Earlier Within-Matrix Protocol

The previous report based on [`atomic_per_run_case_results.csv`](../results/atomic/atomic_per_run_case_results.csv) analyzed a weaker benchmark: train/test splits inside the same matrix. Comparing the two protocols is important because it shows how much of the earlier performance depended on within-world interpolation.

The mean AP changes are:

| Relation | Model | Within-Matrix AP | Leave-Matrix-Out AP | Delta |
| --- | --- | ---: | ---: | ---: |
| `symmetric` | TabICL | 0.503 | 0.505 | +0.002 |
| `antisymmetric` | TabICL | 0.509 | 0.505 | -0.004 |
| `transitive` | TabICL | 0.810 | 0.645 | -0.165 |
| `symmetric_transitive` | TabICL | 0.886 | 0.647 | -0.238 |
| `antisymmetric_transitive` | TabICL | 0.760 | 0.637 | -0.123 |
| `symmetric` | TabPFN | 0.499 | 0.503 | +0.004 |
| `antisymmetric` | TabPFN | 0.505 | 0.504 | -0.002 |
| `transitive` | TabPFN | 0.801 | 0.581 | -0.220 |
| `symmetric_transitive` | TabPFN | 0.922 | 0.586 | -0.336 |
| `antisymmetric_transitive` | TabPFN | 0.825 | 0.589 | -0.236 |

This comparison shows two things:

1. the earlier benchmark was indeed easier, especially for the transitive-family cases;
2. despite the large drop, the new protocol still leaves the transitive-family cases clearly above chance.

So the stronger transfer test does not erase the signal. It narrows it considerably and makes the claim much more defensible.

## 6. Interpretation

### 6.1 What the Strong Protocol Shows

The strongest safe conclusion is:

**TFMs can transfer some atomic relational structure across independent matrices, but only when the relation family imposes strong global constraints, especially transitivity.**

This is a stronger conclusion than the earlier within-matrix report allowed. There, the models might still have been benefiting mostly from interpolation over a single fixed world. Here, they are tested on a genuinely new matrix instance.

The result is therefore meaningful:

- `TabICL` and `TabPFN` do not simply memorize one matrix;
- they carry over a usable inductive bias for transitive-type relations;
- that bias becomes more useful as sparse support from the new matrix increases.

### 6.2 What the Strong Protocol Still Does Not Show

The current leave-matrix-out benchmark is strong, but it is not the strongest imaginable symbolic test.

It still does **not** show that:

- TFMs can infer a new atomic matrix with zero support from that matrix;
- TFMs have learned a fully explicit symbolic rule system independent of the observed universe;
- TFMs can reconstruct arbitrary symmetric or antisymmetric matrices from sparse evidence.

The reason is direct: the protocol gives the model sparse support rows from the held-out matrix. That is appropriate and scientifically useful, but it means the model is performing:

- **transfer plus adaptation**

rather than:

- **pure zero-shot rule induction**

Accordingly, the correct claim is not “TFMs learned logic in full generality.” The correct claim is narrower and stronger:

- TFMs learn a transferable prior for highly structured relation families;
- that prior is weak or useless for underdetermined relation families such as plain symmetry and antisymmetry.

### 6.3 Why F1 Needs Caution

In this benchmark, F1 can be misleading. For example, some `symmetric` rows show moderate F1 despite AP and ROC AUC being essentially at chance. This happens because the model sometimes predicts the positive class too frequently, and with balanced classes that can inflate threshold-dependent metrics.

Therefore, the most reliable scientific metrics here are:

- average precision
- ROC AUC

Those metrics are threshold-independent and directly answer whether the model is ranking true atoms above false ones.

## 7. Model Comparison

Under this stronger transfer protocol, `TabICL` is the better model.

This is visible:

- overall, where `TabICL` exceeds `TabPFN` on AP (`0.588` vs `0.553`);
- on every transitive-family relation;
- and in the way performance scales with more held-out support.

The ranking by relation family is also stable across both models:

1. `transitive`, `symmetric_transitive`, and `antisymmetric_transitive` are learnable above chance;
2. `symmetric` and `antisymmetric` remain effectively random.

So the dominant scientific distinction is **not** “which model wins everywhere?” It is:

- **which relation properties induce transferable structure at all?**

The answer is:

- transitivity does;
- plain symmetry and plain antisymmetry do not.

## 8. Conclusion

The leave-matrix-out atomic experiment supports the following conclusion:

**Tabular foundation models can transfer atomic relation structure to new matrices when that structure is globally constraining, especially in transitive relation families. They do not transfer plain symmetric or antisymmetric relations in a meaningful way.**

This is a stronger and more defensible result than the earlier within-matrix benchmark. It supports:

- partial cross-world relational transfer;
- successful adaptation from sparse support on a new matrix;
- a clear advantage for `TabICL` over `TabPFN` in this harder setting.

At the same time, it does **not** yet justify the strongest symbolic claim. The benchmark still provides support rows from the held-out matrix, so the results are best interpreted as:

- **transferable structural learning with sparse adaptation**

rather than as:

- **pure zero-shot symbolic rule induction**

## 9. Files Referenced

- Generator: [`build_bouchard_atomic_datasets.py`](../scripts/build_bouchard_atomic_datasets.py)
- Runner: [`run_bouchard_atomic_experiment.py`](../experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py)
- Strong-protocol results: [`atomic_per_run_case_matrix_out.csv`](../results/atomic/atomic_per_run_case_matrix_out.csv)
- Earlier within-matrix results: [`atomic_per_run_case_results.csv`](../results/atomic/atomic_per_run_case_results.csv)
- Earlier report: [`bouchard_atomic_tfm_results_report.md`](bouchard_atomic_tfm_results_report.md)
