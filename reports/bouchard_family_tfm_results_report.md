# Can Tabular Foundation Models Learn Synthetic Kinship Relations?

## Abstract

This report analyzes whether tabular foundation models (TFMs), specifically `TabPFN-2.5` and `TabICLv2`, can learn the synthetic kinship relations used in the Bouchard family experiment. The answer is **partially yes**. Both models learn useful relational signal under supportive training conditions, and `TabICL` consistently outperforms `TabPFN`. However, the results do **not** support the stronger claim that these models robustly acquire transferable rule-like genealogical reasoning. In the hardest transfer setting, where the held-out family contributes only the four core relations and no derived relations to training (`family`, `p = 0.0`), both models remain weak, especially under threshold-independent metrics such as average precision. The evidence therefore supports **partial relational learning**, but not strong symbolic generalization.

## 1. Materials and Data

The experiment uses the generated relational dataset in [datasets/bouchard_family_relational](../datasets/bouchard_family_relational). The dataset metadata in [metadata.json](../datasets/bouchard_family_relational/metadata.json) states:

- `5` families
- `115` people in total
- `17` kinship relations
- `44,965` candidate atoms

The relation inventory is defined in [relation_types.csv](../datasets/bouchard_family_relational/relation_types.csv):

- Core relations: `mother`, `father`, `son`, `daughter`
- Derived relations: `husband`, `wife`, `brother`, `sister`, `uncle`, `aunt`, `nephew`, `niece`, `cousin`, `grandfather`, `grandson`, `grandmother`, `granddaughter`

The main table is [pair_relation_atoms.csv](../datasets/bouchard_family_relational/pair_relation_atoms.csv). Each row is a candidate fact of the form `(relation, source_person_id, target_person_id, label)`, where `label = 1` denotes a true fact and `label = 0` a false fact. The dataset is therefore a **binary relation classification benchmark**, not an association-rule mining dataset.

Two structural properties of the dataset are important for interpretation:

- Only **within-family** candidate atoms are present; cross-family pairs are omitted.
- The dataset is strongly **imbalanced**. The overall positive rate is about `2.42%`.

The imbalance also varies by relation. `cousin` is the densest relation at about `10.2%` positives, while most other relations lie around `1.6%` to `2.3%`. This matters because high accuracy can be achieved by predicting mostly negatives, whereas average precision is more informative.

## 2. Experimental Setup

The experiment is implemented in [run_bouchard_family_experiment.py](../experiments/bouchard_family_tfm_experiment/run_bouchard_family_experiment.py) and documented in [README.md](../experiments/bouchard_family_tfm_experiment/README.md). The script uses the **minimal Bouchard-faithful triple representation**:

- `relation`
- `source_person_id`
- `target_person_id`

No engineered family attributes are used in the main runner. This design is described in [faithfulness_report.md](../experiments/bouchard_family_tfm_experiment/faithfulness_report.md).

Three split regimes are evaluated:

1. `random`: random partition of atoms across train, validation, and test.
2. `evidence`: all core relations are kept in training; derived relations are sampled.
3. `family`: four full families are observed in training; from the held-out family, training always contains the four core relations, while derived relations are sampled with proportion `p`.

The run grid follows the Bouchard family setup:

- `p in {0.8, 0.4, 0.2, 0.1}`
- plus `family, p = 0.0`
- `10` runs per configuration

The models evaluated are:

- `TabPFN-2.5`
- `TabICLv2`

The relevant result files are:

- Fixed-threshold overall summary: [bouchard_family_comparable_summary.csv](../results/family/bouchard_family_comparable_summary.csv)
- Fixed-threshold per-relation summary: [summary_by_model_split_p_relation.csv](../results/family/summary_by_model_split_p_relation.csv)
- Tuned per-relation F1 overall summary: [bouchard_family_comparable_summary_tune_per_f1.csv](../results/family/bouchard_family_comparable_summary_tune_per_f1.csv)
- Tuned per-relation F1 per-relation summary: [summary_by_model_split_p_relation_tune_per_f1.csv](../results/family/summary_by_model_split_p_relation_tune_per_f1.csv)

## 3. Evaluation Metrics

The script reports:

- average precision (AP)
- ROC AUC
- F1
- precision
- recall
- accuracy

For scientific interpretation, **average precision** is the primary metric because:

- it is threshold-independent,
- it is more robust to severe class imbalance,
- it better reflects whether the model ranks true facts above false facts.

F1, precision, recall, and accuracy are useful diagnostics, but they depend on the classification threshold. In the original run, the default threshold mode was fixed at `0.5`. In the tuned run, per-relation thresholds were chosen on the validation split.

## 4. Main Findings

### 4.1 Overall Result: `TabICL` Consistently Outperforms `TabPFN`

Across every split and every `p` value, `TabICL` achieves higher average precision than `TabPFN`.

Selected results from [bouchard_family_comparable_summary.csv](../results/family/bouchard_family_comparable_summary.csv):

| Split | `p` | TabICL AP | TabPFN AP |
| --- | ---: | ---: | ---: |
| `evidence` | 0.8 | 0.542 | 0.421 |
| `random` | 0.8 | 0.474 | 0.370 |
| `family` | 0.8 | 0.535 | 0.315 |
| `family` | 0.0 | 0.044 | 0.021 |

Averaged over all configurations, the mean AP is approximately:

- `TabICL`: `0.360`
- `TabPFN`: `0.247`

This is a stable and substantial advantage for `TabICL`.

### 4.2 Performance Decreases as Supervision Decreases

For both models, performance declines as `p` decreases. This is true in all split regimes. For example:

- `TabICL`, `evidence`: AP drops from `0.542` at `p = 0.8` to `0.265` at `p = 0.1`
- `TabPFN`, `random`: AP drops from `0.370` at `p = 0.8` to `0.121` at `p = 0.1`

This pattern indicates that the models are using the available support effectively, but their relational performance is strongly dependent on how much evidence is observed.

### 4.3 The Hardest Test, `family` with `p = 0.0`, Remains Weak

The most important configuration is `family, p = 0.0`. In this case, the held-out family contributes only the four core relations (`mother`, `father`, `son`, `daughter`) to training, and **no derived relations** from that family are observed during training.

This setting is the closest test of whether the model can transfer genealogical structure from one family to another without direct derived examples from the target family.

Results:

- `TabICL`: AP `0.044`
- `TabPFN`: AP `0.021`

Under the fixed threshold, both models had `F1 = 0.0` overall in this setting. This means they largely failed as classifiers. Even after per-relation threshold tuning, the AP values remain unchanged and very low. Therefore, the strongest claim that the models learned a transferable set of kinship rules is **not supported**.

### 4.4 Accuracy Is Misleading

In the hardest settings, accuracy remains very high even when the models are poor.

For `family, p = 0.0`:

- fixed-threshold accuracy is about `0.9755` for both models
- yet fixed-threshold `F1 = 0`

This happens because the dataset is dominated by negatives. High accuracy here does not mean successful kinship reasoning; it mostly means that predicting negatives is easy.

## 5. Relation-Level Analysis

The per-relation summary in [summary_by_model_split_p_relation.csv](../results/family/summary_by_model_split_p_relation.csv) shows that the relations are not equally learnable.

### 5.1 Easiest Relations

The strongest relation by far is `cousin`.

Mean AP across all configurations:

- `TabICL`: `0.734`
- `TabPFN`: `0.580`

Mean tuned per-relation F1:

- `TabICL`: `0.701`
- `TabPFN`: `0.570`

The next strongest relation is `grandfather`:

- Mean AP:
  - `TabICL`: `0.681`
  - `TabPFN`: `0.390`
- Mean tuned F1:
  - `TabICL`: `0.483`
  - `TabPFN`: `0.240`

Other moderately learnable relations include:

- `grandmother`
- `granddaughter`
- `grandson`

### 5.2 Hardest Relations

The weakest relations are:

- `husband`
- `wife`
- `brother`
- `sister`

For example, mean AP across all configurations is extremely low for spouse relations:

- `TabICL`: `husband = 0.029`, `wife = 0.043`
- `TabPFN`: `husband = 0.039`, `wife = 0.032`

This indicates that the models do not learn all kinship relations equally well. Instead, they appear to capture some relations much more easily than others.

### 5.3 Derived Family Relations Are Not Uniformly Solved

Several relations, such as `aunt`, `uncle`, `nephew`, and `niece`, often show decent AP or ROC AUC but poor fixed-threshold F1. This means the models may contain useful ranking signal for these relations, but the raw probability outputs are not well calibrated at a global threshold of `0.5`.

This distinction is critical:

- High AP means the model can often rank true facts above false facts.
- Low F1 means that when forced to make hard binary decisions, it often predicts too conservatively.

## 6. Effect of Per-Relation Threshold Tuning

The tuned results in [summary_by_model_split_p_relation_tune_per_f1.csv](../results/family/summary_by_model_split_p_relation_tune_per_f1.csv) change the interpretation of the classifier metrics, but not the ranking metrics.

What remains unchanged:

- AP
- ROC AUC

What improves substantially:

- relation-level F1
- recall
- the number of relations with nonzero positive predictions

At the relation level, zero-F1 cases were almost eliminated:

- `TabICL`: `124` zero-F1 relation/configuration cells to `0`
- `TabPFN`: `159` to `3`

Mean relation-level F1 increased by:

- `+0.144` for `TabICL`
- `+0.118` for `TabPFN`

This shows that the fixed-threshold runs understated the available signal. However, threshold tuning does **not** change the fundamental ranking quality. Therefore, the tuned results should be interpreted as evidence that the models can produce better binary decisions when calibrated, not as evidence that they learned deeper relational structure than indicated by AP.

## 7. Can TFMs Learn These Relations?

The answer depends on the strength of the claim.

### 7.1 Weak Claim: Yes, Partially

If the question is whether TFMs can learn statistically useful patterns over kinship triples, the answer is **yes**.

Evidence:

- Both models achieve AP well above the extreme transfer baseline in many configurations.
- `TabICL` is particularly strong on `cousin` and `grandfather`.
- Per-relation threshold tuning reveals that much of the learned signal was present but hidden by a fixed global threshold.

### 7.2 Strong Claim: No, Not Reliably

If the question is whether TFMs learn **transferable genealogical rules** in the stronger symbolic sense, the answer is **not convincingly**.

Evidence:

- In `family, p = 0.0`, where derived relations for the held-out family are entirely absent during training, AP remains very low for both models.
- The strongest model, `TabICL`, still reaches only `0.044` AP overall in that setting.
- Threshold tuning can force nonzero F1 in that regime, but this reflects threshold selection, not improved ranking.

Thus, the results support the view that TFMs can exploit **relational regularities**, but do not yet show robust rule transfer comparable to symbolic inference.

## 8. Interpretation and Limitations

Several limitations should be stated explicitly.

First, this implementation uses one fixed synthetic family world rather than regenerating a new world for each run, as noted in [faithfulness_report.md](../experiments/bouchard_family_tfm_experiment/faithfulness_report.md).

Second, the held-out family defaults to `family_05`, rather than cycling families.

Third, the model input is deliberately minimal: only relation and person IDs are provided. This makes the task faithful to the Bouchard triple formulation, but it also means the models are operating as tabular classifiers over symbolic identifiers rather than over an explicit logical proof system.

Fourth, the benchmark is a **closed-world binary classification** task containing both true and false facts. This is a legitimate evaluation setup, but it is not the same as open-world inductive logic learning from positive facts only.

## 9. Conclusion

The present evidence supports the following conclusion:

`TabICL` and `TabPFN` can learn **some** synthetic kinship relations from tabular relational triples, but their success is uneven across relations and highly dependent on the amount of supporting evidence. `TabICL` is consistently stronger than `TabPFN`. The models are particularly effective on relations such as `cousin` and, to a lesser extent, `grandfather`. However, the hard transfer setting (`family, p = 0.0`) remains weak, which indicates that these TFMs do not yet show strong, reliable rule-like generalization across disjoint families.

Accordingly, the best academic summary is:

> TFMs learn useful relational signal in the Bouchard family benchmark, but the current results do not justify the stronger claim that they have acquired robust transferable genealogical rules.

## 10. Files Referenced

- Dataset root: [datasets/bouchard_family_relational](../datasets/bouchard_family_relational)
- Dataset metadata: [metadata.json](../datasets/bouchard_family_relational/metadata.json)
- Relation definitions: [relation_types.csv](../datasets/bouchard_family_relational/relation_types.csv)
- Atom table: [pair_relation_atoms.csv](../datasets/bouchard_family_relational/pair_relation_atoms.csv)
- Main experiment runner: [run_bouchard_family_experiment.py](../experiments/bouchard_family_tfm_experiment/run_bouchard_family_experiment.py)
- Overnight launcher: [run_bouchard_family_overnight.py](../experiments/bouchard_family_tfm_experiment/run_bouchard_family_overnight.py)
- Fixed-threshold overall summary: [bouchard_family_comparable_summary.csv](../results/family/bouchard_family_comparable_summary.csv)
- Fixed-threshold per-relation summary: [summary_by_model_split_p_relation.csv](../results/family/summary_by_model_split_p_relation.csv)
- Tuned overall summary: [bouchard_family_comparable_summary_tune_per_f1.csv](../results/family/bouchard_family_comparable_summary_tune_per_f1.csv)
- Tuned per-relation summary: [summary_by_model_split_p_relation_tune_per_f1.csv](../results/family/summary_by_model_split_p_relation_tune_per_f1.csv)

## Split Regimes

  The Bouchard family experiment evaluates three different train/test split
  regimes.

  | Split | Training Set | Test Set | Interpretation |
  | --- | --- | --- | --- |
  | `random` | A random subset of all atoms from all families | The remaining
  atoms from the same global pool | Tests standard interpolation. Train and test
  come from the same families and the same overall world. |
  | `evidence` | All core relations (`mother`, `father`, `son`, `daughter`) for
  all families, plus a fraction `p` of the derived relations | Held-out derived
  relations | Tests whether the model can use the basic genealogical evidence to
  predict derived kinship relations. |
  | `family` | All atoms from four families, plus only the core relations of the
  held-out family, plus a fraction `p` of the held-out family’s derived
  relations | Held-out derived relations from the held-out family | Tests
  whether the model can transfer relational structure from known families to a
  new family. |

  The parameter `p` controls how much of the sampled part is included in
  training.

  | `p` value | Meaning |
  | --- | --- |
  | `0.8`, `0.4`, `0.2`, `0.1` | Increasingly smaller fractions of the sampled
  atoms are included in training. |
  | `0.0` in the `family` split | No derived relations from the held-out family
  are included in training. |

  A crucial clarification is that `family, p = 0.0` does **not** mean that the
  model has no training context. In that setting, the model still sees:

  - all facts from the other four families;
  - all core relations (`mother`, `father`, `son`, `daughter`) from the held-out
  family.

  What is withheld is only the held-out family’s **derived** relations.
  Therefore, `family, p = 0.0` is best interpreted as a test of whether the
  model can transfer genealogical structure to a new family when given only the
  core evidence for that family.


