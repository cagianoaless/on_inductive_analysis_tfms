# Family Experiment Faithfulness Report

This note records what the packaged family experiment implements relative to the Bouchard-style inductive-abilities setup.

## Current Scope

The packaged experiment covers the synthetic family/kinship benchmark only. The atomic binary-relation benchmark is implemented separately in [../bouchard_atomic_tfm_experiment](../bouchard_atomic_tfm_experiment).

The runnable family files included in this package are:

- [run_bouchard_family_experiment.py](run_bouchard_family_experiment.py)
- [run_bouchard_family_overnight.py](run_bouchard_family_overnight.py)
- [build_bouchard_family_datasets.py](../../scripts/build_bouchard_family_datasets.py)

## Dataset Correspondence

The included dataset is [datasets/bouchard_family_relational](../../datasets/bouchard_family_relational). It is a normalized relational table with closed-world binary labels for every candidate triple:

```text
(relation, source_person_id, target_person_id) -> label
```

The generated package contains:

- `5` disjoint synthetic families.
- `115` people.
- `17` kinship relations.
- `44,965` candidate relation atoms.
- Positive and negative examples for each relation, produced by closed-world completion over within-family ordered pairs.

The experiment runner uses only the symbolic triple columns as model features. Family-construction metadata such as sex, generation, branch, and role code is retained in the dataset files for auditability, but it is not used as model input by the runner.

## Split Correspondence

The runner implements the three family protocols:

- `random`: atom-level interpolation within the generated families.
- `evidence`: core parent/child relations are supplied as evidence while derived relations are partially observed.
- `family`: one family is held out, with optional support facts from that same family controlled by `p`.

The `family` split can include `p = 0.0` through `--include-family-zero`. This is the strongest family protocol in this runner because the held-out family's derived facts are not available as support examples.

## Known Deviations

The current implementation is faithful to the tabular TFM version of the benchmark, not to the original model family used in the 2019 paper.

Important differences:

- The models are `TabPFN` and `TabICL`, not latent-factor or tensor-factorization models.
- The benchmark uses generated synthetic family instances rather than a recovered original data file.
- Metrics include threshold-dependent scores such as F1, precision, and recall in addition to average precision.
- Threshold tuning, when enabled, is performed on the validation split and should be reported separately from fixed-threshold results.

These differences mean that the experiment is appropriate for evaluating whether TFMs exploit the tabular relational representation, but it should not be described as an exact reproduction of the original training pipeline.

## Validation Pointers

Useful audit files:

- [validation_report.md](../../datasets/bouchard_family_relational/validation_report.md)
- [relation_types.csv](../../datasets/bouchard_family_relational/relation_types.csv)
- [pair_relation_atoms.csv](../../datasets/bouchard_family_relational/pair_relation_atoms.csv)

Useful dry run:

```bash
python experiments/bouchard_family_tfm_experiment/run_bouchard_family_experiment.py \
  --data-root datasets/bouchard_family_relational \
  --output-dir experiments/bouchard_family_tfm_experiment \
  --include-family-zero \
  --dry-run
```
