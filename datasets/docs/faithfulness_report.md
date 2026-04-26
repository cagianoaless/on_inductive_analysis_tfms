# Dataset Faithfulness Report

This report covers the packaged family dataset:

- [bouchard_family_relational/](../bouchard_family_relational)

The generator used to build it is:

- [build_bouchard_family_datasets.py](../../scripts/build_bouchard_family_datasets.py)

## Scope

The included dataset is the family/kinship part of the Bouchard-style inductive-abilities benchmark. It is exported as a normalized relational package because the TFM experiments consume triples of the form:

```text
(relation, source_person_id, target_person_id) -> label
```

The generator can also produce a denormalized flat table, but the flat export is not included in this curated repo folder.

The atomic binary-relation benchmark is separate and is documented under [../../experiments/bouchard_atomic_tfm_experiment](../../experiments/bouchard_atomic_tfm_experiment).

## What Matches The Intended Family Setup

These points are aligned with the family benchmark structure:

- `5` disjoint families
- `3` generations
- `17` kinship relations
- `4` core relations: `mother`, `father`, `son`, `daughter`
- `13` derived relations
- `23` persons per family
- `115` persons total
- within-family candidate atoms only
- total family atom count `5 x 23 x 23 x 17 = 44,965`
- paper-inspired random, evidence, and family split columns

Observed generated counts:

- families: `5`
- persons: `115`
- marriages: `50`
- parent-child edges: `120`
- relation types: `17`
- pair-relation atoms: `44,965`

The structural checks are recorded in:

- [bouchard_family_relational/validation_report.md](../bouchard_family_relational/validation_report.md)

## Split Columns

The generated atoms include:

- `split_random`
- `split_evidence_p10`
- `split_family_p10`
- `split_family_p00`

Interpretation:

- `split_random`: standard train/validation/test atom split.
- `split_evidence_p10`: all core relations are train evidence, with `p = 0.1` for derived relations.
- `split_family_p10`: four families are train evidence; the held-out family keeps core relations plus `p = 0.1` of derived facts as support.
- `split_family_p00`: same family holdout setup, but with no held-out-family derived support facts.

Observed split sizes:

- `split_random`: `train 35,980`, `val 4,503`, `test 4,482`
- `split_evidence_p10`: `train 14,023`, `val 3,443`, `test 27,499`
- `split_family_p10`: `train 38,777`, `val 689`, `test 5,499`
- `split_family_p00`: `train 38,088`, `val 689`, `test 6,188`

## Deliberate Deviations

These are pragmatic reconstruction choices:

- The package includes one deterministic synthetic family world, not a recovered unpublished original generator output.
- The exact low-level family topology is reconstructed from the stated constraints and counts.
- The split logic is materialized as columns on every atom, which makes the dataset easy to audit and reuse.
- The flat export produced by the generator is omitted from this curated package because it is redundant for the included experiments.

## Bottom Line

The packaged relational dataset preserves the intended family-experiment scale, relation inventory, disjoint-family assumption, closed-world positive/negative labeling, and split logic. It should not be described as a byte-for-byte reproduction of an unpublished original data file.
