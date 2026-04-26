# Faithfulness Report

This report covers the datasets generated under:

- [bouchard_family_relational/](../bouchard_family_relational)

The original working tree also contained a flat denormalized export. That flat export is not included in this curated package because the experiments use the normalized relational triples.

The generator used was:

- [build_bouchard_family_datasets.py](../../scripts/build_bouchard_family_datasets.py)

## Scope

This is a faithful build of the **family / kinship** half of the 2019 Bouchard paper in two export formats:

- one flat denormalized table
- one normalized relational package with explicit PK/FK links

It is **not** a full reproduction of the paper's other experiment family on atomic binary-relation properties.

## What Matches The Paper

These points are paper-aligned:

- `5` disjoint families
- `3` generations
- `17` kinship relations
- `4` core relations: `mother`, `father`, `son`, `daughter`
- `13` derived relations
- `23` persons per family
- `115` persons total
- within-family candidate atoms only
- total family atom count `5 x 23 x 23 x 17 = 44,965`
- paper-inspired split logic included as columns on every atom

These claims are validated in:

- [bouchard_family_relational/validation_report.md](../bouchard_family_relational/validation_report.md)

Observed generated counts:

- families: `5`
- persons: `115`
- marriages: `50`
- parent-child edges: `120`
- relation types: `17`
- pair-relation atoms: `44,965`

Per-family structure in the generated data:

- persons per family: `23`
- marriages per family: `10`
- parent-child edges per family: `24`

## Split Faithfulness

The generated atoms include these split columns:

- `split_random`
- `split_evidence_p10`
- `split_family_p10`
- `split_family_p00`

How they map to the paper:

- `split_random`: standard train/val/test random split
- `split_evidence_p10`: all 4 core relations forced into train, with `p = 0.1` for the 13 derived relations
- `split_family_p10`: all relations for four families in train, core relations for the fifth family in train, plus `p = 0.1` of fifth-family derived relations in train
- `split_family_p00`: same family holdout setup, but with `p = 0.0`, the hardest paper setting

Observed split sizes:

- `split_random`: `train 35,980`, `val 4,503`, `test 4,482`
- `split_evidence_p10`: `train 14,023`, `val 3,443`, `test 27,499`
- `split_family_p10`: `train 38,777`, `val 689`, `test 5,499`
- `split_family_p00`: `train 38,088`, `val 689`, `test 6,188`

## Deliberate Deviations

These are the main places where the build is pragmatic rather than a strict line-by-line reproduction:

- I exported the same synthetic family world in two formats because you asked for one flat dataset and one relational dataset. The paper itself does not present this flat-vs-relational packaging.
- I materialized one deterministic world, not the paper's full repeated benchmark sweep.
- I included only the family experiment family, not the separate atomic-relation-property generator.
- I implemented the split logic as reusable split columns on every atom rather than as separate benchmark runs.
- I added a synthetic `event_time` column to the flat table so it is usable with the existing flat-table tooling in this repo.

## Important Structural Approximation

The paper gives the high-level family recipe and the target counts, but not a complete low-level construction listing for all `23` persons.

To hit the paper count exactly while keeping the family graph simple and rule-compatible, the generator uses:

- one root couple
- three generation-1 children
- one spouse for each generation-1 child
- three generation-2 children for each generation-1 couple
- spouses for six of the generation-2 children

This yields exactly `23` persons per family and preserves the kinship logic needed for:

- sibling relations
- aunt/uncle and nephew/niece
- cousin
- grandparent and grandchild

So the **counts and kinship semantics are faithful**, while the exact hidden family micro-topology is a reasonable reconstruction rather than something explicitly enumerated in the paper text.

## Output-Specific Notes

### Flat Table

The flat dataset is:

- one CSV file
- one row per candidate atom
- target column: `label`
- joined source/target person attributes
- aligned with the relational atoms by `atom_id`

Flat-table validation passed with:

- `44,965` rows
- `39` columns
- no missing values
- both classes present in train

### Relational Package

The relational dataset includes:

- `families.csv`
- `persons.csv`
- `marriages.csv`
- `parent_child_edges.csv`
- `relation_types.csv`
- `pair_relation_atoms.csv`

All declared PK/FK checks passed with zero violations.

## Bottom Line

If your criterion is "does this preserve the paper's family-experiment scale, relation inventory, disjoint-family assumption, and split logic in a usable dataset package?", the answer is **yes**.

If your criterion is "is this a byte-for-byte reproduction of the original unpublished family generator?", the answer is **no**. The exact internal family topology is reconstructed from the paper's stated constraints and counts, then exported in the flat and relational forms you requested.
