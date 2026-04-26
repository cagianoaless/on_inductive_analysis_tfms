# Bouchard Family Dataset

This curated package includes the normalized synthetic family-kinship relational dataset:

- [bouchard_family_relational/](../bouchard_family_relational)

The generator can also create a denormalized flat export, but that flat table is not committed here because the packaged experiments use the relational triples directly.

Current included dataset shape:

- `5` disjoint families
- `23` persons per family
- `115` persons total
- `17` kinship relations
- `44,965` within-family candidate relation atoms
- closed-world labels for `(relation, source_person_id, target_person_id)`

The included validation report is:

- [bouchard_family_relational/validation_report.md](../bouchard_family_relational/validation_report.md)

To regenerate the family dataset from the repository root:

```bash
python scripts/build_bouchard_family_datasets.py \
  --output-root datasets
```
