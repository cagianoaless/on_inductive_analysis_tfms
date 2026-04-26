# Bouchard Family Datasets

This directory contains one synthetic family-kinship world exported in two forms:

- `bouchard_family_flat/`: one denormalized flat table for tabular models
- `bouchard_family_relational/`: a normalized multi-table relational package

Design choices:

- `5` disjoint families
- `23` persons per family
- `17` kinship relations
- only within-family candidate atoms are materialized, matching the paper-scale family count
- split columns include a random split plus paper-inspired evidence and family holdouts

The flat and relational versions are aligned at the `atom_id` level.
