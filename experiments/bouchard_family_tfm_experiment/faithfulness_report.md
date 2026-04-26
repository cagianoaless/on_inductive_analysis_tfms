# TFM Faithfulness Report

This report covers the TFM experiment scripts now available for the Bouchard setups.

## Available Runners

Already present before this change:

- `../bouchard_tfm_experiment/run_bouchard_countries_experiment.py`
- `../bouchard_tfm_experiment/run_bouchard_countries_overnight.py`

Added in this change:

- [run_bouchard_family_experiment.py](run_bouchard_family_experiment.py)
- [run_bouchard_family_overnight.py](run_bouchard_family_overnight.py)
- [README.md](README.md)

## What Is Covered

### 2015 Bouchard Countries Setup

Covered by the existing runner in `../bouchard_tfm_experiment`.

That script already supports:

- the `isInside` / `isNeighbor` task
- continent-wise holdout
- `TabPFN`
- `TabICL`
- single runs and overnight suites

### 2019 Bouchard Family / Kinship Setup

Covered by the new runner in this folder.

The new family runner is faithful on these points:

- input dataset: the generated family dataset at `datasets/bouchard_family_relational`
- candidate facts are binary kinship atoms
- feature view is the minimal Bouchard-style triple representation:
  - `relation`
  - `source_person_id`
  - `target_person_id`
- split regimes:
  - `random`
  - `evidence`
  - `family`
- `p` values:
  - `0.8`
  - `0.4`
  - `0.2`
  - `0.1`
- `family` split also supports `p = 0.0`
- default run count is `10`, matching the paper's repeated-run evaluation style
- outputs include per-run results and aggregated summaries

## What The New Family Script Does

For each configuration, it:

- builds the requested split regime
- fits `TabPFN` and/or `TabICL`
- evaluates on the held-out test set
- records:
  - average precision
  - ROC AUC
  - F1
  - precision
  - recall
  - accuracy
  - fit / predict time

It writes:

- `results/per_run_results.csv`
- `results/summary_by_model_split_p_relation.csv`
- `results/summary_overall_by_model_split_p.csv`
- `results/bouchard_family_comparable_summary.csv`
- `results/findings.md`
- `results/metadata.json`

## Deliberate Deviations

These are the main places where the family runner is pragmatic rather than a literal reproduction:

- It runs on one fixed synthetic family world that was generated locally, rather than regenerating a fresh family world per run.
- The held-out family defaults to `family_05`, rather than cycling holdout families automatically.
- The script includes optional threshold-based classification metrics because TFMs return probabilities and these diagnostics are useful, but the paper-aligned metric remains available through average precision.
- There is no latent-rank sweep because that part is specific to the original latent factor models, not to TFMs.

## What Is Not Yet Covered

The current new TFM runner covers the **family experiment** only.

It does **not** yet implement the separate 2019 **atomic binary-relation property** experiment family:

- symmetric
- antisymmetric
- transitive
- symmetric + transitive
- antisymmetric + transitive
- plus the reflexive / irreflexive appendix cases

So the current TFM coverage is:

- 2015 countries: covered
- 2019 family: covered
- 2019 atomic relation properties: not yet scripted here

## Validation Performed

The new family scripts were checked locally with:

- Python compile:
  - `python3 -m py_compile ...`
- dry-run of the main family runner:
  - dataset loaded correctly
  - `5` families
  - `17` relations
  - `44,965` atoms
  - `130` planned configurations for the full `random/evidence/family` grid with `10` runs and `p in {0.8,0.4,0.2,0.1}` plus family `p=0.0`
- dry-run of the overnight launcher:
  - smoke command rendered correctly
  - full paper-style command rendered correctly

## Bottom Line

If your criterion is "can I now rerun the Bouchard TFM setups for countries and family later on a GPU machine?", the answer is **yes**.

If your criterion is "is every experiment family from the 2019 paper now implemented for TFMs?", the answer is **no**. The atomic relation-property runner is still missing.
