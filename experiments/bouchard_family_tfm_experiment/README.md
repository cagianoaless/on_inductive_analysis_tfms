# Bouchard Family TFM Experiment

This folder contains launchers to rerun the **family / kinship** part of the 2019 Bouchard paper with:

- `TabPFN-2.5`
- `TabICLv2`

The experiment reads the generated family dataset from:

- [datasets/bouchard_family_relational](../../datasets/bouchard_family_relational)

Seed-specific `dataset_groups/` outputs from the working tree were intentionally not included in this curated package because they duplicate the relational family dataset.

and evaluates the paper's three split regimes:

- `random`
- `evidence`
- `family`

## Main Script

The main runner is:

- [run_bouchard_family_experiment.py](run_bouchard_family_experiment.py)

Dry-run the full schedule without loading models:

```bash
cd <repo-root>
python experiments/bouchard_family_tfm_experiment/run_bouchard_family_experiment.py \
  --data-root datasets/bouchard_family_relational \
  --output-dir experiments/bouchard_family_tfm_experiment \
  --dry-run --include-family-zero
```

To run on another generated dataset, pass its relational dataset directory through `--data-root`.

Paper-style full run on GPU:

```bash
cd <repo-root>
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python experiments/bouchard_family_tfm_experiment/run_bouchard_family_experiment.py \
  --data-root datasets/bouchard_family_relational \
  --output-dir experiments/bouchard_family_tfm_experiment \
  --device cuda \
  --include-family-zero \
  --num-runs 10 \
  --splits random,evidence,family \
  --p-values 0.8,0.4,0.2,0.1
```

Useful smoke run:

```bash
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python run_bouchard_family_experiment.py \
  --device cuda \
  --splits family \
  --p-values 0.1 \
  --num-runs 1 \
  --max-train-rows 12000 \
  --max-valid-rows 1000 \
  --max-test-rows 2500
```

## Overnight Launcher

The suite launcher is:

- [run_bouchard_family_overnight.py](run_bouchard_family_overnight.py)

Dry-run the command plan:

```bash
cd <repo-root>
python experiments/bouchard_family_tfm_experiment/run_bouchard_family_overnight.py --dry-run
```

Run the smoke step plus the full paper-style suite:

```bash
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python run_bouchard_family_overnight.py \
  --device cuda \
  --num-runs 10
```

Smoke only:

```bash
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python run_bouchard_family_overnight.py \
  --device cuda \
  --smoke-only
```

## Outputs

Each run writes under its chosen `output-dir`:

- `data/dataset_summary.json`
- `results/planned_runs.csv`
- `results/per_run_results.csv`
- `results/thresholds.csv`
- `results/summary_by_model_split_p_relation.csv`
- `results/summary_overall_by_model_split_p.csv`
- `results/bouchard_family_comparable_summary.csv`
- `results/metadata.json`
- `results/findings.md`

## Notes

- The script uses the **minimal Bouchard-faithful feature view** of each atom: `relation`, `source_person_id`, `target_person_id`.
- `random` and `evidence` use the requested `p` values.
- `family` also supports `p = 0.0` via `--include-family-zero`.
- Threshold tuning, if enabled, uses the validation split. The paper metric is still available via average precision in the outputs.
- The original `countries` TFM experiment is not part of this curated Bouchard family/atomic package.
