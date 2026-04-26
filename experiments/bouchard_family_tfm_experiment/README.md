# Bouchard Family TFM Experiment

This experiment tests `TabPFN` and `TabICL` on the synthetic family/kinship relational dataset. It evaluates whether tabular foundation models can use in-context relational facts to predict held-out kinship triples.

The available runners are:

- [run_bouchard_family_experiment.py](run_bouchard_family_experiment.py): direct experiment runner.
- [run_bouchard_family_overnight.py](run_bouchard_family_overnight.py): launcher for a smoke run plus the full suite.

The dataset generator is:

- [build_bouchard_family_datasets.py](../../scripts/build_bouchard_family_datasets.py)

## Dataset

The generated relational dataset is included at:

- [datasets/bouchard_family_relational](../../datasets/bouchard_family_relational)

It contains five disjoint synthetic families, 115 people, 17 kinship relations, and 44,965 closed-world candidate atoms of the form:

```text
(relation, source_person_id, target_person_id) -> label
```

The experiment uses the minimal feature view needed for the relational task: `relation`, `source_person_id`, and `target_person_id`. It does not pass engineered attributes such as sex, generation, branch, or role code as model features.

To regenerate the family dataset from the repository root:

```bash
python scripts/build_bouchard_family_datasets.py \
  --output-root datasets
```

## Split Protocols

The runner supports three split regimes:

- `random`: random atom-level split stratified by relation, family, and label.
- `evidence`: core relations are always available as evidence; derived relations are partially observed according to `p`.
- `family`: one family is held out; a fraction `p` of that held-out family's derived facts is available as in-context support.

The `family` split also supports `p = 0.0` with `--include-family-zero`. In that setting, derived facts from the held-out family are absent from the training context, so the task is closer to transfer across family instances than interpolation within one family.

## Direct Runner

Dry-run the schedule without loading models:

```bash
python experiments/bouchard_family_tfm_experiment/run_bouchard_family_experiment.py \
  --data-root datasets/bouchard_family_relational \
  --output-dir experiments/bouchard_family_tfm_experiment \
  --include-family-zero \
  --dry-run
```

Run the full family suite on GPU:

```bash
TABPFN_DISABLE_TELEMETRY=1 python experiments/bouchard_family_tfm_experiment/run_bouchard_family_experiment.py \
  --data-root datasets/bouchard_family_relational \
  --output-dir experiments/bouchard_family_tfm_experiment \
  --tabpfn-model-path /path/to/tabpfn-classifier.ckpt \
  --tabicl-model-path /path/to/tabicl-classifier.ckpt \
  --device cuda \
  --include-family-zero \
  --splits random,evidence,family \
  --p-values 0.8,0.4,0.2,0.1 \
  --num-runs 10
```

Run a small family-split smoke test:

```bash
TABPFN_DISABLE_TELEMETRY=1 python experiments/bouchard_family_tfm_experiment/run_bouchard_family_experiment.py \
  --data-root datasets/bouchard_family_relational \
  --output-dir experiments/bouchard_family_tfm_experiment \
  --tabpfn-model-path /path/to/tabpfn-classifier.ckpt \
  --tabicl-model-path /path/to/tabicl-classifier.ckpt \
  --device cuda \
  --splits family \
  --p-values 0.1 \
  --num-runs 1 \
  --max-train-rows 12000 \
  --max-valid-rows 1000 \
  --max-test-rows 2500
```

To reproduce the tuned-threshold summaries, add:

```text
--threshold-mode tune_per_relation_f1
```

## Overnight Launcher

Dry-run the launcher plan:

```bash
python experiments/bouchard_family_tfm_experiment/run_bouchard_family_overnight.py \
  --data-root datasets/bouchard_family_relational \
  --tabpfn-model-path /path/to/tabpfn-classifier.ckpt \
  --tabicl-model-path /path/to/tabicl-classifier.ckpt \
  --dry-run
```

Run the smoke step plus the full suite:

```bash
TABPFN_DISABLE_TELEMETRY=1 python experiments/bouchard_family_tfm_experiment/run_bouchard_family_overnight.py \
  --data-root datasets/bouchard_family_relational \
  --tabpfn-model-path /path/to/tabpfn-classifier.ckpt \
  --tabicl-model-path /path/to/tabicl-classifier.ckpt \
  --device cuda \
  --num-runs 10
```

Run only the smoke step:

```bash
TABPFN_DISABLE_TELEMETRY=1 python experiments/bouchard_family_tfm_experiment/run_bouchard_family_overnight.py \
  --data-root datasets/bouchard_family_relational \
  --tabpfn-model-path /path/to/tabpfn-classifier.ckpt \
  --tabicl-model-path /path/to/tabicl-classifier.ckpt \
  --device cuda \
  --smoke-only
```

## Outputs

Each direct run writes `data/` and `results/` under the selected `--output-dir`. The overnight launcher writes each step under its selected `--run-root`.

Main output files:

- `results/planned_runs.csv`
- `results/per_run_results.csv`
- `results/thresholds.csv`
- `results/summary_by_model_split_p_relation.csv`
- `results/summary_overall_by_model_split_p.csv`
- `results/bouchard_family_comparable_summary.csv`
- `results/metadata.json`
- `results/findings.md`

Archived result CSVs from previous runs are included under [results/family](../../results/family).
