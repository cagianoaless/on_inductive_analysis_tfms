# Bouchard Atomic TFM Experiment

This experiment tests `TabPFN` and `TabICL` on synthetic binary-relation matrices whose logical properties are controlled by construction. It is the atomic relation benchmark in this repository, separate from the family/kinship benchmark.

The runner is:

- [run_bouchard_atomic_experiment.py](run_bouchard_atomic_experiment.py)

The dataset generator is:

- [build_bouchard_atomic_datasets.py](../../scripts/build_bouchard_atomic_datasets.py)

## Dataset

The atomic dataset is deterministic and is not stored by default in this curated package. Regenerate it from the repository root:

```bash
python scripts/build_bouchard_atomic_datasets.py \
  --output-root datasets/bouchard_atomic_relational \
  --matrices-per-case 10
```

The default generator creates five main cases:

- `symmetric`
- `antisymmetric`
- `transitive`
- `symmetric_transitive`
- `antisymmetric_transitive`

Each generated matrix is a binary relation over all ordered entity pairs. With the default `--num-entities 50`, each matrix contributes `2,500` labeled atoms. Entries are generated internally as `+1` or `-1` and exported as binary labels.

To include the additional reflexive and irreflexive appendix cases:

```bash
python scripts/build_bouchard_atomic_datasets.py \
  --output-root datasets/bouchard_atomic_relational \
  --matrices-per-case 10 \
  --include-appendix-cases
```

## Split Protocols

The runner supports three protocols:

- `random_rows`: random train/validation/test rows inside each matrix.
- `entity_block`: holds out a query block of entities and evaluates on query-by-query pairs inside the same matrix.
- `leave_matrix_out_support`: trains on other matrices of the same relation type plus a support subset from the held-out matrix, then evaluates on the remaining held-out rows.

The strongest protocol currently implemented is `leave_matrix_out_support`, because it tests transfer across independently generated matrices instead of only interpolation inside one matrix. It requires at least two matrices per case, so use `--matrices-per-case 10` when generating the dataset.

## Dry Run

From the repository root:

```bash
python experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py \
  --data-root datasets/bouchard_atomic_relational \
  --output-dir experiments/bouchard_atomic_tfm_experiment \
  --split-mode leave_matrix_out_support \
  --dry-run
```

The dry run validates the dataset and writes the planned configuration grid without fitting models.

## GPU Run

The model checkpoints are external files and must be passed explicitly:

```bash
TABPFN_DISABLE_TELEMETRY=1 python experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py \
  --data-root datasets/bouchard_atomic_relational \
  --output-dir experiments/bouchard_atomic_tfm_experiment \
  --split-mode leave_matrix_out_support \
  --tabpfn-model-path /path/to/tabpfn-classifier.ckpt \
  --tabicl-model-path /path/to/tabicl-classifier.ckpt \
  --device cuda
```

Run only `TabICL`:

```bash
TABPFN_DISABLE_TELEMETRY=1 python experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py \
  --data-root datasets/bouchard_atomic_relational \
  --output-dir experiments/bouchard_atomic_tfm_experiment \
  --models tabicl \
  --split-mode leave_matrix_out_support \
  --tabicl-model-path /path/to/tabicl-classifier.ckpt \
  --tabicl-batch-size 512 \
  --device cuda
```

Run a smaller smoke test:

```bash
TABPFN_DISABLE_TELEMETRY=1 python experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py \
  --data-root datasets/bouchard_atomic_relational \
  --output-dir experiments/bouchard_atomic_tfm_experiment \
  --models tabpfn \
  --cases symmetric \
  --train-fractions 0.8 \
  --num-runs 1 \
  --tabpfn-model-path /path/to/tabpfn-classifier.ckpt \
  --tabpfn-n-estimators 1 \
  --device cpu
```

## Outputs

Each run writes `data/` and `results/` under the selected `--output-dir`.

Main output files:

- `results/planned_runs.csv`
- `results/per_run_case_results.csv`
- `results/atomic_per_run_case_results.csv`
- `results/per_run_overall_results.csv`
- `results/summary_by_model_train_fraction_relation.csv`
- `results/summary_overall_by_model_train_fraction.csv`
- `results/bouchard_atomic_comparable_summary.csv`
- `results/thresholds.csv`
- `results/findings.md`
- `results/metadata.json`

Archived result CSVs from previous runs are included under [results/atomic](../../results/atomic).
