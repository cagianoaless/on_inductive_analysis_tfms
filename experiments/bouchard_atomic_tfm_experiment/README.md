# Bouchard Atomic TFM Experiment

This runner evaluates `TabPFN` and `TabICL` on the Bouchard atomic binary-relation property setup.

Important design choice:

- the atomic experiment is run **one relation case at a time**
- each case is a binary relation over all ordered entity pairs
- the runner then aggregates the per-case predictions into an overall summary by train fraction

## 1. Build The Atomic Dataset

The default generator writes the five main-text cases:

- `symmetric`
- `antisymmetric`
- `transitive`
- `symmetric_transitive`
- `antisymmetric_transitive`

Run:

```bash
cd <repo-root>

python scripts/build_bouchard_atomic_datasets.py \
  --output-root datasets/bouchard_atomic_relational
```

This writes:

- `datasets/bouchard_atomic_relational/atomic_relation_atoms.csv`
- `datasets/bouchard_atomic_relational/relation_types.csv`
- `datasets/bouchard_atomic_relational/metadata.json`
- `datasets/bouchard_atomic_relational/matrices/*.npy`
- `datasets/bouchard_atomic_relational/matrices/*.csv`

To generate multiple independent matrices per case:

```bash
python scripts/build_bouchard_atomic_datasets.py \
  --output-root datasets/bouchard_atomic_relational \
  --matrices-per-case 10
```

In that setting, the triples table includes a `matrix_id` column and the matrix files are written under per-case subdirectories.

To also include the appendix reflexive / irreflexive cases:

```bash
python scripts/build_bouchard_atomic_datasets.py \
  --output-root datasets/bouchard_atomic_relational \
  --include-appendix-cases
```

## 2. Dry Run The Experiment Grid

```bash
cd <repo-root>

python experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py \
  --data-root datasets/bouchard_atomic_relational \
  --output-dir experiments/bouchard_atomic_tfm_experiment \
  --dry-run
```

## 3. Run On GPU

Use the same environment as the family runs:

```bash
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python \
  experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py \
  --data-root datasets/bouchard_atomic_relational \
  --output-dir experiments/bouchard_atomic_tfm_experiment \
  --device cuda
```

That default command runs:

- both `tabpfn` and `tabicl`
- all available atomic cases in `relation_types.csv`
- every available `matrix_id` for each case
- train fractions `0.8, 0.4, 0.2, 0.1`
- `10` random runs per configuration
- split mode `random_rows`
- threshold mode `fixed`

## 4. Useful Variants

Run only `TabICL`:

```bash
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python \
  run_bouchard_atomic_experiment.py \
  --models tabicl \
  --device cuda
```

Run only one case:

```bash
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python \
  run_bouchard_atomic_experiment.py \
  --cases symmetric_transitive \
  --device cuda
```

Run the stronger held-out-entity protocol:

```bash
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python \
  run_bouchard_atomic_experiment.py \
  --split-mode entity_block \
  --query-entity-fraction 0.5 \
  --device cuda
```

`entity_block` works like this for each matrix:

- sample a held-out query subset of entities
- training support rows include every pair with at least one anchor entity
- evaluation uses only `query x query` pairs
- `--train-fractions` controls how much of the support pool is kept for training
- `--validation-fraction` controls how much of the `query x query` pool is reserved for validation

Run many matrices per case with the stronger split:

```bash
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python \
  run_bouchard_atomic_experiment.py \
  --data-root /path/to/bouchard_atomic_relational \
  --split-mode entity_block \
  --query-entity-fraction 0.5 \
  --device cuda
```

Run the strongest cross-matrix transfer protocol:

```bash
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python \
  run_bouchard_atomic_experiment.py \
  --data-root /path/to/bouchard_atomic_relational \
  --split-mode leave_matrix_out_support \
  --device cuda
```

`leave_matrix_out_support` works like this for each relation case:

- choose one `matrix_id` as the held-out matrix
- put **all rows from the other matrices** into training
- sample a support subset from the held-out matrix according to `--train-fractions`
- split the remaining held-out rows into validation and test using `--validation-fraction`
- train with matrix-scoped entity IDs so entities from different matrices do not collide

This protocol requires at least `2` matrices per case, so generate the dataset with:

```bash
python build_bouchard_atomic_datasets.py --matrices-per-case 10
```

Tune thresholds on validation F1 instead of keeping `0.5`:

```bash
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python \
  run_bouchard_atomic_experiment.py \
  --threshold-mode tune_global_f1 \
  --device cuda
```

Smaller smoke run:

```bash
TABPFN_DISABLE_TELEMETRY=1 conda run --no-capture-output -n tfms-a python \
  run_bouchard_atomic_experiment.py \
  --models tabpfn \
  --cases symmetric \
  --train-fractions 0.8 \
  --num-runs 1 \
  --tabpfn-n-estimators 1 \
  --device cpu
```

## 5. Outputs

The runner writes results under:

```text
bouchard_atomic_tfm_experiment/
  data/
  results/
```

Main result files:

- `results/per_run_case_results.csv`
- `results/atomic_per_run_case_results.csv`
- `results/per_run_overall_results.csv`
- `results/summary_by_model_train_fraction_relation.csv`
- `results/summary_overall_by_model_train_fraction.csv`
- `results/bouchard_atomic_comparable_summary.csv`
- `results/thresholds.csv`
- `results/findings.md`
- `results/metadata.json`

`summary_by_model_train_fraction_relation.csv` is the main relation-level report.

`bouchard_atomic_comparable_summary.csv` is the quickest overall comparison across train fractions.

When `--split-mode entity_block` is used, `atomic_per_run_case_results.csv` also includes:

- `matrix_id`
- `split_mode`
- `entity_count`
- `anchor_entity_count`
- `query_entity_count`
- `support_pool_rows`
- `evaluation_pool_rows`

When `--split-mode leave_matrix_out_support` is used, the same file also records:

- `transfer_matrix_count`
- `transfer_train_rows`
- `heldout_support_rows`
