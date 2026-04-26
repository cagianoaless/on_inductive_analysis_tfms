# Bouchard TFM Relational Experiments

This repository contains a compact implementation of Bouchard-style relational benchmarks for tabular foundation models. The experiments test whether `TabPFN` and `TabICL` can exploit relational structure when facts are represented as tabular in-context examples.

The repository covers two benchmark families:

- **Atomic binary-relation matrices**, where relation properties such as symmetry, antisymmetry, and transitivity are controlled by construction.
- **Synthetic kinship triples**, where core family relations generate derived relations such as `cousin`, `grandfather`, `uncle`, and `wife`.

The main empirical conclusion is conservative: TFMs learn useful relational signal in some regimes, especially transitive atomic relations, but the evidence does not support a strong claim of general symbolic rule induction.

## Repository Layout

```text
.
├── scripts/
│   ├── build_bouchard_atomic_datasets.py
│   ├── build_bouchard_family_datasets.py
│   └── build_bouchard_family_dataset_groups.py
├── experiments/
│   ├── bouchard_atomic_tfm_experiment/
│   │   ├── run_bouchard_atomic_experiment.py
│   │   └── README.md
│   └── bouchard_family_tfm_experiment/
│       ├── run_bouchard_family_experiment.py
│       ├── run_bouchard_family_overnight.py
│       ├── README.md
│       └── faithfulness_report.md
├── datasets/
│   ├── bouchard_family_relational/
│   └── docs/
├── results/
│   ├── atomic/
│   └── family/
├── reports/
├── requirements.txt
└── .gitignore
```

## Included Data and Results

The repository includes the generated relational kinship dataset:

```text
datasets/bouchard_family_relational/
```

This dataset contains:

- `5` disjoint families;
- `115` people;
- `17` kinship relations;
- `44,965` within-family candidate relation atoms;
- closed-world binary labels for every `(relation, source_person_id, target_person_id)` candidate.

The atomic raw dataset is not included because it is deterministic and can be regenerated from `scripts/build_bouchard_atomic_datasets.py`. The result CSVs from the atomic experiments are included under:

```text
results/atomic/
```

The result CSVs from the family experiments are included under:

```text
results/family/
```

## Installation

Create an environment with Python 3.10+ or 3.11+ and install the Python dependencies:

```bash
pip install -r requirements.txt
```

For GPU execution, ensure that the installed `torch` build matches your CUDA runtime before running with `--device cuda`.

The experiment runners expect local `TabPFN` and `TabICL` classifier checkpoints. Checkpoints are intentionally not part of the visible repository files. Pass their paths explicitly with:

```text
--tabpfn-model-path /path/to/tabpfn-classifier.ckpt
--tabicl-model-path /path/to/tabicl-classifier.ckpt
```

## Atomic Dataset Generation

Generate the five default atomic relation cases with ten independent matrices per case:

```bash
python scripts/build_bouchard_atomic_datasets.py \
  --output-root datasets/bouchard_atomic_relational \
  --matrices-per-case 10
```

The default atomic cases are:

- `symmetric`
- `antisymmetric`
- `transitive`
- `symmetric_transitive`
- `antisymmetric_transitive`

Each matrix has `50` entities and therefore `2,500` ordered-pair atoms. Matrix entries are internally represented as `+1` or `-1`, then exported as binary labels.

## Atomic Experiments

Dry-run the atomic experiment grid:

```bash
python experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py \
  --data-root datasets/bouchard_atomic_relational \
  --output-dir experiments/bouchard_atomic_tfm_experiment \
  --split-mode leave_matrix_out_support \
  --dry-run
```

Run the strongest atomic transfer protocol:

```bash
TABPFN_DISABLE_TELEMETRY=1 python experiments/bouchard_atomic_tfm_experiment/run_bouchard_atomic_experiment.py \
  --data-root datasets/bouchard_atomic_relational \
  --output-dir experiments/bouchard_atomic_tfm_experiment \
  --split-mode leave_matrix_out_support \
  --tabpfn-model-path /path/to/tabpfn-classifier.ckpt \
  --tabicl-model-path /path/to/tabicl-classifier.ckpt \
  --device cuda
```

The strongest protocol is `leave_matrix_out_support`. For each relation family, it holds out one matrix, trains on the other matrices plus a support subset from the held-out matrix, and evaluates on the remaining held-out rows.

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

## Family Dataset Generation

The relational family dataset is already included. To regenerate it:

```bash
python scripts/build_bouchard_family_datasets.py \
  --output-root datasets
```

The generator creates one normalized relational package and one flat export. The experiments in this repository use the relational triples under:

```text
datasets/bouchard_family_relational/
```

## Family Experiments

Dry-run the family experiment schedule:

```bash
python experiments/bouchard_family_tfm_experiment/run_bouchard_family_experiment.py \
  --data-root datasets/bouchard_family_relational \
  --output-dir experiments/bouchard_family_tfm_experiment \
  --include-family-zero \
  --dry-run
```

Run the full family suite:

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

To reproduce the tuned-threshold summaries, add:

```text
--threshold-mode tune_per_relation_f1
```

## Main Result Files

Atomic:

- `results/atomic/atomic_per_run_case_results.csv`: within-matrix atomic experiment.
- `results/atomic/atomic_per_run_case_matrix_out.csv`: leave-matrix-out atomic transfer experiment.

Family:

- `results/family/bouchard_family_comparable_summary.csv`: fixed-threshold overall family summary.
- `results/family/bouchard_family_comparable_summary_tune_per_f1.csv`: tuned-threshold overall family summary.
- `results/family/summary_by_model_split_p_relation.csv`: fixed-threshold relation-level family summary.
- `results/family/summary_by_model_split_p_relation_tune_per_f1.csv`: tuned-threshold relation-level family summary.

## Reports

The main interpretation files are:

- `reports/bouchard_atomic_tfm_matrix_out_results_report.md`
- `reports/bouchard_atomic_tfm_results_report.md`
- `reports/bouchard_family_tfm_results_report.md`
- `reports/bouchard_materials_how_to_check_them.md`