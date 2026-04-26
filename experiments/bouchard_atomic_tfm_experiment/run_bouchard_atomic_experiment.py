#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODELS = ("tabpfn", "tabicl")
SPLIT_CHOICES = ("random_rows", "entity_block", "leave_matrix_out_support")


@dataclass(frozen=True)
class DatasetBundle:
    atoms: pd.DataFrame
    relation_types: pd.DataFrame
    relation_names: list[str]
    entity_categories: list[str]
    scoped_entity_categories: list[str]
    relation_lookup: dict[str, dict[str, Any]]
    matrix_ids_by_relation: dict[str, list[str]]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    workspace_root = root.parent
    external_root = workspace_root.parent

    parser = argparse.ArgumentParser(
        description=(
            "Run the Bouchard atomic binary-relation property experiment with TabPFN-2.5 "
            "and TabICLv2 on the generated synthetic atomic dataset."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=workspace_root / "datasets" / "bouchard_atomic_relational",
        help="Directory containing atomic_relation_atoms.csv and relation_types.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root,
        help="Root directory where data/ and results/ will be written.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="tabpfn,tabicl",
        help="Comma-separated model list: tabpfn, tabicl.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="Optional comma-separated subset of atomic relation names. Defaults to all cases in relation_types.csv.",
    )
    parser.add_argument(
        "--train-fractions",
        type=str,
        default="0.8,0.4,0.2,0.1",
        help=(
            "Comma-separated train fractions used per atomic relation. Under entity_block mode, this is the "
            "fraction of support rows kept for training."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help=(
            "Validation fraction used per atomic relation. Under random_rows it is a row fraction. Under "
            "entity_block it is the fraction of query-query rows reserved for validation."
        ),
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="random_rows",
        choices=list(SPLIT_CHOICES),
        help=(
            "Split protocol: random row holdout, held-out query-query entity block evaluation within one matrix, "
            "or leave-one-matrix-out transfer with support rows from the held-out matrix."
        ),
    )
    parser.add_argument(
        "--query-entity-fraction",
        type=float,
        default=0.5,
        help="Under entity_block mode, fraction of entities assigned to the held-out query block.",
    )
    parser.add_argument("--num-runs", type=int, default=10, help="Number of random samplings per configuration.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Threshold used in fixed-threshold mode and as fallback in tuning modes.",
    )
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="fixed",
        choices=["fixed", "tune_global_f1", "tune_per_relation_f1"],
        help="How to convert probabilities into hard labels using the validation split.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap for the training split.",
    )
    parser.add_argument(
        "--max-valid-rows",
        type=int,
        default=None,
        help="Optional cap for the validation split.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional cap for the test split.",
    )
    parser.add_argument(
        "--tabpfn-model-path",
        type=Path,
        default=external_root / "binary_gbdt_vs_tabpfn" / "tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
    )
    parser.add_argument("--tabpfn-n-estimators", type=int, default=4)
    parser.add_argument(
        "--tabpfn-fit-mode",
        type=str,
        default="fit_preprocessors",
        choices=["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
    )
    parser.add_argument(
        "--tabpfn-memory-saving-mode",
        type=str,
        default="auto",
        help="Forwarded to TabPFNClassifier(memory_saving_mode=...).",
    )
    parser.add_argument(
        "--tabpfn-inference-precision",
        type=str,
        default="auto",
        help="Forwarded to TabPFNClassifier(inference_precision=...).",
    )
    parser.add_argument(
        "--tabicl-model-path",
        type=Path,
        default=external_root / "TabPFNvsTabICL" / "tabicl-classifier-v2-20260212.ckpt",
    )
    parser.add_argument("--tabicl-n-estimators", type=int, default=4)
    parser.add_argument("--tabicl-batch-size", type=int, default=8)
    parser.add_argument(
        "--tabicl-use-amp",
        type=str,
        default="auto",
        help="Forwarded to TabICLClassifier(use_amp=...). Accepts auto/true/false.",
    )
    parser.add_argument(
        "--tabicl-use-fa3",
        type=str,
        default="auto",
        help="Forwarded to TabICLClassifier(use_fa3=...). Accepts auto/true/false.",
    )
    parser.add_argument(
        "--tabicl-offload-mode",
        type=str,
        default="auto",
        help="Forwarded to TabICLClassifier(offload_mode=...).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the full experiment grid, validate inputs, and print the schedule without fitting models.",
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_bool_or_auto(text: str) -> bool | str:
    lowered = text.strip().lower()
    if lowered == "auto":
        return "auto"
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Expected one of auto/true/false, got {text!r}.")


def validate_args(args: argparse.Namespace) -> None:
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be positive.")
    if not 0.0 <= args.decision_threshold <= 1.0:
        raise ValueError("--decision-threshold must be in [0, 1].")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1).")
    if not 0.0 < args.query_entity_fraction < 1.0:
        raise ValueError("--query-entity-fraction must be strictly between 0 and 1.")
    for arg_name in ["max_train_rows", "max_valid_rows", "max_test_rows"]:
        value = getattr(args, arg_name)
        if value is not None and value <= 0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be positive.")

    selected_models = parse_csv_list(args.models)
    unknown_models = sorted(set(selected_models) - set(MODELS))
    if unknown_models:
        raise ValueError(f"Unknown model(s): {unknown_models}. Valid options are {list(MODELS)}.")
    if not selected_models:
        raise ValueError("No models selected.")

    train_fractions = parse_float_list(args.train_fractions)
    if not train_fractions:
        raise ValueError("At least one --train-fractions entry is required.")
    for value in train_fractions:
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"Each train fraction must be strictly between 0 and 1, got {value}.")
        if args.split_mode == "random_rows" and value + args.validation_fraction >= 1.0:
            raise ValueError(
                f"train_fraction + validation_fraction must leave a non-empty test split; got {value} + "
                f"{args.validation_fraction}."
            )

    required_paths = [args.data_root / "atomic_relation_atoms.csv", args.data_root / "relation_types.csv"]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")

    if not args.dry_run:
        if "tabpfn" in selected_models and not args.tabpfn_model_path.exists():
            raise FileNotFoundError(f"TabPFN checkpoint not found: {args.tabpfn_model_path}")
        if "tabicl" in selected_models and not args.tabicl_model_path.exists():
            raise FileNotFoundError(f"TabICL checkpoint not found: {args.tabicl_model_path}")


def load_dataset_bundle(data_root: Path) -> DatasetBundle:
    atoms = pd.read_csv(data_root / "atomic_relation_atoms.csv")
    if "matrix_id" not in atoms.columns:
        atoms["matrix_id"] = "matrix_001"
    atoms["matrix_id"] = atoms["matrix_id"].astype(str)
    relation_types = pd.read_csv(data_root / "relation_types.csv")
    relation_types = relation_types.sort_values(["wave", "relation_id"]).reset_index(drop=True)

    relation_names = relation_types["relation_name"].astype(str).tolist()
    entity_categories = sorted(
        set(atoms["source_entity_id"].astype(str).tolist()) | set(atoms["target_entity_id"].astype(str).tolist())
    )
    scoped_entity_categories = sorted(
        set((atoms["matrix_id"] + "::" + atoms["source_entity_id"].astype(str)).tolist())
        | set((atoms["matrix_id"] + "::" + atoms["target_entity_id"].astype(str)).tolist())
    )
    relation_lookup = {
        str(row["relation_name"]): dict(row)
        for row in relation_types.to_dict(orient="records")
    }
    matrix_ids_by_relation = {
        relation_name: sorted(
            atoms.loc[atoms["relation"].astype(str) == relation_name, "matrix_id"].astype(str).unique().tolist()
        )
        for relation_name in relation_names
    }

    return DatasetBundle(
        atoms=atoms,
        relation_types=relation_types,
        relation_names=relation_names,
        entity_categories=entity_categories,
        scoped_entity_categories=scoped_entity_categories,
        relation_lookup=relation_lookup,
        matrix_ids_by_relation=matrix_ids_by_relation,
    )


def make_feature_frame(
    df: pd.DataFrame,
    *,
    entity_categories: list[str],
    scope_entities_by_matrix: bool = False,
) -> pd.DataFrame:
    entity_dtype = pd.CategoricalDtype(categories=entity_categories, ordered=False)
    source_values = df["source_entity_id"].astype(str)
    target_values = df["target_entity_id"].astype(str)
    if scope_entities_by_matrix:
        matrix_values = df["matrix_id"].astype(str)
        source_values = matrix_values + "::" + source_values
        target_values = matrix_values + "::" + target_values
    return pd.DataFrame(
        {
            "source_entity_id": pd.Series(source_values, dtype=entity_dtype),
            "target_entity_id": pd.Series(target_values, dtype=entity_dtype),
        }
    )


def encode_for_tabpfn(X: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            X["source_entity_id"].cat.codes.to_numpy(dtype=np.int64),
            X["target_entity_id"].cat.codes.to_numpy(dtype=np.int64),
        ]
    )


def stratified_cap(
    df: pd.DataFrame,
    max_rows: int | None,
    *,
    seed: int,
    stratify_cols: Iterable[str],
) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    groups = list(df.groupby(list(stratify_cols), sort=False))
    total = len(df)
    pieces: list[pd.DataFrame] = []
    remaining_budget = max_rows
    remaining_groups = len(groups)

    for _, group_df in groups:
        remaining_groups -= 1
        ideal = max_rows * (len(group_df) / total)
        take = int(round(ideal))
        take = max(1, min(len(group_df), take))
        max_allowed = remaining_budget - remaining_groups
        take = min(take, max_allowed)
        if take <= 0:
            continue
        choice = rng.choice(group_df.index.to_numpy(), size=take, replace=False)
        pieces.append(group_df.loc[np.sort(choice)])
        remaining_budget -= take

    sampled = pd.concat(pieces, axis=0).sample(frac=1.0, random_state=seed)
    return sampled.reset_index(drop=True)


def uniform_partition(
    df: pd.DataFrame,
    *,
    train_fraction: float,
    valid_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_fraction < 0.0 or valid_fraction < 0.0 or train_fraction + valid_fraction > 1.0:
        raise ValueError("Invalid train/valid fractions.")

    n_rows = len(df)
    if n_rows == 0:
        return df.copy(), df.copy(), df.copy()

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    n_train = int(round(train_fraction * n_rows))
    n_valid = int(round(valid_fraction * n_rows))
    n_train = min(max(n_train, 0), n_rows)
    n_valid = min(max(n_valid, 0), n_rows - n_train)
    if n_train + n_valid > n_rows:
        n_valid = max(0, n_rows - n_train)

    train_idx = order[:n_train]
    valid_idx = order[n_train : n_train + n_valid]
    test_idx = order[n_train + n_valid :]

    return (
        df.iloc[np.sort(train_idx)].reset_index(drop=True),
        df.iloc[np.sort(valid_idx)].reset_index(drop=True),
        df.iloc[np.sort(test_idx)].reset_index(drop=True),
    )


def sample_fraction(df: pd.DataFrame, *, fraction: float, seed: int) -> pd.DataFrame:
    if len(df) == 0 or fraction <= 0.0:
        return df.iloc[0:0].copy().reset_index(drop=True)
    if fraction >= 1.0:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n_take = int(round(fraction * len(df)))
    n_take = min(max(n_take, 1), len(df))
    selected = rng.choice(df.index.to_numpy(), size=n_take, replace=False)
    return df.loc[np.sort(selected)].reset_index(drop=True)


def build_split_frames_for_case(
    relation_df: pd.DataFrame,
    *,
    split_mode: str,
    heldout_matrix_id: str,
    train_fraction: float,
    validation_fraction: float,
    query_entity_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int | float | str]]:
    heldout_df = relation_df.loc[relation_df["matrix_id"].astype(str) == heldout_matrix_id].reset_index(drop=True)
    if len(heldout_df) == 0:
        raise ValueError(f"Held-out matrix {heldout_matrix_id!r} is not present in the provided relation frame.")

    entity_ids = sorted(
        set(heldout_df["source_entity_id"].astype(str).tolist()) | set(heldout_df["target_entity_id"].astype(str).tolist())
    )
    if split_mode == "random_rows":
        train_df, valid_df, test_df = uniform_partition(
            heldout_df,
            train_fraction=train_fraction,
            valid_fraction=validation_fraction,
            seed=seed,
        )
        return train_df, valid_df, test_df, {
            "heldout_matrix_id": heldout_matrix_id,
            "transfer_matrix_count": 0,
            "transfer_train_rows": 0,
            "heldout_support_rows": int(len(train_df)),
            "entity_count": len(entity_ids),
            "anchor_entity_count": len(entity_ids),
            "query_entity_count": 0,
            "support_pool_rows": int(len(heldout_df)),
            "evaluation_pool_rows": int(len(heldout_df)),
        }

    if split_mode == "entity_block":
        if len(entity_ids) < 3:
            raise ValueError("entity_block mode requires at least 3 entities.")

        rng = np.random.default_rng(seed)
        n_query = int(round(query_entity_fraction * len(entity_ids)))
        n_query = min(max(n_query, 2), len(entity_ids) - 1)
        query_entities = set(rng.choice(np.asarray(entity_ids, dtype=object), size=n_query, replace=False).tolist())

        source_in_query = heldout_df["source_entity_id"].astype(str).isin(query_entities)
        target_in_query = heldout_df["target_entity_id"].astype(str).isin(query_entities)
        evaluation_df = heldout_df.loc[source_in_query & target_in_query].reset_index(drop=True)
        support_df = heldout_df.loc[~(source_in_query & target_in_query)].reset_index(drop=True)
        if len(support_df) == 0 or len(evaluation_df) == 0:
            raise ValueError("entity_block mode produced an empty support or evaluation pool.")

        train_df = sample_fraction(support_df, fraction=train_fraction, seed=seed)
        valid_df, _, test_df = uniform_partition(
            evaluation_df,
            train_fraction=validation_fraction,
            valid_fraction=0.0,
            seed=seed + 1,
        )
        return train_df, valid_df, test_df, {
            "heldout_matrix_id": heldout_matrix_id,
            "transfer_matrix_count": 0,
            "transfer_train_rows": 0,
            "heldout_support_rows": int(len(train_df)),
            "entity_count": len(entity_ids),
            "anchor_entity_count": len(entity_ids) - n_query,
            "query_entity_count": n_query,
            "support_pool_rows": int(len(support_df)),
            "evaluation_pool_rows": int(len(evaluation_df)),
        }

    if split_mode != "leave_matrix_out_support":
        raise ValueError(f"Unsupported split mode: {split_mode}")

    transfer_df = relation_df.loc[relation_df["matrix_id"].astype(str) != heldout_matrix_id].reset_index(drop=True)
    if len(transfer_df) == 0:
        raise ValueError("leave_matrix_out_support requires at least one non-held-out matrix.")

    rng = np.random.default_rng(seed)
    support_size = int(round(train_fraction * len(heldout_df)))
    support_size = min(max(support_size, 1), len(heldout_df))
    selected_idx = rng.choice(heldout_df.index.to_numpy(), size=support_size, replace=False)
    heldout_support_df = heldout_df.loc[np.sort(selected_idx)].reset_index(drop=True)
    heldout_eval_df = heldout_df.loc[~heldout_df.index.isin(selected_idx)].reset_index(drop=True)
    if len(heldout_eval_df) == 0:
        raise ValueError("leave_matrix_out_support produced an empty held-out evaluation pool.")

    valid_df, _, test_df = uniform_partition(
        heldout_eval_df,
        train_fraction=validation_fraction,
        valid_fraction=0.0,
        seed=seed + 1,
    )
    train_df = pd.concat([transfer_df, heldout_support_df], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, valid_df, test_df, {
        "heldout_matrix_id": heldout_matrix_id,
        "transfer_matrix_count": int(relation_df["matrix_id"].astype(str).nunique() - 1),
        "transfer_train_rows": int(len(transfer_df)),
        "heldout_support_rows": int(len(heldout_support_df)),
        "entity_count": len(entity_ids),
        "anchor_entity_count": len(entity_ids),
        "query_entity_count": 0,
        "support_pool_rows": int(len(heldout_df)),
        "evaluation_pool_rows": int(len(heldout_eval_df)),
    }


def build_tabpfn_classifier(*, args: argparse.Namespace):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        model_path=str(args.tabpfn_model_path),
        device=args.device,
        n_estimators=args.tabpfn_n_estimators,
        categorical_features_indices=[0, 1],
        ignore_pretraining_limits=True,
        fit_mode=args.tabpfn_fit_mode,
        memory_saving_mode=args.tabpfn_memory_saving_mode,
        inference_precision=args.tabpfn_inference_precision,
        n_preprocessing_jobs=1,
        random_state=args.seed,
    )


def build_tabicl_classifier(*, args: argparse.Namespace):
    from tabicl import TabICLClassifier

    return TabICLClassifier(
        model_path=str(args.tabicl_model_path),
        allow_auto_download=False,
        device=args.device,
        n_estimators=args.tabicl_n_estimators,
        batch_size=args.tabicl_batch_size,
        use_amp=parse_bool_or_auto(args.tabicl_use_amp),
        use_fa3=parse_bool_or_auto(args.tabicl_use_fa3),
        offload_mode=args.tabicl_offload_mode,
        random_state=args.seed,
        verbose=False,
    )


def extract_positive_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    classes = np.asarray(classes)
    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError("Unexpected predict_proba output shape.")
    positive_matches = np.where(classes == 1)[0]
    if positive_matches.size != 1:
        raise ValueError(f"Could not identify the positive class in classes={classes!r}.")
    return proba[:, int(positive_matches[0])].astype(np.float64)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_proba: np.ndarray | None = None,
) -> dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    if positive_proba is None or np.unique(y_true).size < 2:
        metrics["average_precision"] = float("nan")
        metrics["roc_auc"] = float("nan")
        return metrics

    metrics["average_precision"] = float(average_precision_score(y_true, positive_proba))
    metrics["roc_auc"] = float(roc_auc_score(y_true, positive_proba))
    return metrics


def threshold_grid(probabilities: np.ndarray, base_threshold: float) -> np.ndarray:
    quantiles = np.quantile(probabilities, np.linspace(0.0, 1.0, 101))
    coarse = np.linspace(0.0, 1.0, 101)
    grid = np.unique(np.concatenate([quantiles, coarse, np.asarray([base_threshold])]))
    return grid[(grid >= 0.0) & (grid <= 1.0)]


def best_f1_threshold(y_true: np.ndarray, positive_proba: np.ndarray, default_threshold: float) -> float:
    if np.unique(y_true).size < 2:
        return float(default_threshold)

    best_threshold = float(default_threshold)
    best_score = -1.0
    for threshold in threshold_grid(positive_proba, default_threshold):
        y_pred = (positive_proba >= threshold).astype(np.int64)
        score = float(f1_score(y_true, y_pred, zero_division=0))
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def resolve_thresholds(
    *,
    threshold_mode: str,
    base_threshold: float,
    validation_y: np.ndarray,
    validation_proba: np.ndarray,
    relation_name: str,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    threshold_map = {"overall": float(base_threshold), relation_name: float(base_threshold)}
    records: list[dict[str, object]] = []

    if threshold_mode == "fixed" or validation_y.size == 0 or np.unique(validation_y).size < 2:
        for relation, value in threshold_map.items():
            records.append(
                {
                    "relation": relation,
                    "threshold": float(value),
                    "validation_rows": int(len(validation_y)),
                    "validation_positive_rate": float(np.mean(validation_y)) if len(validation_y) else float("nan"),
                }
            )
        return threshold_map, records

    tuned = best_f1_threshold(validation_y, validation_proba, base_threshold)
    threshold_map = {"overall": tuned, relation_name: tuned}
    for relation, value in threshold_map.items():
        records.append(
            {
                "relation": relation,
                "threshold": float(value),
                "validation_rows": int(len(validation_y)),
                "validation_positive_rate": float(np.mean(validation_y)),
            }
        )
    return threshold_map, records


def apply_threshold(positive_proba: np.ndarray, threshold: float) -> np.ndarray:
    return (positive_proba >= float(threshold)).astype(np.int64)


def predict_positive_proba(model: Any, X_query_in: Any) -> np.ndarray:
    return extract_positive_proba(np.asarray(model.predict_proba(X_query_in)), np.asarray(model.classes_))


def fit_and_predict(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
    relation_name: str,
) -> dict[str, Any]:
    if model_name == "tabpfn":
        model = build_tabpfn_classifier(args=args)
        X_train_in = encode_for_tabpfn(X_train)
        X_valid_in = encode_for_tabpfn(X_valid)
        X_test_in = encode_for_tabpfn(X_test)
    elif model_name == "tabicl":
        model = build_tabicl_classifier(args=args)
        X_train_in = X_train
        X_valid_in = X_valid
        X_test_in = X_test
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    fit_start = time.perf_counter()
    model.fit(X_train_in, y_train)
    fit_time_s = time.perf_counter() - fit_start

    pred_start = time.perf_counter()
    valid_proba = predict_positive_proba(model, X_valid_in) if len(X_valid) else np.asarray([], dtype=np.float64)
    test_proba = predict_positive_proba(model, X_test_in)
    predict_time_s = time.perf_counter() - pred_start

    threshold_map, threshold_records = resolve_thresholds(
        threshold_mode=args.threshold_mode,
        base_threshold=args.decision_threshold,
        validation_y=y_valid,
        validation_proba=valid_proba,
        relation_name=relation_name,
    )
    y_pred = apply_threshold(test_proba, threshold_map[relation_name])

    return {
        "y_pred": y_pred,
        "positive_proba": test_proba,
        "fit_time_s": float(fit_time_s),
        "predict_time_s": float(predict_time_s),
        "threshold_records": threshold_records,
    }


def build_run_plan(
    *,
    matrix_ids_by_relation: dict[str, list[str]],
    relation_names: list[str],
    train_fractions: list[float],
    num_runs: int,
    seed: int,
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for relation_idx, relation_name in enumerate(relation_names):
        matrix_ids = matrix_ids_by_relation[relation_name]
        for fraction_idx, train_fraction in enumerate(train_fractions):
            for matrix_idx, matrix_id in enumerate(matrix_ids):
                for run_idx in range(num_runs):
                    split_seed = (
                        seed
                        + 1_000_000 * relation_idx
                        + 10_000 * matrix_idx
                        + 1_000 * fraction_idx
                        + run_idx
                    )
                    plan.append(
                        {
                            "relation": relation_name,
                            "matrix_id": matrix_id,
                            "train_fraction": float(train_fraction),
                            "run_idx": int(run_idx),
                            "split_seed": int(split_seed),
                        }
                    )
    return plan


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_findings_md(
    *,
    bundle: DatasetBundle,
    selected_relation_names: list[str],
    summary_relation_df: pd.DataFrame,
    summary_overall_df: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append("# Findings")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Atomic relation cases: `{len(selected_relation_names)}`")
    lines.append(
        f"- Matrices per case: `{sorted({len(bundle.matrix_ids_by_relation[name]) for name in selected_relation_names})}`"
    )
    lines.append(f"- Entities per relation: `{len(bundle.entity_categories)}`")
    lines.append(f"- Total atoms available: `{len(bundle.atoms):,}`")
    lines.append(f"- Number of runs per configuration: `{args.num_runs}`")
    lines.append(f"- Split mode: `{args.split_mode}`")
    lines.append(f"- Threshold mode: `{args.threshold_mode}`")
    lines.append(f"- Device: `{args.device}`")
    lines.append("")
    lines.append("## Overall Mean Metrics By Train Fraction")
    lines.append("")
    lines.append("| Split Mode | Train Fraction | Model | Avg Precision Mean | Avg Precision Std | F1 Mean | Accuracy Mean |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for _, row in summary_overall_df.sort_values(
        ["split_mode", "train_fraction", "average_precision_mean"],
        ascending=[True, False, False],
    ).iterrows():
        lines.append(
            f"| {row['split_mode']} | {row['train_fraction']:.1f} | {row['model']} | "
            f"{row['average_precision_mean']:.4f} | {row['average_precision_std']:.4f} | "
            f"{row['f1_mean']:.4f} | {row['accuracy_mean']:.4f} |"
        )
    lines.append("")
    lines.append("## Relation-Level Mean AP")
    lines.append("")
    lines.append("| Split Mode | Relation | Train Fraction | Model | Avg Precision Mean | F1 Mean |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: |")
    for _, row in summary_relation_df.sort_values(
        ["split_mode", "relation", "train_fraction", "average_precision_mean"],
        ascending=[True, True, False, False],
    ).iterrows():
        lines.append(
            f"| {row['split_mode']} | {row['relation']} | {row['train_fraction']:.1f} | {row['model']} | "
            f"{row['average_precision_mean']:.4f} | {row['f1_mean']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def summarize_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary_df = (
        df.groupby(group_cols, as_index=False)
        .agg(
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            accuracy_mean=("accuracy", "mean"),
            fit_time_mean_s=("fit_time_s", "mean"),
            predict_time_mean_s=("predict_time_s", "mean"),
            runs=("run_idx", "nunique"),
        )
    )
    return summary_df.fillna(0.0)


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_root = args.output_dir.resolve()
    data_dir = output_root / "data"
    results_dir = output_root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset_bundle(args.data_root.resolve())
    selected_relation_names = parse_csv_list(args.cases) if args.cases else bundle.relation_names
    unknown_cases = sorted(set(selected_relation_names) - set(bundle.relation_names))
    if unknown_cases:
        raise ValueError(f"Unknown atomic relation case(s): {unknown_cases}. Valid options: {bundle.relation_names}")
    selected_relation_names = [name for name in bundle.relation_names if name in set(selected_relation_names)]
    selected_relation_types = bundle.relation_types.loc[
        bundle.relation_types["relation_name"].astype(str).isin(selected_relation_names)
    ].reset_index(drop=True)
    if args.split_mode == "leave_matrix_out_support":
        invalid_relations = [
            name for name in selected_relation_names if len(bundle.matrix_ids_by_relation[name]) < 2
        ]
        if invalid_relations:
            raise ValueError(
                "leave_matrix_out_support requires at least 2 matrices per relation. "
                f"Invalid relation(s): {invalid_relations}"
            )

    train_fractions = parse_float_list(args.train_fractions)
    run_plan = build_run_plan(
        matrix_ids_by_relation=bundle.matrix_ids_by_relation,
        relation_names=selected_relation_names,
        train_fractions=train_fractions,
        num_runs=args.num_runs,
        seed=args.seed,
    )
    pd.DataFrame(run_plan).to_csv(results_dir / "planned_runs.csv", index=False)

    dataset_summary = {
        "dataset_name": "bouchard_atomic_relational",
        "data_root": str(args.data_root.resolve()),
        "relation_names": selected_relation_names,
        "split_mode": args.split_mode,
        "query_entity_fraction": float(args.query_entity_fraction),
        "entity_feature_mode": "matrix_scoped"
        if args.split_mode == "leave_matrix_out_support"
        else "plain_entity_ids",
        "entity_categories": bundle.entity_categories,
        "atom_count_selected_cases": int(
            bundle.atoms.loc[bundle.atoms["relation"].isin(selected_relation_names)].shape[0]
        ),
        "matrix_ids_by_relation": {
            relation_name: bundle.matrix_ids_by_relation[relation_name]
            for relation_name in selected_relation_names
        },
        "atoms_per_relation": {
            relation: int(count)
            for relation, count in bundle.atoms.loc[bundle.atoms["relation"].isin(selected_relation_names)]
            .groupby("relation", sort=False)
            .size()
            .items()
        },
        "positive_counts_by_relation": {
            relation: int(count)
            for relation, count in bundle.atoms.loc[
                (bundle.atoms["relation"].isin(selected_relation_names)) & (bundle.atoms["label"] == 1)
            ]
            .groupby("relation", sort=False)
            .size()
            .items()
        },
    }
    write_json(data_dir / "dataset_summary.json", dataset_summary)

    if args.dry_run:
        print(f"data_root: {args.data_root.resolve()}")
        print(f"atomic_cases: {len(selected_relation_names)}")
        print(f"entities: {len(bundle.entity_categories)}")
        print(f"split_mode: {args.split_mode}")
        print(f"planned_configurations: {len(run_plan)}")
        print(selected_relation_types.to_string(index=False))
        print(pd.DataFrame(run_plan).head(20).to_string(index=False))
        return

    selected_models = parse_csv_list(args.models)
    use_matrix_scoped_entities = args.split_mode == "leave_matrix_out_support"
    case_result_records: list[dict[str, object]] = []
    threshold_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []

    for config_idx, config in enumerate(run_plan):
        relation_name = str(config["relation"])
        matrix_id = str(config["matrix_id"])
        train_fraction = float(config["train_fraction"])
        run_idx = int(config["run_idx"])
        split_seed = int(config["split_seed"])
        relation_info = bundle.relation_lookup[relation_name]

        relation_df = bundle.atoms.loc[bundle.atoms["relation"] == relation_name].reset_index(drop=True)
        train_df, valid_df, test_df, split_metadata = build_split_frames_for_case(
            relation_df,
            split_mode=args.split_mode,
            heldout_matrix_id=matrix_id,
            train_fraction=train_fraction,
            validation_fraction=args.validation_fraction,
            query_entity_fraction=args.query_entity_fraction,
            seed=split_seed,
        )
        train_df = stratified_cap(train_df, args.max_train_rows, seed=split_seed, stratify_cols=["label"])
        valid_df = stratified_cap(valid_df, args.max_valid_rows, seed=split_seed + 1, stratify_cols=["label"])
        test_df = stratified_cap(test_df, args.max_test_rows, seed=split_seed + 2, stratify_cols=["label"])

        y_train = train_df["label"].to_numpy(dtype=np.int64)
        y_valid = valid_df["label"].to_numpy(dtype=np.int64)
        y_test = test_df["label"].to_numpy(dtype=np.int64)
        if np.unique(y_train).size < 2:
            raise ValueError(
                f"Training split collapsed to a single class for relation={relation_name}, "
                f"train_fraction={train_fraction}, run={run_idx}."
            )

        entity_categories = bundle.scoped_entity_categories if use_matrix_scoped_entities else bundle.entity_categories
        X_train = make_feature_frame(
            train_df,
            entity_categories=entity_categories,
            scope_entities_by_matrix=use_matrix_scoped_entities,
        )
        X_valid = make_feature_frame(
            valid_df,
            entity_categories=entity_categories,
            scope_entities_by_matrix=use_matrix_scoped_entities,
        )
        X_test = make_feature_frame(
            test_df,
            entity_categories=entity_categories,
            scope_entities_by_matrix=use_matrix_scoped_entities,
        )

        print(
            f"[atomic] cfg={config_idx + 1}/{len(run_plan)} relation={relation_name} matrix={matrix_id} "
            f"split={args.split_mode} "
            f"train_fraction={train_fraction:.1f} run={run_idx} "
            f"train={len(train_df)} valid={len(valid_df)} test={len(test_df)}",
            flush=True,
        )

        for model_name in selected_models:
            outputs = fit_and_predict(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                args=args,
                relation_name=relation_name,
            )

            relation_metrics = evaluate_predictions(
                y_true=y_test,
                y_pred=outputs["y_pred"],
                positive_proba=outputs["positive_proba"],
            )
            case_result_records.append(
                {
                    "model": model_name,
                    "split_mode": args.split_mode,
                    "train_fraction": train_fraction,
                    "run_idx": run_idx,
                    "relation": relation_name,
                    "relation_id": relation_info["relation_id"],
                    "matrix_id": matrix_id,
                    "wave": int(relation_info["wave"]),
                    "reflexivity": relation_info["reflexivity"],
                    "symmetry": relation_info["symmetry"],
                    "transitive": int(relation_info["transitive"]),
                    "entity_count": int(split_metadata["entity_count"]),
                    "anchor_entity_count": int(split_metadata["anchor_entity_count"]),
                    "query_entity_count": int(split_metadata["query_entity_count"]),
                    "transfer_matrix_count": int(split_metadata["transfer_matrix_count"]),
                    "transfer_train_rows": int(split_metadata["transfer_train_rows"]),
                    "heldout_support_rows": int(split_metadata["heldout_support_rows"]),
                    "support_pool_rows": int(split_metadata["support_pool_rows"]),
                    "evaluation_pool_rows": int(split_metadata["evaluation_pool_rows"]),
                    "train_rows": int(len(train_df)),
                    "valid_rows": int(len(valid_df)),
                    "test_rows": int(len(test_df)),
                    "fit_time_s": outputs["fit_time_s"],
                    "predict_time_s": outputs["predict_time_s"],
                    **relation_metrics,
                }
            )

            for y_true_value, y_pred_value, proba_value in zip(y_test, outputs["y_pred"], outputs["positive_proba"]):
                prediction_records.append(
                    {
                        "model": model_name,
                        "split_mode": args.split_mode,
                        "train_fraction": train_fraction,
                        "run_idx": run_idx,
                        "relation": relation_name,
                        "matrix_id": matrix_id,
                        "y_true": int(y_true_value),
                        "y_pred": int(y_pred_value),
                        "positive_proba": float(proba_value),
                    }
                )

            for threshold_record in outputs["threshold_records"]:
                threshold_records.append(
                    {
                        "model": model_name,
                        "split_mode": args.split_mode,
                        "train_fraction": train_fraction,
                        "run_idx": run_idx,
                        "case_name": relation_name,
                        "matrix_id": matrix_id,
                        "relation_id": relation_info["relation_id"],
                        "wave": int(relation_info["wave"]),
                        **threshold_record,
                    }
                )

    case_results_df = pd.DataFrame(case_result_records)
    predictions_df = pd.DataFrame(prediction_records)
    thresholds_df = pd.DataFrame(threshold_records)

    overall_records: list[dict[str, object]] = []
    if not predictions_df.empty:
        case_size_lookup = (
            case_results_df.groupby(["model", "split_mode", "train_fraction", "run_idx"], as_index=False)
            .agg(
                transfer_matrix_count=("transfer_matrix_count", "sum"),
                transfer_train_rows=("transfer_train_rows", "sum"),
                heldout_support_rows=("heldout_support_rows", "sum"),
                support_pool_rows=("support_pool_rows", "sum"),
                evaluation_pool_rows=("evaluation_pool_rows", "sum"),
                train_rows=("train_rows", "sum"),
                valid_rows=("valid_rows", "sum"),
                test_rows=("test_rows", "sum"),
                fit_time_s=("fit_time_s", "sum"),
                predict_time_s=("predict_time_s", "sum"),
            )
        )
        for keys, group_df in predictions_df.groupby(["model", "split_mode", "train_fraction", "run_idx"], as_index=False):
            model_name, split_mode, train_fraction, run_idx = keys
            metrics = evaluate_predictions(
                y_true=group_df["y_true"].to_numpy(dtype=np.int64),
                y_pred=group_df["y_pred"].to_numpy(dtype=np.int64),
                positive_proba=group_df["positive_proba"].to_numpy(dtype=np.float64),
            )
            size_row = case_size_lookup.loc[
                (case_size_lookup["model"] == model_name)
                & (case_size_lookup["split_mode"] == split_mode)
                & (case_size_lookup["train_fraction"] == train_fraction)
                & (case_size_lookup["run_idx"] == run_idx)
            ].iloc[0]
            overall_records.append(
                {
                    "model": model_name,
                    "split_mode": split_mode,
                    "train_fraction": float(train_fraction),
                    "run_idx": int(run_idx),
                    "relation": "overall",
                    "relation_id": "overall",
                    "matrix_id": "overall",
                    "wave": 0,
                    "reflexivity": "mixed",
                    "symmetry": "mixed",
                    "transitive": -1,
                    "entity_count": int(len(bundle.entity_categories)),
                    "anchor_entity_count": 0,
                    "query_entity_count": 0,
                    "transfer_matrix_count": int(size_row["transfer_matrix_count"]),
                    "transfer_train_rows": int(size_row["transfer_train_rows"]),
                    "heldout_support_rows": int(size_row["heldout_support_rows"]),
                    "support_pool_rows": int(size_row["support_pool_rows"]),
                    "evaluation_pool_rows": int(size_row["evaluation_pool_rows"]),
                    "train_rows": int(size_row["train_rows"]),
                    "valid_rows": int(size_row["valid_rows"]),
                    "test_rows": int(size_row["test_rows"]),
                    "fit_time_s": float(size_row["fit_time_s"]),
                    "predict_time_s": float(size_row["predict_time_s"]),
                    **metrics,
                }
            )

    overall_results_df = pd.DataFrame(overall_records)
    results_df = pd.concat([overall_results_df, case_results_df], axis=0, ignore_index=True)

    results_df.to_csv(results_dir / "per_run_results.csv", index=False)
    case_results_df.to_csv(results_dir / "per_run_case_results.csv", index=False)
    case_results_df.to_csv(results_dir / "atomic_per_run_case_results.csv", index=False)
    overall_results_df.to_csv(results_dir / "per_run_overall_results.csv", index=False)
    thresholds_df.to_csv(results_dir / "thresholds.csv", index=False)
    predictions_df.to_csv(results_dir / "raw_predictions.csv", index=False)

    summary_relation_df = summarize_metrics(
        case_results_df,
        group_cols=[
            "model",
            "split_mode",
            "train_fraction",
            "relation",
            "relation_id",
            "wave",
            "reflexivity",
            "symmetry",
            "transitive",
        ],
    )
    summary_relation_df.to_csv(results_dir / "summary_by_model_train_fraction_relation.csv", index=False)

    summary_overall_df = summarize_metrics(
        overall_results_df,
        group_cols=["model", "split_mode", "train_fraction"],
    )
    summary_overall_df.to_csv(results_dir / "summary_overall_by_model_train_fraction.csv", index=False)

    comparable_df = summary_overall_df.sort_values(
        ["split_mode", "train_fraction", "average_precision_mean"],
        ascending=[True, False, False],
    )
    comparable_df.to_csv(results_dir / "bouchard_atomic_comparable_summary.csv", index=False)

    findings_md = build_findings_md(
        bundle=bundle,
        selected_relation_names=selected_relation_names,
        summary_relation_df=summary_relation_df,
        summary_overall_df=summary_overall_df,
        args=args,
    )
    (results_dir / "findings.md").write_text(findings_md, encoding="utf-8")

    metadata = {
        "launcher": "run_bouchard_atomic_experiment.py",
        "data_root": str(args.data_root.resolve()),
        "output_dir": str(output_root),
        "models": selected_models,
        "relation_names": selected_relation_names,
        "split_mode": args.split_mode,
        "train_fractions": train_fractions,
        "validation_fraction": float(args.validation_fraction),
        "query_entity_fraction": float(args.query_entity_fraction),
        "num_runs": int(args.num_runs),
        "seed": int(args.seed),
        "device": args.device,
        "threshold_mode": args.threshold_mode,
        "decision_threshold": float(args.decision_threshold),
        "max_train_rows": args.max_train_rows,
        "max_valid_rows": args.max_valid_rows,
        "max_test_rows": args.max_test_rows,
        "dataset_summary": dataset_summary,
        "result_row_count": int(len(results_df)),
        "configuration_count": int(len(run_plan)),
    }
    write_json(results_dir / "metadata.json", metadata)


if __name__ == "__main__":
    main()
