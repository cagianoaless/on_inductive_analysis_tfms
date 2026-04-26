#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
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
SPLIT_CHOICES = ("random", "evidence", "family")


@dataclass(frozen=True)
class DatasetBundle:
    atoms: pd.DataFrame
    relation_types: pd.DataFrame
    family_ids: list[str]
    relation_names: list[str]
    core_relations: set[str]
    derived_relations: set[str]
    person_categories: list[str]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    workspace_root = root.parent
    external_root = workspace_root.parent

    parser = argparse.ArgumentParser(
        description=(
            "Run the Bouchard family / kinship experiment with TabPFN-2.5 and TabICLv2 "
            "on the generated synthetic family dataset."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=workspace_root / "datasets" / "bouchard_family_relational",
        help="Directory containing pair_relation_atoms.csv and relation_types.csv.",
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
        "--splits",
        type=str,
        default="random,evidence,family",
        help="Comma-separated experiment split list: random, evidence, family.",
    )
    parser.add_argument(
        "--p-values",
        type=str,
        default="0.8,0.4,0.2,0.1",
        help="Comma-separated p values used for random/evidence/family splits.",
    )
    parser.add_argument(
        "--include-family-zero",
        action="store_true",
        help="Also run the family split with p=0.0.",
    )
    parser.add_argument(
        "--holdout-family-id",
        type=str,
        default="family_05",
        help="Family used as the held-out family in the family split.",
    )
    parser.add_argument("--num-runs", type=int, default=10, help="Number of random split samplings per configuration.")
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
        "--negative-sampling-ratio",
        type=float,
        default=None,
        help="Optional training-only negative sampling ratio per relation.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap for the training split after negative sampling.",
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
    if args.decision_threshold < 0.0 or args.decision_threshold > 1.0:
        raise ValueError("--decision-threshold must be in [0, 1].")
    if args.negative_sampling_ratio is not None and args.negative_sampling_ratio <= 0:
        raise ValueError("--negative-sampling-ratio must be positive when provided.")
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

    selected_splits = parse_csv_list(args.splits)
    unknown_splits = sorted(set(selected_splits) - set(SPLIT_CHOICES))
    if unknown_splits:
        raise ValueError(f"Unknown split(s): {unknown_splits}. Valid options are {list(SPLIT_CHOICES)}.")
    if not selected_splits:
        raise ValueError("No split regimes selected.")

    p_values = parse_float_list(args.p_values)
    if not p_values:
        raise ValueError("At least one --p-values entry is required.")
    for value in p_values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Each p value must be in [0, 1], got {value}.")
    non_family_zero = [value for value in p_values if value == 0.0]
    if non_family_zero and any(split in {"random", "evidence"} for split in selected_splits):
        raise ValueError("p=0.0 is only meaningful for the family split.")

    required_paths = [args.data_root / "pair_relation_atoms.csv", args.data_root / "relation_types.csv"]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")

    if not args.dry_run:
        if "tabpfn" in selected_models and not args.tabpfn_model_path.exists():
            raise FileNotFoundError(f"TabPFN checkpoint not found: {args.tabpfn_model_path}")
        if "tabicl" in selected_models and not args.tabicl_model_path.exists():
            raise FileNotFoundError(f"TabICL checkpoint not found: {args.tabicl_model_path}")


def load_dataset_bundle(data_root: Path) -> DatasetBundle:
    atoms = pd.read_csv(data_root / "pair_relation_atoms.csv")
    relation_types = pd.read_csv(data_root / "relation_types.csv")
    relation_types = relation_types.sort_values("relation_id").reset_index(drop=True)

    family_ids = sorted(atoms["family_id"].astype(str).unique().tolist())
    relation_names = relation_types["relation_name"].astype(str).tolist()
    core_relations = set(
        relation_types.loc[relation_types["relation_group"] == "core", "relation_name"].astype(str).tolist()
    )
    derived_relations = set(relation_names) - core_relations
    person_categories = sorted(
        set(atoms["source_person_id"].astype(str).tolist()) | set(atoms["target_person_id"].astype(str).tolist())
    )

    return DatasetBundle(
        atoms=atoms,
        relation_types=relation_types,
        family_ids=family_ids,
        relation_names=relation_names,
        core_relations=core_relations,
        derived_relations=derived_relations,
        person_categories=person_categories,
    )


def make_feature_frame(
    df: pd.DataFrame,
    *,
    relation_categories: list[str],
    person_categories: list[str],
) -> pd.DataFrame:
    relation_dtype = pd.CategoricalDtype(categories=relation_categories, ordered=False)
    person_dtype = pd.CategoricalDtype(categories=person_categories, ordered=False)
    return pd.DataFrame(
        {
            "relation": pd.Series(df["relation"].astype(str), dtype=relation_dtype),
            "source_person_id": pd.Series(df["source_person_id"].astype(str), dtype=person_dtype),
            "target_person_id": pd.Series(df["target_person_id"].astype(str), dtype=person_dtype),
        }
    )


def encode_for_tabpfn(X: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            X["relation"].cat.codes.to_numpy(dtype=np.int64),
            X["source_person_id"].cat.codes.to_numpy(dtype=np.int64),
            X["target_person_id"].cat.codes.to_numpy(dtype=np.int64),
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


def sample_training_negatives(
    df: pd.DataFrame,
    *,
    negative_sampling_ratio: float | None,
    seed: int,
) -> pd.DataFrame:
    if negative_sampling_ratio is None:
        return df.reset_index(drop=True)

    sampled_parts: list[pd.DataFrame] = []
    for relation_idx, (_, relation_df) in enumerate(df.groupby("relation", sort=False)):
        positives = relation_df.loc[relation_df["label"] == 1]
        negatives = relation_df.loc[relation_df["label"] == 0]
        max_negatives = int(math.ceil(len(positives) * negative_sampling_ratio))
        if max_negatives > 0 and len(negatives) > max_negatives:
            negatives = negatives.sample(n=max_negatives, random_state=seed + relation_idx)
        sampled_parts.extend([positives, negatives])

    sampled_df = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=seed)
    return sampled_df.reset_index(drop=True)


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


def build_split_frames(
    *,
    bundle: DatasetBundle,
    split_name: str,
    p_value: float,
    holdout_family_id: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atoms = bundle.atoms

    if split_name == "random":
        return uniform_partition(atoms, train_fraction=p_value, valid_fraction=0.1, seed=seed)

    if split_name == "evidence":
        core_df = atoms.loc[atoms["relation"].isin(bundle.core_relations)].reset_index(drop=True)
        derived_df = atoms.loc[atoms["relation"].isin(bundle.derived_relations)].reset_index(drop=True)
        derived_train, derived_valid, derived_test = uniform_partition(
            derived_df,
            train_fraction=p_value,
            valid_fraction=0.1,
            seed=seed,
        )
        train_df = pd.concat([core_df, derived_train], axis=0).sample(frac=1.0, random_state=seed)
        return train_df.reset_index(drop=True), derived_valid, derived_test

    if split_name == "family":
        if holdout_family_id not in bundle.family_ids:
            raise ValueError(
                f"Holdout family {holdout_family_id!r} is not present. Available: {bundle.family_ids}"
            )
        train_full_df = atoms.loc[atoms["family_id"] != holdout_family_id].reset_index(drop=True)
        holdout_core_df = atoms.loc[
            (atoms["family_id"] == holdout_family_id) & (atoms["relation"].isin(bundle.core_relations))
        ].reset_index(drop=True)
        holdout_derived_df = atoms.loc[
            (atoms["family_id"] == holdout_family_id) & (atoms["relation"].isin(bundle.derived_relations))
        ].reset_index(drop=True)
        derived_train, derived_valid, derived_test = uniform_partition(
            holdout_derived_df,
            train_fraction=p_value,
            valid_fraction=0.1,
            seed=seed,
        )
        train_df = pd.concat([train_full_df, holdout_core_df, derived_train], axis=0).sample(
            frac=1.0,
            random_state=seed,
        )
        return train_df.reset_index(drop=True), derived_valid, derived_test

    raise ValueError(f"Unsupported split regime: {split_name}")


def build_tabpfn_classifier(*, args: argparse.Namespace):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        model_path=str(args.tabpfn_model_path),
        device=args.device,
        n_estimators=args.tabpfn_n_estimators,
        categorical_features_indices=[0, 1, 2],
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
    validation_df: pd.DataFrame,
    validation_y: np.ndarray,
    validation_proba: np.ndarray,
    relation_names: list[str],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    threshold_map = {"overall": float(base_threshold), **{relation: float(base_threshold) for relation in relation_names}}
    records: list[dict[str, object]] = []

    if threshold_mode == "fixed" or len(validation_df) == 0 or np.unique(validation_y).size < 2:
        for relation, value in threshold_map.items():
            records.append(
                {
                    "relation": relation,
                    "threshold": float(value),
                    "validation_rows": int(len(validation_df)),
                    "validation_positive_rate": float(np.mean(validation_y)) if len(validation_y) else float("nan"),
                }
            )
        return threshold_map, records

    if threshold_mode == "tune_global_f1":
        tuned = best_f1_threshold(validation_y, validation_proba, base_threshold)
        threshold_map = {"overall": tuned, **{relation: tuned for relation in relation_names}}
    elif threshold_mode == "tune_per_relation_f1":
        threshold_map["overall"] = best_f1_threshold(validation_y, validation_proba, base_threshold)
        relation_values = validation_df["relation"].astype(str).to_numpy()
        for relation in relation_names:
            mask = relation_values == relation
            if mask.any():
                threshold_map[relation] = best_f1_threshold(
                    validation_y[mask],
                    validation_proba[mask],
                    base_threshold,
                )
    else:
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")

    relation_values = validation_df["relation"].astype(str).to_numpy()
    for relation, value in threshold_map.items():
        if relation == "overall":
            y_rel = validation_y
        else:
            y_rel = validation_y[relation_values == relation]
        records.append(
            {
                "relation": relation,
                "threshold": float(value),
                "validation_rows": int(len(y_rel)),
                "validation_positive_rate": float(np.mean(y_rel)) if len(y_rel) else float("nan"),
            }
        )
    return threshold_map, records


def apply_thresholds(
    positive_proba: np.ndarray,
    relations: np.ndarray,
    threshold_map: dict[str, float],
) -> np.ndarray:
    thresholds = np.full_like(positive_proba, fill_value=float(threshold_map["overall"]), dtype=np.float64)
    relation_array = np.asarray(relations, dtype=object)
    for relation, threshold in threshold_map.items():
        if relation == "overall":
            continue
        thresholds[relation_array == relation] = float(threshold)
    return (positive_proba >= thresholds).astype(np.int64)


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
    relation_names: list[str],
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
        validation_df=X_valid,
        validation_y=y_valid,
        validation_proba=valid_proba,
        relation_names=relation_names,
    )
    y_pred = apply_thresholds(
        positive_proba=test_proba,
        relations=X_test["relation"].astype(str).to_numpy(),
        threshold_map=threshold_map,
    )

    return {
        "y_pred": y_pred,
        "positive_proba": test_proba,
        "fit_time_s": float(fit_time_s),
        "predict_time_s": float(predict_time_s),
        "threshold_records": threshold_records,
    }


def build_run_plan(args: argparse.Namespace) -> list[dict[str, object]]:
    selected_splits = parse_csv_list(args.splits)
    p_values = parse_float_list(args.p_values)
    family_p_values = list(p_values)
    if args.include_family_zero and 0.0 not in family_p_values:
        family_p_values.append(0.0)
    family_p_values = sorted(set(family_p_values), reverse=True)

    plan: list[dict[str, object]] = []
    for split_name in selected_splits:
        values = family_p_values if split_name == "family" else p_values
        for p_idx, p_value in enumerate(values):
            if split_name in {"random", "evidence"} and p_value == 0.0:
                continue
            for run_idx in range(args.num_runs):
                split_seed = args.seed + 100_000 * selected_splits.index(split_name) + 1_000 * p_idx + run_idx
                plan.append(
                    {
                        "split_name": split_name,
                        "p_value": float(p_value),
                        "run_idx": int(run_idx),
                        "split_seed": int(split_seed),
                        "holdout_family_id": args.holdout_family_id,
                    }
                )
    return plan


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def relation_positive_counts(bundle: DatasetBundle) -> dict[str, int]:
    positives = bundle.atoms.loc[bundle.atoms["label"] == 1]
    return {
        relation: int(count)
        for relation, count in positives.groupby("relation", sort=False).size().items()
    }


def build_findings_md(
    *,
    bundle: DatasetBundle,
    summary_overall_df: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append("# Findings")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Families: `{len(bundle.family_ids)}`")
    lines.append(f"- Relations: `{len(bundle.relation_names)}`")
    lines.append(f"- Atoms: `{len(bundle.atoms):,}`")
    lines.append(f"- Holdout family for family split: `{args.holdout_family_id}`")
    lines.append(f"- Number of runs per configuration: `{args.num_runs}`")
    lines.append(f"- Threshold mode: `{args.threshold_mode}`")
    lines.append(f"- Device: `{args.device}`")
    lines.append("")
    lines.append("## Overall Mean Metrics By Configuration")
    lines.append("")
    lines.append("| Split | p | Model | Avg Precision Mean | Avg Precision Std | F1 Mean | Accuracy Mean |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for _, row in summary_overall_df.sort_values(["split_name", "p_value", "average_precision_mean"], ascending=[True, False, False]).iterrows():
        lines.append(
            f"| {row['split_name']} | {row['p_value']:.1f} | {row['model']} | "
            f"{row['average_precision_mean']:.4f} | {row['average_precision_std']:.4f} | "
            f"{row['f1_mean']:.4f} | {row['accuracy_mean']:.4f} |"
        )
    lines.append("")
    lines.append("## Best Model Per Configuration")
    lines.append("")
    lines.append("| Split | p | Best Model By Mean AP | Mean AP |")
    lines.append("| --- | ---: | --- | ---: |")
    best_rows = (
        summary_overall_df.sort_values(
            ["split_name", "p_value", "average_precision_mean"],
            ascending=[True, False, False],
        )
        .groupby(["split_name", "p_value"], as_index=False)
        .first()
    )
    for _, row in best_rows.iterrows():
        lines.append(
            f"| {row['split_name']} | {row['p_value']:.1f} | {row['model']} | {row['average_precision_mean']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_root = args.output_dir.resolve()
    data_dir = output_root / "data"
    results_dir = output_root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset_bundle(args.data_root.resolve())
    if args.holdout_family_id not in bundle.family_ids:
        raise ValueError(f"Unknown holdout family {args.holdout_family_id!r}; valid: {bundle.family_ids}")

    run_plan = build_run_plan(args)
    pd.DataFrame(run_plan).to_csv(results_dir / "planned_runs.csv", index=False)

    dataset_summary = {
        "dataset_name": "bouchard_family_relational",
        "data_root": str(args.data_root.resolve()),
        "family_ids": bundle.family_ids,
        "relation_names": bundle.relation_names,
        "core_relations": sorted(bundle.core_relations),
        "derived_relations": sorted(bundle.derived_relations),
        "atom_count": int(len(bundle.atoms)),
        "positive_counts_by_relation": relation_positive_counts(bundle),
    }
    write_json(data_dir / "dataset_summary.json", dataset_summary)

    if args.dry_run:
        print(f"data_root: {args.data_root.resolve()}")
        print(f"families: {len(bundle.family_ids)}")
        print(f"relations: {len(bundle.relation_names)}")
        print(f"atoms: {len(bundle.atoms):,}")
        print(f"planned_configurations: {len(run_plan)}")
        print(pd.DataFrame(run_plan).head(20).to_string(index=False))
        return

    selected_models = parse_csv_list(args.models)
    result_records: list[dict[str, object]] = []
    threshold_records: list[dict[str, object]] = []

    for config_idx, config in enumerate(run_plan):
        split_name = str(config["split_name"])
        p_value = float(config["p_value"])
        run_idx = int(config["run_idx"])
        split_seed = int(config["split_seed"])

        train_df, valid_df, test_df = build_split_frames(
            bundle=bundle,
            split_name=split_name,
            p_value=p_value,
            holdout_family_id=args.holdout_family_id,
            seed=split_seed,
        )

        train_df = sample_training_negatives(
            train_df,
            negative_sampling_ratio=args.negative_sampling_ratio,
            seed=split_seed,
        )
        train_df = stratified_cap(train_df, args.max_train_rows, seed=split_seed, stratify_cols=["relation", "label"])
        valid_df = stratified_cap(valid_df, args.max_valid_rows, seed=split_seed + 1, stratify_cols=["relation", "label"])
        test_df = stratified_cap(test_df, args.max_test_rows, seed=split_seed + 2, stratify_cols=["relation", "label"])

        y_train = train_df["label"].to_numpy(dtype=np.int64)
        y_valid = valid_df["label"].to_numpy(dtype=np.int64)
        y_test = test_df["label"].to_numpy(dtype=np.int64)
        if np.unique(y_train).size < 2:
            raise ValueError(
                f"Training split collapsed to a single class for split={split_name}, p={p_value}, run={run_idx}."
            )

        X_train = make_feature_frame(train_df, relation_categories=bundle.relation_names, person_categories=bundle.person_categories)
        X_valid = make_feature_frame(valid_df, relation_categories=bundle.relation_names, person_categories=bundle.person_categories)
        X_test = make_feature_frame(test_df, relation_categories=bundle.relation_names, person_categories=bundle.person_categories)

        print(
            f"[family] cfg={config_idx + 1}/{len(run_plan)} split={split_name} p={p_value:.1f} "
            f"run={run_idx} train={len(train_df)} valid={len(valid_df)} test={len(test_df)}",
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
                relation_names=bundle.relation_names,
            )

            overall_metrics = evaluate_predictions(
                y_true=y_test,
                y_pred=outputs["y_pred"],
                positive_proba=outputs["positive_proba"],
            )
            result_records.append(
                {
                    "model": model_name,
                    "split_name": split_name,
                    "p_value": p_value,
                    "run_idx": run_idx,
                    "relation": "overall",
                    "train_rows": int(len(train_df)),
                    "valid_rows": int(len(valid_df)),
                    "test_rows": int(len(test_df)),
                    "fit_time_s": outputs["fit_time_s"],
                    "predict_time_s": outputs["predict_time_s"],
                    **overall_metrics,
                }
            )

            relation_values = X_test["relation"].astype(str).to_numpy()
            for relation in bundle.relation_names:
                mask = relation_values == relation
                if not mask.any():
                    continue
                relation_metrics = evaluate_predictions(
                    y_true=y_test[mask],
                    y_pred=outputs["y_pred"][mask],
                    positive_proba=outputs["positive_proba"][mask],
                )
                result_records.append(
                    {
                        "model": model_name,
                        "split_name": split_name,
                        "p_value": p_value,
                        "run_idx": run_idx,
                        "relation": relation,
                        "train_rows": int(len(train_df)),
                        "valid_rows": int(len(valid_df)),
                        "test_rows": int(mask.sum()),
                        "fit_time_s": outputs["fit_time_s"],
                        "predict_time_s": outputs["predict_time_s"],
                        **relation_metrics,
                    }
                )

            for threshold_record in outputs["threshold_records"]:
                threshold_records.append(
                    {
                        "model": model_name,
                        "split_name": split_name,
                        "p_value": p_value,
                        "run_idx": run_idx,
                        **threshold_record,
                    }
                )

    results_df = pd.DataFrame(result_records)
    thresholds_df = pd.DataFrame(threshold_records)
    results_df.to_csv(results_dir / "per_run_results.csv", index=False)
    thresholds_df.to_csv(results_dir / "thresholds.csv", index=False)

    summary_df = (
        results_df.groupby(["model", "split_name", "p_value", "relation"], as_index=False)
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
    summary_df = summary_df.fillna(0.0)
    summary_df.to_csv(results_dir / "summary_by_model_split_p_relation.csv", index=False)

    summary_overall_df = summary_df.loc[summary_df["relation"] == "overall"].reset_index(drop=True)
    summary_overall_df.to_csv(results_dir / "summary_overall_by_model_split_p.csv", index=False)

    comparable = summary_overall_df.sort_values(["split_name", "p_value", "average_precision_mean"], ascending=[True, False, False])
    comparable.to_csv(results_dir / "bouchard_family_comparable_summary.csv", index=False)

    findings_md = build_findings_md(bundle=bundle, summary_overall_df=summary_overall_df, args=args)
    (results_dir / "findings.md").write_text(findings_md, encoding="utf-8")

    metadata = {
        "launcher": "run_bouchard_family_experiment.py",
        "data_root": str(args.data_root.resolve()),
        "output_dir": str(output_root),
        "models": selected_models,
        "splits": parse_csv_list(args.splits),
        "p_values": parse_float_list(args.p_values),
        "include_family_zero": bool(args.include_family_zero),
        "holdout_family_id": args.holdout_family_id,
        "num_runs": int(args.num_runs),
        "seed": int(args.seed),
        "device": args.device,
        "threshold_mode": args.threshold_mode,
        "decision_threshold": float(args.decision_threshold),
        "negative_sampling_ratio": None if args.negative_sampling_ratio is None else float(args.negative_sampling_ratio),
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
