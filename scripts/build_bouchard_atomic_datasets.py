#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ReflexivityMode = Literal["reflexive", "irreflexive", "none"]
SymmetryMode = Literal["symmetric", "antisymmetric", "none"]


@dataclass(frozen=True)
class AtomicCaseSpec:
    case_id: str
    case_name: str
    reflexivity: ReflexivityMode
    symmetry: SymmetryMode
    transitive: bool
    wave: int
    description: str


MAIN_CASE_SPECS: tuple[AtomicCaseSpec, ...] = (
    AtomicCaseSpec("A01", "symmetric", "none", "symmetric", False, 1, "Symmetric, not transitive."),
    AtomicCaseSpec("A02", "antisymmetric", "none", "antisymmetric", False, 1, "Antisymmetric, not transitive."),
    AtomicCaseSpec("A03", "transitive", "none", "none", True, 1, "Transitive, neither symmetric nor antisymmetric."),
    AtomicCaseSpec(
        "A04",
        "symmetric_transitive",
        "none",
        "symmetric",
        True,
        1,
        "Symmetric and transitive, with mixed diagonal to avoid forcing a reflexive-only case.",
    ),
    AtomicCaseSpec(
        "A05",
        "antisymmetric_transitive",
        "none",
        "antisymmetric",
        True,
        1,
        "Antisymmetric and transitive, with mixed diagonal to avoid forcing a reflexive-only case.",
    ),
)

APPENDIX_CASE_SPECS: tuple[AtomicCaseSpec, ...] = (
    AtomicCaseSpec("A06", "reflexive_symmetric", "reflexive", "symmetric", False, 2, "Reflexive and symmetric."),
    AtomicCaseSpec(
        "A07",
        "reflexive_antisymmetric",
        "reflexive",
        "antisymmetric",
        False,
        2,
        "Reflexive and antisymmetric.",
    ),
    AtomicCaseSpec("A08", "reflexive_transitive", "reflexive", "none", True, 2, "Reflexive and transitive."),
    AtomicCaseSpec(
        "A09",
        "reflexive_symmetric_transitive",
        "reflexive",
        "symmetric",
        True,
        2,
        "Reflexive, symmetric, and transitive.",
    ),
    AtomicCaseSpec(
        "A10",
        "reflexive_antisymmetric_transitive",
        "reflexive",
        "antisymmetric",
        True,
        2,
        "Reflexive, antisymmetric, and transitive.",
    ),
    AtomicCaseSpec(
        "A11",
        "irreflexive_symmetric",
        "irreflexive",
        "symmetric",
        False,
        2,
        "Irreflexive and symmetric.",
    ),
    AtomicCaseSpec(
        "A12",
        "irreflexive_antisymmetric",
        "irreflexive",
        "antisymmetric",
        False,
        2,
        "Irreflexive and antisymmetric.",
    ),
    AtomicCaseSpec(
        "A13",
        "irreflexive_antisymmetric_transitive",
        "irreflexive",
        "antisymmetric",
        True,
        2,
        "Irreflexive, antisymmetric, and transitive.",
    ),
)

CASE_SPECS = {spec.case_name: spec for spec in MAIN_CASE_SPECS + APPENDIX_CASE_SPECS}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build the Bouchard atomic relation datasets as balanced synthetic binary "
            "relation matrices exported as triples and matrices."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "datasets" / "bouchard_atomic_relational",
        help="Directory where the generated atomic dataset will be written.",
    )
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--num-entities", type=int, default=50, help="Entities per relation matrix.")
    parser.add_argument(
        "--matrices-per-case",
        type=int,
        default=1,
        help="Number of independently generated matrices per atomic case.",
    )
    parser.add_argument(
        "--balance-tolerance",
        type=float,
        default=0.01,
        help="Allowed deviation from 50%% positives.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=",".join(spec.case_name for spec in MAIN_CASE_SPECS),
        help="Comma-separated case list. Defaults to the five main-text cases.",
    )
    parser.add_argument(
        "--include-appendix-cases",
        action="store_true",
        help="Append the eight reflexive/irreflexive appendix cases to --cases.",
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def positive_ratio(matrix: np.ndarray) -> float:
    return float(np.mean(matrix == 1))


def is_reflexive(matrix: np.ndarray) -> bool:
    return bool(np.all(np.diag(matrix) == 1))


def is_irreflexive(matrix: np.ndarray) -> bool:
    return bool(np.all(np.diag(matrix) == -1))


def is_symmetric(matrix: np.ndarray) -> bool:
    return bool(np.array_equal(matrix, matrix.T))


def is_antisymmetric(matrix: np.ndarray) -> bool:
    positive = matrix == 1
    return not bool(np.any(np.triu(positive & positive.T, k=1)))


def is_transitive(matrix: np.ndarray) -> bool:
    positive = matrix == 1
    for pivot in range(matrix.shape[0]):
        implied = positive[:, [pivot]] & positive[[pivot], :]
        if np.any(implied & ~positive):
            return False
    return True


def sample_mixed_diagonal(
    rng: np.random.Generator,
    num_entities: int,
    *,
    positive_count: int | None = None,
) -> np.ndarray:
    if positive_count is None:
        diagonal = np.where(rng.random(num_entities) < 0.5, 1, -1).astype(np.int8)
        if np.all(diagonal == 1) or np.all(diagonal == -1):
            diagonal[0] *= -1
        return diagonal

    diagonal = np.full(num_entities, -1, dtype=np.int8)
    order = rng.permutation(num_entities)
    diagonal[order[:positive_count]] = 1
    if np.all(diagonal == 1) or np.all(diagonal == -1):
        raise ValueError("Requested mixed diagonal but produced a pure one.")
    return diagonal


def diagonal_matches_mode(matrix: np.ndarray, mode: ReflexivityMode) -> bool:
    if mode == "reflexive":
        return is_reflexive(matrix)
    if mode == "irreflexive":
        return is_irreflexive(matrix)
    return not is_reflexive(matrix) and not is_irreflexive(matrix)


def symmetry_matches_mode(matrix: np.ndarray, mode: SymmetryMode) -> bool:
    symmetric = is_symmetric(matrix)
    antisymmetric = is_antisymmetric(matrix)
    if mode == "symmetric":
        return symmetric and not antisymmetric
    if mode == "antisymmetric":
        return antisymmetric and not symmetric
    return not symmetric and not antisymmetric


def validate_case_matrix(
    matrix: np.ndarray,
    spec: AtomicCaseSpec,
    *,
    balance_tolerance: float,
) -> None:
    ratio = positive_ratio(matrix)
    if abs(ratio - 0.5) > balance_tolerance:
        raise ValueError(
            f"{spec.case_name}: positive ratio {ratio:.4f} is outside tolerance ±{balance_tolerance:.4f}."
        )
    if not diagonal_matches_mode(matrix, spec.reflexivity):
        raise ValueError(f"{spec.case_name}: diagonal does not match reflexivity mode {spec.reflexivity!r}.")
    if not symmetry_matches_mode(matrix, spec.symmetry):
        raise ValueError(f"{spec.case_name}: symmetry mode check failed for {spec.symmetry!r}.")
    transitive = is_transitive(matrix)
    if transitive != spec.transitive:
        raise ValueError(
            f"{spec.case_name}: expected transitive={spec.transitive}, got transitive={transitive}."
        )


def generate_random_symmetric_matrix(
    rng: np.random.Generator,
    spec: AtomicCaseSpec,
    *,
    num_entities: int,
    balance_tolerance: float,
    max_attempts: int = 10_000,
) -> tuple[np.ndarray, int]:
    for attempt in range(1, max_attempts + 1):
        matrix = np.empty((num_entities, num_entities), dtype=np.int8)
        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                value = 1 if rng.random() < 0.5 else -1
                matrix[i, j] = value
                matrix[j, i] = value
        if spec.reflexivity == "reflexive":
            np.fill_diagonal(matrix, 1)
        elif spec.reflexivity == "irreflexive":
            np.fill_diagonal(matrix, -1)
        else:
            np.fill_diagonal(matrix, sample_mixed_diagonal(rng, num_entities))

        if abs(positive_ratio(matrix) - 0.5) > balance_tolerance:
            continue
        if is_transitive(matrix):
            continue
        validate_case_matrix(matrix, spec, balance_tolerance=balance_tolerance)
        return matrix, attempt
    raise RuntimeError(f"Could not generate a balanced {spec.case_name!r} matrix after {max_attempts} attempts.")


def generate_random_antisymmetric_matrix(
    rng: np.random.Generator,
    spec: AtomicCaseSpec,
    *,
    num_entities: int,
    balance_tolerance: float,
    max_attempts: int = 10_000,
) -> tuple[np.ndarray, int]:
    for attempt in range(1, max_attempts + 1):
        matrix = np.full((num_entities, num_entities), -1, dtype=np.int8)
        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                value = 1 if rng.random() < 0.5 else -1
                matrix[i, j] = value
                matrix[j, i] = -value
        if spec.reflexivity == "reflexive":
            np.fill_diagonal(matrix, 1)
        elif spec.reflexivity == "irreflexive":
            np.fill_diagonal(matrix, -1)
        else:
            np.fill_diagonal(matrix, sample_mixed_diagonal(rng, num_entities))

        if abs(positive_ratio(matrix) - 0.5) > balance_tolerance:
            continue
        if is_transitive(matrix):
            continue
        validate_case_matrix(matrix, spec, balance_tolerance=balance_tolerance)
        return matrix, attempt
    raise RuntimeError(f"Could not generate a balanced {spec.case_name!r} matrix after {max_attempts} attempts.")


def pick_best_integer(
    candidates: list[int],
    score_fn,
    rng: np.random.Generator,
) -> int:
    scored = [(abs(score_fn(value) - 0.5), value) for value in candidates]
    min_delta = min(delta for delta, _ in scored)
    best = [value for delta, value in scored if abs(delta - min_delta) <= 1e-12]
    return int(rng.choice(best))


def build_total_preorder_matrix(
    rng: np.random.Generator,
    spec: AtomicCaseSpec,
    *,
    num_entities: int,
) -> np.ndarray:
    if spec.reflexivity == "irreflexive":
        raise ValueError("Irreflexive transitive-only case is excluded from the Bouchard atomic setup.")

    def ratio_for_block(block_size: int) -> float:
        base = num_entities * (num_entities - 1) // 2
        if spec.reflexivity == "reflexive":
            positives = base + (block_size * (block_size - 1)) // 2 + num_entities
        else:
            positives = base + (block_size * (block_size + 1)) // 2
        return positives / float(num_entities * num_entities)

    block_size = pick_best_integer(list(range(2, num_entities)), ratio_for_block, rng)
    order = rng.permutation(num_entities)
    block = list(order[:block_size])
    groups: list[list[int]] = [block] + [[int(entity)] for entity in order[block_size:]]

    matrix = np.full((num_entities, num_entities), -1, dtype=np.int8)
    if spec.reflexivity == "reflexive":
        np.fill_diagonal(matrix, 1)

    for group_idx, group in enumerate(groups):
        if len(group) > 1:
            for source in group:
                for target in group:
                    matrix[source, target] = 1
        for later_group in groups[group_idx + 1 :]:
            for source in group:
                for target in later_group:
                    matrix[source, target] = 1

    return matrix


def build_antisymmetric_transitive_matrix(
    rng: np.random.Generator,
    spec: AtomicCaseSpec,
    *,
    num_entities: int,
) -> np.ndarray:
    order = rng.permutation(num_entities)
    matrix = np.full((num_entities, num_entities), -1, dtype=np.int8)
    for order_idx, source in enumerate(order):
        for target in order[order_idx + 1 :]:
            matrix[source, target] = 1
            matrix[target, source] = -1

    if spec.reflexivity == "reflexive":
        np.fill_diagonal(matrix, 1)
    elif spec.reflexivity == "irreflexive":
        np.fill_diagonal(matrix, -1)
    else:
        np.fill_diagonal(
            matrix,
            sample_mixed_diagonal(rng, num_entities, positive_count=num_entities // 2),
        )

    return matrix


def build_symmetric_transitive_matrix(
    rng: np.random.Generator,
    spec: AtomicCaseSpec,
    *,
    num_entities: int,
) -> np.ndarray:
    def best_clique_partition(*, require_full_cover: bool) -> tuple[int, ...]:
        target = 0.5 * num_entities * num_entities
        best_sizes: tuple[int, ...] | None = None
        best_delta = float("inf")

        for clique_count in range(2, 4):
            for sizes in combinations_with_replacement(range(1, num_entities), clique_count):
                size_sum = sum(sizes)
                if require_full_cover:
                    if size_sum != num_entities:
                        continue
                elif size_sum >= num_entities:
                    continue

                delta = abs(sum(size * size for size in sizes) - target)
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_sizes = tuple(int(size) for size in sizes)

        if best_sizes is None:
            raise ValueError("Could not find a balanced clique partition for the symmetric transitive case.")
        return best_sizes

    matrix = np.full((num_entities, num_entities), -1, dtype=np.int8)
    order = rng.permutation(num_entities)

    if spec.reflexivity == "reflexive":
        offset = 0
        for size in best_clique_partition(require_full_cover=True):
            group = order[offset : offset + size]
            for source in group:
                for target in group:
                    matrix[source, target] = 1
            offset += size
        return matrix

    offset = 0
    for size in best_clique_partition(require_full_cover=False):
        group = order[offset : offset + size]
        for source in group:
            for target in group:
                matrix[source, target] = 1
        offset += size
    return matrix


def generate_matrix_for_case(
    rng: np.random.Generator,
    spec: AtomicCaseSpec,
    *,
    num_entities: int,
    balance_tolerance: float,
) -> tuple[np.ndarray, int]:
    if spec.symmetry == "symmetric" and not spec.transitive:
        return generate_random_symmetric_matrix(
            rng,
            spec,
            num_entities=num_entities,
            balance_tolerance=balance_tolerance,
        )
    if spec.symmetry == "antisymmetric" and not spec.transitive:
        return generate_random_antisymmetric_matrix(
            rng,
            spec,
            num_entities=num_entities,
            balance_tolerance=balance_tolerance,
        )
    if spec.symmetry == "none" and spec.transitive:
        matrix = build_total_preorder_matrix(rng, spec, num_entities=num_entities)
        validate_case_matrix(matrix, spec, balance_tolerance=balance_tolerance)
        return matrix, 1
    if spec.symmetry == "antisymmetric" and spec.transitive:
        matrix = build_antisymmetric_transitive_matrix(rng, spec, num_entities=num_entities)
        validate_case_matrix(matrix, spec, balance_tolerance=balance_tolerance)
        return matrix, 1
    if spec.symmetry == "symmetric" and spec.transitive:
        matrix = build_symmetric_transitive_matrix(rng, spec, num_entities=num_entities)
        validate_case_matrix(matrix, spec, balance_tolerance=balance_tolerance)
        return matrix, 1
    raise ValueError(f"Unsupported case specification: {spec}")


def build_dataset(
    selected_specs: list[AtomicCaseSpec],
    *,
    num_entities: int,
    matrices_per_case: int,
    seed: int,
    balance_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    entity_ids = [f"e{i:02d}" for i in range(num_entities)]
    atom_records: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []
    case_metadata: list[dict[str, object]] = []
    atom_counter = 0
    rng = np.random.default_rng(seed)

    for spec_idx, spec in enumerate(selected_specs):
        relation_rows.append(
            {
                "relation_id": spec.case_id,
                "relation_name": spec.case_name,
                "reflexivity": spec.reflexivity,
                "symmetry": spec.symmetry,
                "transitive": int(spec.transitive),
                "wave": spec.wave,
                "description": spec.description,
                "matrix_count": matrices_per_case,
            }
        )
        for matrix_idx in range(matrices_per_case):
            matrix_id = f"{spec.case_id}_m{matrix_idx + 1:03d}"
            case_rng = np.random.default_rng(
                int(rng.integers(0, 2**31 - 1)) + 10_000 * spec_idx + matrix_idx
            )
            matrix, attempts = generate_matrix_for_case(
                case_rng,
                spec,
                num_entities=num_entities,
                balance_tolerance=balance_tolerance,
            )
            case_metadata.append(
                {
                    **asdict(spec),
                    "matrix_id": matrix_id,
                    "matrix_index": matrix_idx,
                    "attempts": attempts,
                    "num_entities": num_entities,
                    "positive_count": int(np.sum(matrix == 1)),
                    "negative_count": int(np.sum(matrix == -1)),
                    "positive_ratio": positive_ratio(matrix),
                    "diagonal_positive_count": int(np.sum(np.diag(matrix) == 1)),
                    "property_checks": {
                        "reflexive": is_reflexive(matrix),
                        "irreflexive": is_irreflexive(matrix),
                        "symmetric": is_symmetric(matrix),
                        "antisymmetric": is_antisymmetric(matrix),
                        "transitive": is_transitive(matrix),
                    },
                }
            )
            for source_idx, source_id in enumerate(entity_ids):
                for target_idx, target_id in enumerate(entity_ids):
                    atom_records.append(
                        {
                            "atom_id": f"atom_{atom_counter:06d}",
                            "relation_id": spec.case_id,
                            "relation": spec.case_name,
                            "matrix_id": matrix_id,
                            "source_entity_id": source_id,
                            "target_entity_id": target_id,
                            "label": int(matrix[source_idx, target_idx] == 1),
                        }
                    )
                    atom_counter += 1

    atoms_df = pd.DataFrame(atom_records)
    relation_df = pd.DataFrame(relation_rows).sort_values(["wave", "relation_id"]).reset_index(drop=True)
    metadata = {
        "dataset_name": "bouchard_atomic_relational",
        "random_seed": seed,
        "num_entities": num_entities,
        "num_relations": len(selected_specs),
        "matrices_per_case": matrices_per_case,
        "num_atoms": int(len(atoms_df)),
        "entity_ids": entity_ids,
        "balance_tolerance": balance_tolerance,
        "case_metadata": case_metadata,
        "notes": [
            "Each relation is exported as all ordered entity pairs with a binary label.",
            "matrix_id identifies independent relation worlds for the same atomic case.",
            "The five wave-1 cases match the main text of the Bouchard atomic experiment section.",
            "Appendix reflexive and irreflexive variants are optional.",
        ],
    }
    return atoms_df, relation_df, metadata


def write_case_matrices(
    atoms_df: pd.DataFrame,
    selected_specs: list[AtomicCaseSpec],
    *,
    num_entities: int,
    output_root: Path,
) -> None:
    matrix_root = output_root / "matrices"
    matrix_root.mkdir(parents=True, exist_ok=True)
    ordered_entities = [f"e{i:02d}" for i in range(num_entities)]
    for spec in selected_specs:
        relation_df = atoms_df.loc[atoms_df["relation"] == spec.case_name].copy()
        matrix_ids = sorted(relation_df["matrix_id"].astype(str).unique().tolist())
        if len(matrix_ids) == 1:
            matrix_groups = [(matrix_ids[0], matrix_root, spec.case_name)]
        else:
            case_root = matrix_root / spec.case_name
            case_root.mkdir(parents=True, exist_ok=True)
            matrix_groups = [(matrix_id, case_root, matrix_id) for matrix_id in matrix_ids]

        for matrix_id, target_root, stem in matrix_groups:
            matrix_relation_df = relation_df.loc[relation_df["matrix_id"] == matrix_id].copy()
            matrix_relation_df["source_entity_id"] = pd.Categorical(
                matrix_relation_df["source_entity_id"],
                categories=ordered_entities,
            )
            matrix_relation_df["target_entity_id"] = pd.Categorical(
                matrix_relation_df["target_entity_id"],
                categories=ordered_entities,
            )
            matrix_df = matrix_relation_df.pivot(
                index="source_entity_id",
                columns="target_entity_id",
                values="label",
            ).reindex(index=ordered_entities, columns=ordered_entities)
            sign_matrix = np.where(matrix_df.to_numpy(dtype=np.int8) == 1, 1, -1).astype(np.int8)
            np.save(target_root / f"{stem}.npy", sign_matrix)
            pd.DataFrame(sign_matrix, index=ordered_entities, columns=ordered_entities).to_csv(
                target_root / f"{stem}.csv"
            )


def main() -> None:
    args = parse_args()
    if args.matrices_per_case <= 0:
        raise ValueError("--matrices-per-case must be positive.")
    selected_case_names = parse_csv_list(args.cases)
    if args.include_appendix_cases:
        selected_case_names.extend(spec.case_name for spec in APPENDIX_CASE_SPECS)
    selected_case_names = list(dict.fromkeys(selected_case_names))
    unknown = sorted(set(selected_case_names) - set(CASE_SPECS))
    if unknown:
        raise ValueError(f"Unknown atomic case(s): {unknown}")

    selected_specs = [CASE_SPECS[name] for name in selected_case_names]
    atoms_df, relation_df, metadata = build_dataset(
        selected_specs,
        num_entities=args.num_entities,
        matrices_per_case=args.matrices_per_case,
        seed=args.seed,
        balance_tolerance=args.balance_tolerance,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atoms_df.to_csv(output_root / "atomic_relation_atoms.csv", index=False)
    relation_df.to_csv(output_root / "relation_types.csv", index=False)
    write_case_matrices(atoms_df, selected_specs, num_entities=args.num_entities, output_root=output_root)
    write_json(output_root / "metadata.json", metadata)

    print(f"output_root: {output_root}")
    print(f"cases: {len(selected_specs)}")
    print(f"entities_per_case: {args.num_entities}")
    print(f"matrices_per_case: {args.matrices_per_case}")
    print(f"rows: {len(atoms_df):,}")
    print(
        relation_df[["relation_name", "reflexivity", "symmetry", "transitive", "wave", "matrix_count"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
