from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_bouchard_family_datasets import build_family_world, build_flat_table, ensure_dir, write_outputs


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def positive_counts_by_relation(bundle: dict[str, object]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in bundle["pair_relation_atoms"]:
        if int(row["label"]) == 1:
            counts[str(row["relation"])] += 1
    return dict(sorted(counts.items()))


def split_counts(bundle: dict[str, object], column: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in bundle["pair_relation_atoms"]:
        counts[str(row[column])] += 1
    return dict(sorted(counts.items()))


def build_group_manifest(
    *,
    group_name: str,
    seed: int,
    output_root: Path,
    bundle: dict[str, object],
    flat_rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "group_name": group_name,
        "seed": int(seed),
        "output_root": str(output_root.resolve()),
        "relational_root": str((output_root / "bouchard_family_relational").resolve()),
        "flat_root": str((output_root / "bouchard_family_flat").resolve()),
        "num_families": int(len(bundle["families"])),
        "num_people": int(len(bundle["persons"])),
        "num_relations": int(len(bundle["relation_types"])),
        "num_atoms": int(len(bundle["pair_relation_atoms"])),
        "num_flat_rows": int(len(flat_rows)),
        "positive_counts_by_relation": positive_counts_by_relation(bundle),
        "split_counts": {
            "split_random": split_counts(bundle, "split_random"),
            "split_evidence_p10": split_counts(bundle, "split_evidence_p10"),
            "split_family_p10": split_counts(bundle, "split_family_p10"),
            "split_family_p00": split_counts(bundle, "split_family_p00"),
        },
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_collection_summary(path: Path, manifests: list[dict[str, object]]) -> None:
    fieldnames = [
        "group_name",
        "seed",
        "num_families",
        "num_people",
        "num_relations",
        "num_atoms",
        "num_flat_rows",
        "relational_root",
        "flat_root",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for manifest in manifests:
            writer.writerow({key: manifest[key] for key in fieldnames})


def build_readme(group_prefix: str, manifests: list[dict[str, object]]) -> str:
    lines = ["# Bouchard Family Dataset Groups", ""]
    lines.append(
        "This directory contains multiple seed-specific copies of the Bouchard-style family dataset, "
        "each exported in flat and relational form."
    )
    lines.append("")
    lines.append("## Groups")
    lines.append("")
    for manifest in manifests:
        lines.append(
            f"- `{manifest['group_name']}`: seed `{manifest['seed']}`, "
            f"`{manifest['num_atoms']:,}` relational atoms, "
            f"`{manifest['num_flat_rows']:,}` flat rows."
        )
    lines.append("")
    lines.append("## How To Use")
    lines.append("")
    lines.append("Point the family TFM runner at one group at a time, for example:")
    lines.append("")
    first_group = manifests[0]["group_name"]
    first_root = manifests[0]["relational_root"]
    lines.append("```bash")
    lines.append(
        "python bouchard_family_tfm_experiment/run_bouchard_family_experiment.py "
        f"--data-root {first_root}"
    )
    lines.append("```")
    lines.append("")
    lines.append("The full collection summary is in `collection_summary.csv` and `manifest.json`.")
    lines.append("")
    lines.append(f"Group prefix used: `{group_prefix}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multiple seed-specific groups of the Bouchard family dataset."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset_groups"),
        help="Directory that will receive the grouped datasets.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="20260419,20260420,20260421,20260422",
        help="Comma-separated seeds. By default this includes the existing canonical seed plus three new seeds.",
    )
    parser.add_argument(
        "--group-prefix",
        type=str,
        default="bouchard_family_group",
        help="Prefix used for each generated group directory.",
    )
    parser.add_argument(
        "--num-families",
        type=int,
        default=5,
        help="Number of families per generated dataset group.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    seeds = parse_int_list(args.seeds)
    manifests: list[dict[str, object]] = []

    for group_index, seed in enumerate(seeds, start=1):
        group_name = f"{args.group_prefix}_{group_index:02d}"
        group_root = args.output_root / group_name
        bundle = build_family_world(seed=seed, num_families=args.num_families)
        flat_rows = build_flat_table(bundle)
        write_outputs(bundle=bundle, flat_rows=flat_rows, output_root=group_root, seed=seed)
        manifest = build_group_manifest(
            group_name=group_name,
            seed=seed,
            output_root=group_root,
            bundle=bundle,
            flat_rows=flat_rows,
        )
        write_json(group_root / "group_manifest.json", manifest)
        manifests.append(manifest)

    write_json(args.output_root / "manifest.json", {"groups": manifests})
    write_collection_summary(args.output_root / "collection_summary.csv", manifests)
    (args.output_root / "README.md").write_text(
        build_readme(args.group_prefix, manifests),
        encoding="utf-8",
    )

    print(f"Wrote {len(manifests)} dataset groups to {args.output_root}")


if __name__ == "__main__":
    main()
