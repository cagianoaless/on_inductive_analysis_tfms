from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


CORE_RELATIONS = {"mother", "father", "son", "daughter"}
RELATION_TYPES = [
    ("R01", "mother", "core"),
    ("R02", "father", "core"),
    ("R03", "son", "core"),
    ("R04", "daughter", "core"),
    ("R05", "husband", "derived"),
    ("R06", "wife", "derived"),
    ("R07", "brother", "derived"),
    ("R08", "sister", "derived"),
    ("R09", "uncle", "derived"),
    ("R10", "aunt", "derived"),
    ("R11", "nephew", "derived"),
    ("R12", "niece", "derived"),
    ("R13", "cousin", "derived"),
    ("R14", "grandfather", "derived"),
    ("R15", "grandson", "derived"),
    ("R16", "grandmother", "derived"),
    ("R17", "granddaughter", "derived"),
]
RELATION_ID_BY_NAME = {name: relation_id for relation_id, name, _ in RELATION_TYPES}
RELATION_GROUP_BY_NAME = {name: group for _, name, group in RELATION_TYPES}
SPLIT_COLUMNS = [
    "split_random",
    "split_evidence_p10",
    "split_family_p10",
    "split_family_p00",
]


def opposite_sex(sex: str) -> str:
    return "F" if sex == "M" else "M"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def share_both_parents(
    left: str,
    right: str,
    parents_by_child: dict[str, dict[str, str]],
) -> bool:
    if left == right:
        return False
    left_parents = parents_by_child.get(left)
    right_parents = parents_by_child.get(right)
    if not left_parents or not right_parents:
        return False
    return (
        left_parents.get("father") == right_parents.get("father")
        and left_parents.get("mother") == right_parents.get("mother")
    )


def siblings_of(
    person_id: str,
    family_members: list[str],
    parents_by_child: dict[str, dict[str, str]],
) -> set[str]:
    siblings = set()
    for candidate in family_members:
        if share_both_parents(person_id, candidate, parents_by_child):
            siblings.add(candidate)
    return siblings


def grouped_split_assignments(
    row_indices: list[int],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[int, str]:
    rng = random.Random(seed)
    shuffled = list(row_indices)
    rng.shuffle(shuffled)
    n_rows = len(shuffled)
    n_train = int(round(n_rows * train_frac))
    n_train = min(max(n_train, 0), n_rows)
    n_val = int(round(n_rows * val_frac))
    n_val = min(max(n_val, 0), n_rows - n_train)
    assignments: dict[int, str] = {}
    for idx in shuffled[:n_train]:
        assignments[idx] = "train"
    for idx in shuffled[n_train : n_train + n_val]:
        assignments[idx] = "val"
    for idx in shuffled[n_train + n_val :]:
        assignments[idx] = "test"
    return assignments


def attach_split_columns(pair_relation_atoms: list[dict[str, object]], seed: int) -> None:
    grouped_random: dict[tuple[object, ...], list[int]] = defaultdict(list)
    grouped_evidence_derived: dict[tuple[object, ...], list[int]] = defaultdict(list)
    grouped_family_holdout: dict[tuple[object, ...], list[int]] = defaultdict(list)

    for idx, row in enumerate(pair_relation_atoms):
        family_id = str(row["family_id"])
        relation = str(row["relation"])
        label = int(row["label"])
        grouped_random[(family_id, relation, label)].append(idx)

        if relation in CORE_RELATIONS:
            row["split_evidence_p10"] = "train"
            row["split_family_p10"] = "train"
            row["split_family_p00"] = "train"
        else:
            grouped_evidence_derived[(family_id, relation, label)].append(idx)
            if family_id == "family_05":
                grouped_family_holdout[(relation, label)].append(idx)
            else:
                row["split_family_p10"] = "train"
                row["split_family_p00"] = "train"

    for group_idx, indices in enumerate(grouped_random.values()):
        assignments = grouped_split_assignments(indices, train_frac=0.8, val_frac=0.1, seed=seed + group_idx)
        for idx, split in assignments.items():
            pair_relation_atoms[idx]["split_random"] = split

    for group_idx, indices in enumerate(grouped_evidence_derived.values()):
        assignments = grouped_split_assignments(indices, train_frac=0.1, val_frac=0.1, seed=seed + 10_000 + group_idx)
        for idx, split in assignments.items():
            pair_relation_atoms[idx]["split_evidence_p10"] = split

    for group_idx, indices in enumerate(grouped_family_holdout.values()):
        p10_assignments = grouped_split_assignments(indices, train_frac=0.1, val_frac=0.1, seed=seed + 20_000 + group_idx)
        p00_assignments = grouped_split_assignments(indices, train_frac=0.0, val_frac=0.1, seed=seed + 30_000 + group_idx)
        for idx, split in p10_assignments.items():
            pair_relation_atoms[idx]["split_family_p10"] = split
        for idx, split in p00_assignments.items():
            pair_relation_atoms[idx]["split_family_p00"] = split

    for row in pair_relation_atoms:
        row["split"] = row["split_random"]


def build_family_world(seed: int, num_families: int) -> dict[str, object]:
    families: list[dict[str, object]] = []
    persons: list[dict[str, object]] = []
    marriages: list[dict[str, object]] = []
    parent_child_edges: list[dict[str, object]] = []

    people_by_family: dict[str, list[str]] = defaultdict(list)
    person_by_id: dict[str, dict[str, object]] = {}
    spouse_by_person: dict[str, str] = {}
    parents_by_child: dict[str, dict[str, str]] = defaultdict(dict)
    children_by_parent: dict[str, set[str]] = defaultdict(set)

    marriage_counter = 1
    edge_counter = 1

    for family_num in range(1, num_families + 1):
        family_id = f"family_{family_num:02d}"
        family_name = f"Synthetic Family {family_num:02d}"
        local_rng = random.Random(seed + family_num * 1_000)
        local_person_counter = 1

        def add_person(
            *,
            sex: str,
            generation: int,
            branch_id: int,
            role_code: str,
            is_blood_relative_to_root: int,
        ) -> str:
            nonlocal local_person_counter
            person_id = f"{family_id}_person_{local_person_counter:02d}"
            person_row = {
                "person_id": person_id,
                "family_id": family_id,
                "display_name": f"{family_name} Person {local_person_counter:02d}",
                "family_local_index": local_person_counter,
                "sex": sex,
                "generation": generation,
                "branch_id": branch_id,
                "role_code": role_code,
                "is_blood_relative_to_root": is_blood_relative_to_root,
            }
            persons.append(person_row)
            person_by_id[person_id] = person_row
            people_by_family[family_id].append(person_id)
            local_person_counter += 1
            return person_id

        def add_marriage(person_a: str, person_b: str) -> None:
            nonlocal marriage_counter
            male = person_a if str(person_by_id[person_a]["sex"]) == "M" else person_b
            female = person_b if male == person_a else person_a
            marriages.append(
                {
                    "marriage_id": f"marriage_{marriage_counter:03d}",
                    "family_id": family_id,
                    "husband_id": male,
                    "wife_id": female,
                }
            )
            spouse_by_person[person_a] = person_b
            spouse_by_person[person_b] = person_a
            marriage_counter += 1

        def add_parent_child(parent_id: str, child_id: str, parent_role: str) -> None:
            nonlocal edge_counter
            parent_child_edges.append(
                {
                    "edge_id": f"edge_{edge_counter:04d}",
                    "family_id": family_id,
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "parent_role": parent_role,
                }
            )
            parents_by_child[child_id][parent_role] = parent_id
            children_by_parent[parent_id].add(child_id)
            edge_counter += 1

        root_father = add_person(
            sex="M",
            generation=0,
            branch_id=0,
            role_code="root_partner",
            is_blood_relative_to_root=1,
        )
        root_mother = add_person(
            sex="F",
            generation=0,
            branch_id=0,
            role_code="root_partner",
            is_blood_relative_to_root=1,
        )
        add_marriage(root_father, root_mother)

        gen1_couples: list[tuple[str, str, int]] = []
        for branch_id in range(1, 4):
            child_sex = local_rng.choice(["M", "F"])
            child_id = add_person(
                sex=child_sex,
                generation=1,
                branch_id=branch_id,
                role_code="gen1_child",
                is_blood_relative_to_root=1,
            )
            spouse_id = add_person(
                sex=opposite_sex(child_sex),
                generation=1,
                branch_id=branch_id,
                role_code="gen1_spouse",
                is_blood_relative_to_root=0,
            )
            add_marriage(child_id, spouse_id)
            add_parent_child(root_father, child_id, "father")
            add_parent_child(root_mother, child_id, "mother")
            gen1_couples.append((child_id, spouse_id, branch_id))

        gen2_children: list[tuple[str, int]] = []
        for child_id, spouse_id, branch_id in gen1_couples:
            father_id = child_id if str(person_by_id[child_id]["sex"]) == "M" else spouse_id
            mother_id = child_id if str(person_by_id[child_id]["sex"]) == "F" else spouse_id
            for _ in range(3):
                grandchild_id = add_person(
                    sex=local_rng.choice(["M", "F"]),
                    generation=2,
                    branch_id=branch_id,
                    role_code="gen2_child",
                    is_blood_relative_to_root=1,
                )
                add_parent_child(father_id, grandchild_id, "father")
                add_parent_child(mother_id, grandchild_id, "mother")
                gen2_children.append((grandchild_id, branch_id))

        selected_for_spouses = local_rng.sample([person_id for person_id, _ in gen2_children], 6)
        for grandchild_id in selected_for_spouses:
            spouse_id = add_person(
                sex=opposite_sex(str(person_by_id[grandchild_id]["sex"])),
                generation=2,
                branch_id=int(person_by_id[grandchild_id]["branch_id"]),
                role_code="gen2_spouse",
                is_blood_relative_to_root=0,
            )
            add_marriage(grandchild_id, spouse_id)

        family_person_ids = people_by_family[family_id]
        if len(family_person_ids) != 23:
            raise ValueError(f"{family_id} has {len(family_person_ids)} persons, expected 23.")

        families.append(
            {
                "family_id": family_id,
                "family_name": family_name,
                "num_generations": 3,
                "num_people": len(family_person_ids),
            }
        )

    relation_types = [
        {
            "relation_id": relation_id,
            "relation_name": relation_name,
            "relation_group": relation_group,
            "arity": 2,
        }
        for relation_id, relation_name, relation_group in RELATION_TYPES
    ]

    pair_relation_atoms: list[dict[str, object]] = []
    atom_counter = 1

    for family in families:
        family_id = str(family["family_id"])
        family_members = sorted(people_by_family[family_id])
        positives: dict[str, set[tuple[str, str]]] = {name: set() for _, name, _ in RELATION_TYPES}

        for edge in parent_child_edges:
            if edge["family_id"] != family_id:
                continue
            parent_id = str(edge["parent_id"])
            child_id = str(edge["child_id"])
            if edge["parent_role"] == "mother":
                positives["mother"].add((parent_id, child_id))
            else:
                positives["father"].add((parent_id, child_id))
            child_sex = str(person_by_id[child_id]["sex"])
            if child_sex == "M":
                positives["son"].add((child_id, parent_id))
            else:
                positives["daughter"].add((child_id, parent_id))

        for marriage in marriages:
            if marriage["family_id"] != family_id:
                continue
            husband_id = str(marriage["husband_id"])
            wife_id = str(marriage["wife_id"])
            positives["husband"].add((husband_id, wife_id))
            positives["wife"].add((wife_id, husband_id))

        for source_id in family_members:
            for target_id in family_members:
                if share_both_parents(source_id, target_id, parents_by_child):
                    if str(person_by_id[source_id]["sex"]) == "M":
                        positives["brother"].add((source_id, target_id))
                    else:
                        positives["sister"].add((source_id, target_id))

        for person_id in family_members:
            parent_ids = set(parents_by_child.get(person_id, {}).values())
            grandparent_ids: set[str] = set()
            for parent_id in parent_ids:
                grandparent_ids.update(parents_by_child.get(parent_id, {}).values())
            for grandparent_id in grandparent_ids:
                if str(person_by_id[grandparent_id]["sex"]) == "M":
                    positives["grandfather"].add((grandparent_id, person_id))
                else:
                    positives["grandmother"].add((grandparent_id, person_id))
                if str(person_by_id[person_id]["sex"]) == "M":
                    positives["grandson"].add((person_id, grandparent_id))
                else:
                    positives["granddaughter"].add((person_id, grandparent_id))

        for person_id in family_members:
            parent_ids = set(parents_by_child.get(person_id, {}).values())
            relatives: set[str] = set()
            for parent_id in parent_ids:
                relatives.update(siblings_of(parent_id, family_members, parents_by_child))
            for relative_id in relatives:
                if str(person_by_id[relative_id]["sex"]) == "M":
                    positives["uncle"].add((relative_id, person_id))
                else:
                    positives["aunt"].add((relative_id, person_id))
                if str(person_by_id[person_id]["sex"]) == "M":
                    positives["nephew"].add((person_id, relative_id))
                else:
                    positives["niece"].add((person_id, relative_id))

        for source_id in family_members:
            source_parents = set(parents_by_child.get(source_id, {}).values())
            if not source_parents:
                continue
            for target_id in family_members:
                if source_id == target_id:
                    continue
                target_parents = set(parents_by_child.get(target_id, {}).values())
                if not target_parents:
                    continue
                if share_both_parents(source_id, target_id, parents_by_child):
                    continue
                is_cousin = False
                for source_parent in source_parents:
                    for target_parent in target_parents:
                        if share_both_parents(source_parent, target_parent, parents_by_child):
                            is_cousin = True
                            break
                    if is_cousin:
                        break
                if is_cousin:
                    positives["cousin"].add((source_id, target_id))

        for _, relation_name, _ in RELATION_TYPES:
            relation_id = RELATION_ID_BY_NAME[relation_name]
            for source_id in family_members:
                for target_id in family_members:
                    pair_relation_atoms.append(
                        {
                            "atom_id": f"atom_{atom_counter:06d}",
                            "family_id": family_id,
                            "relation_id": relation_id,
                            "relation": relation_name,
                            "source_person_id": source_id,
                            "target_person_id": target_id,
                            "label": int((source_id, target_id) in positives[relation_name]),
                        }
                    )
                    atom_counter += 1

    attach_split_columns(pair_relation_atoms, seed=seed)

    return {
        "families": families,
        "persons": persons,
        "marriages": marriages,
        "parent_child_edges": parent_child_edges,
        "relation_types": relation_types,
        "pair_relation_atoms": pair_relation_atoms,
        "person_by_id": person_by_id,
        "parents_by_child": parents_by_child,
        "children_by_parent": children_by_parent,
        "spouse_by_person": spouse_by_person,
    }


def build_flat_table(bundle: dict[str, object]) -> list[dict[str, object]]:
    pair_relation_atoms = list(bundle["pair_relation_atoms"])
    person_by_id = dict(bundle["person_by_id"])
    parents_by_child = dict(bundle["parents_by_child"])
    children_by_parent = dict(bundle["children_by_parent"])
    spouse_by_person = dict(bundle["spouse_by_person"])

    sorted_atoms = sorted(
        pair_relation_atoms,
        key=lambda row: (
            str(row["family_id"]),
            str(row["relation"]),
            str(row["source_person_id"]),
            str(row["target_person_id"]),
        ),
    )

    start_time = datetime(2020, 1, 1, 0, 0, 0)
    flat_rows: list[dict[str, object]] = []
    for index, atom in enumerate(sorted_atoms):
        source = person_by_id[str(atom["source_person_id"])]
        target = person_by_id[str(atom["target_person_id"])]
        source_id = str(source["person_id"])
        target_id = str(target["person_id"])
        event_time = start_time + timedelta(minutes=index)
        flat_rows.append(
            {
                "atom_id": atom["atom_id"],
                "dataset_name": "bouchard_family_flat",
                "anchor_table": "pair_relation_atoms",
                "event_time": event_time.isoformat(timespec="seconds"),
                "split": atom["split_random"],
                "split_random": atom["split_random"],
                "split_evidence_p10": atom["split_evidence_p10"],
                "split_family_p10": atom["split_family_p10"],
                "split_family_p00": atom["split_family_p00"],
                "family_id": atom["family_id"],
                "relation_id": atom["relation_id"],
                "relation": atom["relation"],
                "relation_group": RELATION_GROUP_BY_NAME[str(atom["relation"])],
                "label": atom["label"],
                "source_person_id": source_id,
                "source_local_index": source["family_local_index"],
                "source_sex": source["sex"],
                "source_generation": source["generation"],
                "source_branch_id": source["branch_id"],
                "source_role_code": source["role_code"],
                "source_is_blood_relative_to_root": source["is_blood_relative_to_root"],
                "source_parent_count": len(parents_by_child.get(source_id, {})),
                "source_child_count": len(children_by_parent.get(source_id, set())),
                "source_spouse_count": int(source_id in spouse_by_person),
                "target_person_id": target_id,
                "target_local_index": target["family_local_index"],
                "target_sex": target["sex"],
                "target_generation": target["generation"],
                "target_branch_id": target["branch_id"],
                "target_role_code": target["role_code"],
                "target_is_blood_relative_to_root": target["is_blood_relative_to_root"],
                "target_parent_count": len(parents_by_child.get(target_id, {})),
                "target_child_count": len(children_by_parent.get(target_id, set())),
                "target_spouse_count": int(target_id in spouse_by_person),
                "same_generation": int(source["generation"] == target["generation"]),
                "same_branch": int(source["branch_id"] == target["branch_id"]),
                "same_sex": int(source["sex"] == target["sex"]),
                "generation_gap_signed": int(target["generation"]) - int(source["generation"]),
                "generation_gap_abs": abs(int(target["generation"]) - int(source["generation"])),
            }
        )
    return flat_rows


def split_counts(rows: list[dict[str, object]], column: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row[column])] += 1
    return dict(sorted(counts.items()))


def relational_schema() -> dict[str, object]:
    return {
        "dataset_name": "bouchard_family_relational",
        "tables": {
            "families": {
                "primary_key": "family_id",
                "columns": {
                    "family_id": "string",
                    "family_name": "string",
                    "num_generations": "integer",
                    "num_people": "integer",
                },
            },
            "persons": {
                "primary_key": "person_id",
                "columns": {
                    "person_id": "string",
                    "family_id": "string",
                    "display_name": "string",
                    "family_local_index": "integer",
                    "sex": "categorical",
                    "generation": "integer",
                    "branch_id": "integer",
                    "role_code": "categorical",
                    "is_blood_relative_to_root": "boolean_int",
                },
            },
            "marriages": {
                "primary_key": "marriage_id",
                "columns": {
                    "marriage_id": "string",
                    "family_id": "string",
                    "husband_id": "string",
                    "wife_id": "string",
                },
            },
            "parent_child_edges": {
                "primary_key": "edge_id",
                "columns": {
                    "edge_id": "string",
                    "family_id": "string",
                    "parent_id": "string",
                    "child_id": "string",
                    "parent_role": "categorical",
                },
            },
            "relation_types": {
                "primary_key": "relation_id",
                "columns": {
                    "relation_id": "string",
                    "relation_name": "string",
                    "relation_group": "categorical",
                    "arity": "integer",
                },
            },
            "pair_relation_atoms": {
                "primary_key": "atom_id",
                "columns": {
                    "atom_id": "string",
                    "family_id": "string",
                    "relation_id": "string",
                    "relation": "string",
                    "source_person_id": "string",
                    "target_person_id": "string",
                    "label": "boolean_int",
                    "split": "categorical",
                    "split_random": "categorical",
                    "split_evidence_p10": "categorical",
                    "split_family_p10": "categorical",
                    "split_family_p00": "categorical",
                },
            },
        },
        "foreign_keys": [
            {"child_table": "persons", "child_column": "family_id", "parent_table": "families", "parent_column": "family_id"},
            {"child_table": "marriages", "child_column": "family_id", "parent_table": "families", "parent_column": "family_id"},
            {"child_table": "marriages", "child_column": "husband_id", "parent_table": "persons", "parent_column": "person_id"},
            {"child_table": "marriages", "child_column": "wife_id", "parent_table": "persons", "parent_column": "person_id"},
            {"child_table": "parent_child_edges", "child_column": "family_id", "parent_table": "families", "parent_column": "family_id"},
            {"child_table": "parent_child_edges", "child_column": "parent_id", "parent_table": "persons", "parent_column": "person_id"},
            {"child_table": "parent_child_edges", "child_column": "child_id", "parent_table": "persons", "parent_column": "person_id"},
            {"child_table": "pair_relation_atoms", "child_column": "family_id", "parent_table": "families", "parent_column": "family_id"},
            {"child_table": "pair_relation_atoms", "child_column": "relation_id", "parent_table": "relation_types", "parent_column": "relation_id"},
            {"child_table": "pair_relation_atoms", "child_column": "source_person_id", "parent_table": "persons", "parent_column": "person_id"},
            {"child_table": "pair_relation_atoms", "child_column": "target_person_id", "parent_table": "persons", "parent_column": "person_id"},
        ],
    }


def flat_schema() -> dict[str, object]:
    return {
        "dataset_name": "bouchard_family_flat",
        "primary_key": "atom_id",
        "target_column": "label",
        "event_time_column": "event_time",
        "columns": {
            "atom_id": "string",
            "dataset_name": "string",
            "anchor_table": "string",
            "event_time": "timestamp_iso8601",
            "split": "categorical",
            "split_random": "categorical",
            "split_evidence_p10": "categorical",
            "split_family_p10": "categorical",
            "split_family_p00": "categorical",
            "family_id": "string",
            "relation_id": "string",
            "relation": "string",
            "relation_group": "categorical",
            "label": "boolean_int",
            "source_person_id": "string",
            "source_local_index": "integer",
            "source_sex": "categorical",
            "source_generation": "integer",
            "source_branch_id": "integer",
            "source_role_code": "categorical",
            "source_is_blood_relative_to_root": "boolean_int",
            "source_parent_count": "integer",
            "source_child_count": "integer",
            "source_spouse_count": "integer",
            "target_person_id": "string",
            "target_local_index": "integer",
            "target_sex": "categorical",
            "target_generation": "integer",
            "target_branch_id": "integer",
            "target_role_code": "categorical",
            "target_is_blood_relative_to_root": "boolean_int",
            "target_parent_count": "integer",
            "target_child_count": "integer",
            "target_spouse_count": "integer",
            "same_generation": "boolean_int",
            "same_branch": "boolean_int",
            "same_sex": "boolean_int",
            "generation_gap_signed": "integer",
            "generation_gap_abs": "integer",
        },
    }


def validate_relational(bundle: dict[str, object]) -> str:
    families = list(bundle["families"])
    persons = list(bundle["persons"])
    marriages = list(bundle["marriages"])
    parent_child_edges = list(bundle["parent_child_edges"])
    relation_types = list(bundle["relation_types"])
    pair_relation_atoms = list(bundle["pair_relation_atoms"])

    family_ids = {str(row["family_id"]) for row in families}
    person_ids = {str(row["person_id"]) for row in persons}
    relation_ids = {str(row["relation_id"]) for row in relation_types}

    lines = ["# Validation report: bouchard_family_relational", "", "## Row counts"]
    lines.append(f"- `families`: {len(families):,}")
    lines.append(f"- `persons`: {len(persons):,}")
    lines.append(f"- `marriages`: {len(marriages):,}")
    lines.append(f"- `parent_child_edges`: {len(parent_child_edges):,}")
    lines.append(f"- `relation_types`: {len(relation_types):,}")
    lines.append(f"- `pair_relation_atoms`: {len(pair_relation_atoms):,}")
    lines.append("")
    lines.append("## Primary key checks")
    for table_name, rows, key in [
        ("families", families, "family_id"),
        ("persons", persons, "person_id"),
        ("marriages", marriages, "marriage_id"),
        ("parent_child_edges", parent_child_edges, "edge_id"),
        ("relation_types", relation_types, "relation_id"),
        ("pair_relation_atoms", pair_relation_atoms, "atom_id"),
    ]:
        values = [str(row[key]) for row in rows]
        duplicate_count = len(values) - len(set(values))
        lines.append(f"- `{table_name}.{key}` duplicates: {duplicate_count}")

    lines.append("")
    lines.append("## Foreign key checks")
    fk_violations = {
        "persons.family_id -> families.family_id": sum(1 for row in persons if str(row["family_id"]) not in family_ids),
        "marriages.family_id -> families.family_id": sum(1 for row in marriages if str(row["family_id"]) not in family_ids),
        "marriages.husband_id -> persons.person_id": sum(1 for row in marriages if str(row["husband_id"]) not in person_ids),
        "marriages.wife_id -> persons.person_id": sum(1 for row in marriages if str(row["wife_id"]) not in person_ids),
        "parent_child_edges.family_id -> families.family_id": sum(1 for row in parent_child_edges if str(row["family_id"]) not in family_ids),
        "parent_child_edges.parent_id -> persons.person_id": sum(1 for row in parent_child_edges if str(row["parent_id"]) not in person_ids),
        "parent_child_edges.child_id -> persons.person_id": sum(1 for row in parent_child_edges if str(row["child_id"]) not in person_ids),
        "pair_relation_atoms.family_id -> families.family_id": sum(1 for row in pair_relation_atoms if str(row["family_id"]) not in family_ids),
        "pair_relation_atoms.relation_id -> relation_types.relation_id": sum(1 for row in pair_relation_atoms if str(row["relation_id"]) not in relation_ids),
        "pair_relation_atoms.source_person_id -> persons.person_id": sum(1 for row in pair_relation_atoms if str(row["source_person_id"]) not in person_ids),
        "pair_relation_atoms.target_person_id -> persons.person_id": sum(1 for row in pair_relation_atoms if str(row["target_person_id"]) not in person_ids),
    }
    for label, count in fk_violations.items():
        lines.append(f"- `{label}` violations: {count}")

    family_person_counts: dict[str, int] = defaultdict(int)
    for row in persons:
        family_person_counts[str(row["family_id"])] += 1

    cross_family_atom_mismatches = 0
    person_family_lookup = {str(row["person_id"]): str(row["family_id"]) for row in persons}
    for row in pair_relation_atoms:
        family_id = str(row["family_id"])
        source_family = person_family_lookup[str(row["source_person_id"])]
        target_family = person_family_lookup[str(row["target_person_id"])]
        if family_id != source_family or family_id != target_family:
            cross_family_atom_mismatches += 1

    lines.append("")
    lines.append("## Semantic checks")
    lines.append(f"- families with `23` persons: {sum(1 for count in family_person_counts.values() if count == 23)} / {len(family_person_counts)}")
    lines.append(f"- cross-family atom mismatches: {cross_family_atom_mismatches}")
    lines.append(f"- expected within-family atom count (`5 x 23 x 23 x 17`): {5 * 23 * 23 * 17:,}")
    lines.append(f"- actual atom count: {len(pair_relation_atoms):,}")
    for split_column in ["split_random", "split_evidence_p10", "split_family_p10", "split_family_p00"]:
        lines.append(f"- {split_column} counts: {split_counts(pair_relation_atoms, split_column)}")

    return "\n".join(lines) + "\n"


def validate_flat(flat_rows: list[dict[str, object]]) -> str:
    atom_ids = [str(row["atom_id"]) for row in flat_rows]
    duplicate_count = len(atom_ids) - len(set(atom_ids))
    label_counts = defaultdict(int)
    for row in flat_rows:
        label_counts[int(row["label"])] += 1

    lines = ["# Validation report: bouchard_family_flat", "", "## Table checks"]
    lines.append(f"- rows: {len(flat_rows):,}")
    lines.append(f"- duplicate `atom_id` values: {duplicate_count}")
    lines.append(f"- label counts: {dict(sorted(label_counts.items()))}")
    for split_column in ["split", "split_random", "split_evidence_p10", "split_family_p10", "split_family_p00"]:
        lines.append(f"- {split_column} counts: {split_counts(flat_rows, split_column)}")
    if flat_rows:
        lines.append(f"- event_time min: {flat_rows[0]['event_time']}")
        lines.append(f"- event_time max: {flat_rows[-1]['event_time']}")
    return "\n".join(lines) + "\n"


def build_readme() -> str:
    return """# Bouchard Family Datasets

This directory contains one synthetic family-kinship world exported in two forms:

- `bouchard_family_flat/`: one denormalized flat table for tabular models
- `bouchard_family_relational/`: a normalized multi-table relational package

Design choices:

- `5` disjoint families
- `23` persons per family
- `17` kinship relations
- only within-family candidate atoms are materialized, matching the paper-scale family count
- split columns include a random split plus paper-inspired evidence and family holdouts

The flat and relational versions are aligned at the `atom_id` level.
"""


def write_outputs(bundle: dict[str, object], flat_rows: list[dict[str, object]], output_root: Path, seed: int) -> None:
    relational_dir = output_root / "bouchard_family_relational"
    flat_dir = output_root / "bouchard_family_flat"
    ensure_dir(relational_dir)
    ensure_dir(flat_dir)

    write_csv(
        relational_dir / "families.csv",
        list(bundle["families"]),
        ["family_id", "family_name", "num_generations", "num_people"],
    )
    write_csv(
        relational_dir / "persons.csv",
        list(bundle["persons"]),
        [
            "person_id",
            "family_id",
            "display_name",
            "family_local_index",
            "sex",
            "generation",
            "branch_id",
            "role_code",
            "is_blood_relative_to_root",
        ],
    )
    write_csv(
        relational_dir / "marriages.csv",
        list(bundle["marriages"]),
        ["marriage_id", "family_id", "husband_id", "wife_id"],
    )
    write_csv(
        relational_dir / "parent_child_edges.csv",
        list(bundle["parent_child_edges"]),
        ["edge_id", "family_id", "parent_id", "child_id", "parent_role"],
    )
    write_csv(
        relational_dir / "relation_types.csv",
        list(bundle["relation_types"]),
        ["relation_id", "relation_name", "relation_group", "arity"],
    )
    write_csv(
        relational_dir / "pair_relation_atoms.csv",
        list(bundle["pair_relation_atoms"]),
        [
            "atom_id",
            "family_id",
            "relation_id",
            "relation",
            "source_person_id",
            "target_person_id",
            "label",
            "split",
            "split_random",
            "split_evidence_p10",
            "split_family_p10",
            "split_family_p00",
        ],
    )
    write_json(relational_dir / "schema.json", relational_schema())
    write_json(
        relational_dir / "metadata.json",
        {
            "dataset_name": "bouchard_family_relational",
            "random_seed": seed,
            "num_families": len(bundle["families"]),
            "num_people": len(bundle["persons"]),
            "num_relations": len(bundle["relation_types"]),
            "num_atoms": len(bundle["pair_relation_atoms"]),
            "notes": [
                "Within-family candidate atoms only; cross-family pairs are omitted.",
                "The family layout is paper-aligned at 23 persons per family.",
                "split_evidence_p10 keeps all core relations in train and samples 10% of derived atoms into train.",
                "split_family_p00 withholds all derived atoms from the fifth family from training.",
            ],
        },
    )
    (relational_dir / "validation_report.md").write_text(validate_relational(bundle), encoding="utf-8")

    write_csv(
        flat_dir / "family_kinship_flat.csv",
        flat_rows,
        [
            "atom_id",
            "dataset_name",
            "anchor_table",
            "event_time",
            "split",
            "split_random",
            "split_evidence_p10",
            "split_family_p10",
            "split_family_p00",
            "family_id",
            "relation_id",
            "relation",
            "relation_group",
            "label",
            "source_person_id",
            "source_local_index",
            "source_sex",
            "source_generation",
            "source_branch_id",
            "source_role_code",
            "source_is_blood_relative_to_root",
            "source_parent_count",
            "source_child_count",
            "source_spouse_count",
            "target_person_id",
            "target_local_index",
            "target_sex",
            "target_generation",
            "target_branch_id",
            "target_role_code",
            "target_is_blood_relative_to_root",
            "target_parent_count",
            "target_child_count",
            "target_spouse_count",
            "same_generation",
            "same_branch",
            "same_sex",
            "generation_gap_signed",
            "generation_gap_abs",
        ],
    )
    write_json(flat_dir / "schema.json", flat_schema())
    write_json(
        flat_dir / "metadata.json",
        {
            "dataset_name": "bouchard_family_flat",
            "random_seed": seed,
            "row_count": len(flat_rows),
            "target_column": "label",
            "event_time_column": "event_time",
            "primary_split_column": "split",
            "alternate_split_columns": ["split_evidence_p10", "split_family_p10", "split_family_p00"],
        },
    )
    (flat_dir / "validation_report.md").write_text(validate_flat(flat_rows), encoding="utf-8")
    (output_root / "README.md").write_text(build_readme(), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build flat and relational Bouchard-style family datasets.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets"),
        help="Directory that will receive the generated datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260419,
        help="Random seed for deterministic family generation and split assignment.",
    )
    parser.add_argument(
        "--num-families",
        type=int,
        default=5,
        help="Number of disjoint synthetic families to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_family_world(seed=args.seed, num_families=args.num_families)
    flat_rows = build_flat_table(bundle)
    write_outputs(bundle=bundle, flat_rows=flat_rows, output_root=args.output_root, seed=args.seed)
    print(f"Wrote datasets to {args.output_root}")


if __name__ == "__main__":
    main()
