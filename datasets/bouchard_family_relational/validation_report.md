# Validation report: bouchard_family_relational

## Row counts
- `families`: 5
- `persons`: 115
- `marriages`: 50
- `parent_child_edges`: 120
- `relation_types`: 17
- `pair_relation_atoms`: 44,965

## Primary key checks
- `families.family_id` duplicates: 0
- `persons.person_id` duplicates: 0
- `marriages.marriage_id` duplicates: 0
- `parent_child_edges.edge_id` duplicates: 0
- `relation_types.relation_id` duplicates: 0
- `pair_relation_atoms.atom_id` duplicates: 0

## Foreign key checks
- `persons.family_id -> families.family_id` violations: 0
- `marriages.family_id -> families.family_id` violations: 0
- `marriages.husband_id -> persons.person_id` violations: 0
- `marriages.wife_id -> persons.person_id` violations: 0
- `parent_child_edges.family_id -> families.family_id` violations: 0
- `parent_child_edges.parent_id -> persons.person_id` violations: 0
- `parent_child_edges.child_id -> persons.person_id` violations: 0
- `pair_relation_atoms.family_id -> families.family_id` violations: 0
- `pair_relation_atoms.relation_id -> relation_types.relation_id` violations: 0
- `pair_relation_atoms.source_person_id -> persons.person_id` violations: 0
- `pair_relation_atoms.target_person_id -> persons.person_id` violations: 0

## Semantic checks
- families with `23` persons: 5 / 5
- cross-family atom mismatches: 0
- expected within-family atom count (`5 x 23 x 23 x 17`): 44,965
- actual atom count: 44,965
- split_random counts: {'test': 4482, 'train': 35980, 'val': 4503}
- split_evidence_p10 counts: {'test': 27499, 'train': 14023, 'val': 3443}
- split_family_p10 counts: {'test': 5499, 'train': 38777, 'val': 689}
- split_family_p00 counts: {'test': 6188, 'train': 38088, 'val': 689}
