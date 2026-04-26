# How To Check If The Bouchard Materials Make Sense

This file is a practical audit guide.

It answers:

- how do I know the dataset is not nonsense?
- how do I check grandmother and grandfather?
- how do I check the multiseed groups?
- how do I tell structure checking from logic checking?

## 1. There Are Different Kinds Of "Correct"

A dataset can be correct in more than one way.

### 1.1 Structure Correct

This means:

- files exist
- columns exist
- keys match
- there are no broken references

Example:

- every `parent_id` in `parent_child_edges.csv` must point to a real person in `persons.csv`

### 1.2 Logic Correct

This means:

- the family relations really follow from the family graph

Example:

- if a row says `grandmother(A, B) = 1`, then A really must be a female parent of a parent of B

### 1.3 Paper Faithful

This means:

- the dataset follows the paper's intended setup well enough

Example:

- `5` families
- `23` persons per family
- `17` relations
- family split logic exists

These three checks are related, but they are not the same thing.

## 2. The Fastest Sanity Checks

Read these files:

- [datasets/bouchard_family_relational/validation_report.md](../datasets/bouchard_family_relational/validation_report.md)
- [datasets/faithfulness_report.md](../datasets/docs/faithfulness_report.md)

The flat denormalized family export from the original working tree is not included in this curated package because the experiments use the relational triples.

These already tell you:

- person counts
- family counts
- atom counts
- split counts
- PK/FK violations
- whether the build matches the intended Bouchard shape

## 3. The Core Files To Inspect

If you want to check the logic yourself, focus on these:

- [datasets/bouchard_family_relational/persons.csv](../datasets/bouchard_family_relational/persons.csv)
- [datasets/bouchard_family_relational/marriages.csv](../datasets/bouchard_family_relational/marriages.csv)
- [datasets/bouchard_family_relational/parent_child_edges.csv](../datasets/bouchard_family_relational/parent_child_edges.csv)
- [datasets/bouchard_family_relational/pair_relation_atoms.csv](../datasets/bouchard_family_relational/pair_relation_atoms.csv)

Think of them like this:

- `persons.csv`: the cast
- `marriages.csv`: the couples
- `parent_child_edges.csv`: the parent arrows
- `pair_relation_atoms.csv`: the exam sheet

## 4. How To Check Grandmother And Grandfather

This is one of the best checks because it is simple and strict.

### 4.1 The Rule

For a positive `grandmother(g, c)` row:

- `g` must be female
- there must exist some `p` such that:
  - `g` is a parent of `p`
  - `p` is a parent of `c`

For a positive `grandfather(g, c)` row:

- `g` must be male
- there must exist some `p` such that:
  - `g` is a parent of `p`
  - `p` is a parent of `c`

### 4.2 The Reverse Check

For every two-step parent chain:

- grandparent sex tells you whether it should be `grandmother` or `grandfather`

So the check goes both ways:

- no false positives
- no missing positives

### 4.3 What You Should Compare

Build these from `parent_child_edges.csv`:

- `parents(child)`
- `grandparents(child)`

Then compare them to the positive rows in `pair_relation_atoms.csv` for:

- `grandmother`
- `grandfather`
- `grandson`
- `granddaughter`

If they match exactly, that part of the dataset is strong.

## 5. Why Grandmother And Grandfather Are A Good Audit

Because they are:

- not too easy
- not too hard
- directly tied to the parent graph
- easy to explain

If this part fails, something is badly wrong.

If this part works, it gives you real confidence that the generator is doing sensible kinship reasoning.

## 6. How To Check The Other Relations

### 6.1 Mother / Father

Check:

- every positive `mother(a, b)` means:
  - `a` is a parent of `b`
  - `a.sex = F`

- every positive `father(a, b)` means:
  - `a` is a parent of `b`
  - `a.sex = M`

### 6.2 Son / Daughter

Check:

- every positive `son(a, b)` means:
  - `a` is a child of `b`
  - `a.sex = M`

- every positive `daughter(a, b)` means:
  - `a` is a child of `b`
  - `a.sex = F`

### 6.3 Husband / Wife

Check:

- every `husband(a, b)` must have a matching `wife(b, a)`
- marriage pairs must be consistent with sex

### 6.4 Brother / Sister

Check:

- source and target share both parents
- source and target are not the same person
- source sex decides `brother` or `sister`

### 6.5 Uncle / Aunt

Check:

- source is sibling of one of target's parents
- source sex decides `uncle` or `aunt`

### 6.6 Nephew / Niece

Check:

- target is aunt/uncle of source
- source sex decides `nephew` or `niece`

### 6.7 Cousin

Check:

- one parent of source is sibling of one parent of target
- source and target are not siblings

## 7. How To Check The Split Logic

Look at the split columns in:

- [datasets/bouchard_family_relational/pair_relation_atoms.csv](../datasets/bouchard_family_relational/pair_relation_atoms.csv)

### 7.1 `split_random`

Check:

- rows are spread across train, val, test
- no weird missing split labels

### 7.2 `split_evidence_p10`

Check:

- all core relations are always `train`
- only derived relations are divided into train, val, test

### 7.3 `split_family_p10`

Check:

- all atoms from four families are in train
- for the held-out family:
  - core relations are train
  - only about 10% of derived rows go to train
  - the rest go to val or test

### 7.4 `split_family_p00`

Check:

- for the held-out family:
  - core relations are train
  - no derived rows are train

This is the hardest transfer setting.

## 8. How To Check The Flat Table

The flat file is:

- `datasets/bouchard_family_flat/family_kinship_flat.csv` in the original working tree; omitted here as redundant with the relational export.

You want to check:

- same number of rows as the relational atoms
- same `atom_id` values
- same labels
- no nonsense in the joined features

Fast check:

```bash
python3 check_flat_smoke_dataset.py \
  --flat-path datasets/bouchard_family_flat/family_kinship_flat.csv \
  --target label
```

This checks:

- row count
- split count
- target balance
- missingness
- whether train has both classes

## 9. How To Check The Multiseed Groups

Folder:

- `dataset_groups/` in the original working tree; omitted here to avoid duplicate generated datasets.

Read:

- `dataset_groups/collection_summary.csv` in the original working tree
- `dataset_groups/manifest.json` in the original working tree

Then check each group's:

- `group_manifest.json`

What should stay the same across groups:

- `5` families
- `115` people
- `17` relations
- `44,965` atoms

What should change across groups:

- exact positive counts for some relations
- exact family sex patterns
- exact local family topology details

If nothing changes across groups, the groups are suspicious.

If everything changes wildly, the generator may be unstable.

You want:

- same shape
- different realizations

## 10. What The Current Validation Is Good At

Right now the materials are strong at:

- shape checking
- count checking
- foreign key checking
- split checking
- family-size checking
- multiseed summary checking

This is good and useful.

## 11. What The Current Validation Is Not Yet Best At

The strongest remaining check would be an **independent logic validator**.

That means:

- do not trust the generator
- do not trust the saved atoms
- rebuild every relation from the graph alone
- compare rebuilt facts to saved facts

That is the gold standard.

Why?

Because it tests the output from the outside.

It is like checking a student's homework by solving the problem yourself.

## 12. The Best Audit Order

If you want a serious audit, use this order:

1. Check counts and PK/FK reports
2. Check one family manually
3. Check `mother` and `father`
4. Check `grandmother` and `grandfather`
5. Check `husband` and `wife`
6. Check `brother` and `sister`
7. Check `uncle`, `aunt`, `nephew`, `niece`
8. Check `cousin`
9. Check split columns
10. Check multiseed groups

This order moves from simple to harder logic.

## 13. What Would Make You Confident

You should feel strong confidence if:

- all PK/FK checks are zero-violation
- all families have `23` persons
- no cross-family atom mismatches exist
- grandparent relations match two-hop parent chains
- marriage relations are symmetric in the expected reverse form
- sibling relations match shared-parent logic
- split rules behave exactly as intended
- different seeds give different but similarly sized worlds

## 14. Bottom Line

The most important question is not:

- "Do the files look professional?"

The most important question is:

- "Can I recompute the logic from the graph and get the same answers?"

That is the real truth test.

If you want the next strongest step, I can build an independent validator script that recomputes:

- `grandmother`
- `grandfather`
- `grandson`
- `granddaughter`
- and then all 17 relations

from the graph and prints every mismatch.
