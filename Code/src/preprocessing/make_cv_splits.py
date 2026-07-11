#!/usr/bin/env python3
"""
make_cv_splits.py

Greedy multilabel-aware stratified k-fold CV splitter for the
preprocessed Machining_Tools dataset.

Splits parts into K folds such that each part appears in exactly one
fold's test set and in K-1 folds' train sets.  Each part is one file
(no augmented variants on disk), so splitting at the part level and at
the file level are the same thing.

Stratification strategy:
    1. Compute class popularity = how many parts contain each class.
    2. Compute per-part rarity score = sum of 1/popularity over the
       part's classes.  Parts with rare classes have higher scores.
    3. Process parts in descending rarity (rarest classes first).
    4. For each part, assign it to the fold that currently has the
       lowest count of its classes.  Tiebreak by fold size.

This spreads rare classes across folds as evenly as possible while
keeping common classes balanced.  Classes that appear in fewer than K
parts will inevitably be missing from training in some folds -- that's
structural, not a bug, and the splitter reports it transparently.

Inputs:
    <root>/seg/*.seg     - per-point labels (this is what determines class membership)
    <root>/_global_mapping_<label>.json  - optional, used for class names in the report

Outputs (under <root>/splits/):
    fold0_train.txt, fold0_test.txt, ..., foldK-1_test.txt
        One part stem per line (no path, no extension).
    _split_summary.json
        Machine-readable summary.
    _split_summary.txt
        Human-readable coverage table.

Example:
    python make_cv_splits.py \
        --in "D:/MASTER THESIS/Data/processed/Machining_Tools/train" \
        --k 5 --seed 0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def get_part_classes(seg_path):
    """Class IDs present in a .seg file (excluding -1 unmapped)."""
    labels = np.loadtxt(seg_path, dtype=np.int64)
    return {int(c) for c in np.unique(labels) if c >= 0}


def load_class_names(in_dir):
    """Read class id->name from _global_mapping_*.json if available."""
    for mapping_file in in_dir.glob("_global_mapping_*.json"):
        try:
            with open(mapping_file, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if isinstance(d, dict):
            if "name_to_id" in d and isinstance(d["name_to_id"], dict):
                return {int(v): k for k, v in d["name_to_id"].items()}
            if "id_to_name" in d and isinstance(d["id_to_name"], dict):
                return {int(k): v for k, v in d["id_to_name"].items()}
            try:
                return {int(v): k for k, v in d.items() if isinstance(v, int)}
            except Exception:
                pass
    return {}


def greedy_stratified_split(part_classes, k, rng):
    """Greedy multilabel-aware assignment.  Returns {part_stem: fold_idx}."""
    class_popularity = Counter()
    for cs in part_classes.values():
        for c in cs:
            class_popularity[c] += 1

    rarity = {}
    for stem, cs in part_classes.items():
        rarity[stem] = sum(1.0 / class_popularity[c] for c in cs) if cs else 0.0

    keyed = sorted(part_classes.keys(), key=lambda s: (-rarity[s], rng.random()))

    fold_class_counts = [Counter() for _ in range(k)]
    fold_sizes = [0] * k
    assignments = {}

    for stem in keyed:
        cs = part_classes[stem]
        best_fold = 0
        best_cost = float("inf")
        for f in range(k):
            cost = sum(fold_class_counts[f][c] for c in cs)
            if cost < best_cost or (
                cost == best_cost and fold_sizes[f] < fold_sizes[best_fold]
            ):
                best_cost = cost
                best_fold = f
        assignments[stem] = best_fold
        for c in cs:
            fold_class_counts[best_fold][c] += 1
        fold_sizes[best_fold] += 1

    return assignments


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="in_dir", type=Path, required=True,
                    help="Root containing seg/ subfolder.")
    ap.add_argument("--k", type=int, default=5, help="Number of folds (default 5).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = ap.parse_args()

    seg_dir = args.in_dir / "seg"
    if not seg_dir.is_dir():
        raise SystemExit(f"ERROR: {seg_dir} not found.")

    rng = np.random.default_rng(args.seed)

    seg_files = sorted(seg_dir.glob("*.seg"))
    if not seg_files:
        raise SystemExit(f"ERROR: no .seg files in {seg_dir}.")

    part_classes = {}
    for sf in seg_files:
        part_classes[sf.stem] = get_part_classes(sf)

    class_names = load_class_names(args.in_dir)
    n_parts = len(part_classes)
    all_classes = sorted({c for cs in part_classes.values() for c in cs})
    n_classes = len(all_classes)

    assignments = greedy_stratified_split(part_classes, args.k, rng)

    splits_dir = args.in_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for f in range(args.k):
        test_stems = sorted([s for s, fold in assignments.items() if fold == f])
        train_stems = sorted([s for s, fold in assignments.items() if fold != f])
        (splits_dir / f"fold{f}_train.txt").write_text(
            "\n".join(train_stems) + "\n", encoding="utf-8"
        )
        (splits_dir / f"fold{f}_test.txt").write_text(
            "\n".join(test_stems) + "\n", encoding="utf-8"
        )

    class_popularity = Counter()
    for cs in part_classes.values():
        for c in cs:
            class_popularity[c] += 1

    coverage = {}
    for c in all_classes:
        coverage[c] = []
        for f in range(args.k):
            train_n = sum(1 for s, fold in assignments.items()
                          if fold != f and c in part_classes[s])
            test_n = sum(1 for s, fold in assignments.items()
                         if fold == f and c in part_classes[s])
            coverage[c].append((train_n, test_n))

    fold_sizes = [sum(1 for fold in assignments.values() if fold == f)
                  for f in range(args.k)]

    warnings = []
    for c in all_classes:
        for f in range(args.k):
            if coverage[c][f][0] == 0:
                warnings.append(
                    f"Class {c} ({class_names.get(c, '?')}) MISSING from "
                    f"training in fold {f}."
                )

    summary = {
        "k": args.k,
        "seed": args.seed,
        "n_parts": n_parts,
        "n_classes": n_classes,
        "fold_sizes": {f"fold{f}": fold_sizes[f] for f in range(args.k)},
        "class_popularity": {
            f"id_{c}_{class_names.get(c, 'unknown')}": class_popularity[c]
            for c in all_classes
        },
        "coverage": {
            f"id_{c}_{class_names.get(c, 'unknown')}": {
                f"fold{f}": {"train_parts": tr, "test_parts": te}
                for f, (tr, te) in enumerate(coverage[c])
            }
            for c in all_classes
        },
        "warnings": warnings,
    }
    (splits_dir / "_split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    name_w = max(
        (len(class_names.get(c, f"class_{c}")) for c in all_classes),
        default=12,
    )
    name_w = max(name_w, 12)

    lines = []
    lines.append(f"{args.k}-fold CV split summary")
    lines.append("=" * len(lines[0]))
    lines.append(f"Total parts    : {n_parts}")
    lines.append(f"Total classes  : {n_classes}")
    lines.append(f"Seed           : {args.seed}")
    lines.append("")
    lines.append("Fold sizes (test partition):")
    for f in range(args.k):
        lines.append(
            f"  Fold {f}:  {fold_sizes[f]} test parts, "
            f"{n_parts - fold_sizes[f]} train parts"
        )
    lines.append("")
    lines.append("Per-class coverage (train_parts t / test_parts e), per fold:")
    lines.append("")

    header = f"  {'ID':>3}  {'Class':<{name_w}}  {'Total':>5}"
    for f in range(args.k):
        header += f"   {'F'+str(f):>7}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for c in all_classes:
        name = class_names.get(c, f"class_{c}")
        line = f"  {c:>3}  {name:<{name_w}}  {class_popularity[c]:>5}"
        for f in range(args.k):
            tr, te = coverage[c][f]
            marker = "*" if tr == 0 else " "
            line += f"   {tr:>2}t/{te}e{marker}"
        lines.append(line)

    lines.append("")
    if warnings:
        lines.append(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            lines.append(f"  * {w}")
        lines.append("")
        lines.append("(* marks (train, fold) cells where the class is missing from training.)")
    else:
        lines.append("All classes present in training for every fold. OK.")

    text = "\n".join(lines)
    (splits_dir / "_split_summary.txt").write_text(text, encoding="utf-8")

    print(text)
    print(f"\nSplit files: {splits_dir}/fold{{0..{args.k - 1}}}_{{train,test}}.txt")


if __name__ == "__main__":
    main()
