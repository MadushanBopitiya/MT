#!/usr/bin/env python3
"""
prepare_moduletype_data.py

Specialized preprocessor for the ModuleType modality.  Same pipeline as
prepare_tooltype_data.py but with:
  - The canonical 6-class ModuleType mapping embedded (no external JSON).
  - The tab10 colormap embedded (no external JSON).
  - Source and output paths hardcoded at the top of the file.

Per part:
    1. Read raw CSV  ->  {Sequence: module_name}
    2. Translate module_name -> canonical ModuleType ID (0..5).
    3. Read the AreaColor .idx and .ply (geometry + normals).
    4. Remap each point's Sequence -> ModuleType ID.
    5. Unit-sphere normalize XYZ.
    6. Write .xyz, .seg, .ply for the ModuleType modality.

Output (matches Phase 2 ThesisDatasetPhase2):
    <OUT>/
      xyz/   <part>_ModuleType_pointcloud_n<N>.xyz
      seg/   <part>_ModuleType_pointcloud_n<N>.seg
      ply/   <part>_ModuleType_pointcloud_n<N>.ply           (QA only)
      _global_mapping_ModuleType.json
      _class_inventory_ModuleType.{json,txt}
      _preprocessing_summary_ModuleType.json

Usage:
    python prepare_moduletype_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    sys.exit("ERROR: plyfile required.  Install with:  pip install plyfile")


# ============================================================================
# === EDIT THIS BLOCK ========================================================

SRC      = r"D:\MASTER THESIS\Data\raw\Machining_Tools"
OUT      = r"D:\MASTER THESIS\Data\processed\Industrial_Dataset\Module_Types\train"
N_POINTS = 4096

# ============================================================================
# === CANONICAL MODULETYPE DEFINITIONS (do not edit) =========================

LABEL_COLUMN = "ModuleType"

# Externally-fixed name -> ID assignment.  IDs match the dataset documentation.
NAME_TO_ID: dict = {
    "Drill2ax":    0,    # 2-axis drilling
    "DrillRev2ax": 1,    # 2-axis reverse drilling
    "MPath":       2,    # multi-path
    "Mill2ax":     3,    # 2-axis milling
    "Mill3ax":     4,    # 3-axis milling
    "Mill5ax":     5,    # 5-axis milling
}

# matplotlib tab10 colors, C0..C5 in order.  Used to color the QA PLY files.
COLORMAP: dict = {
    0: ( 31, 119, 180),   # tab10 C0  blue     Drill2ax
    1: (255, 127,  14),   # tab10 C1  orange   DrillRev2ax
    2: ( 44, 160,  44),   # tab10 C2  green    MPath
    3: (214,  39,  40),   # tab10 C3  red      Mill2ax
    4: (148, 103, 189),   # tab10 C4  purple   Mill3ax
    5: (140,  86,  75),   # tab10 C5  brown    Mill5ax
}
UNMAPPED_RGB = (128, 128, 128)


# ============================================================================
# === CSV helpers ============================================================
def find_raw_csv(part_dir: Path) -> Optional[Path]:
    csvs = list(part_dir.glob("*.csv"))
    raws = [p for p in csvs if not p.name.endswith("_encoded.csv")]
    if len(raws) != 1:
        return None
    return raws[0]


def read_raw_csv_seq_to_name(csv_path: Path, label_column: str) -> tuple[dict, int, bool]:
    """Read raw CSV -> {Sequence (int): module_name (str)}."""
    mapping: dict = {}
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fields = reader.fieldnames or []
        if label_column not in fields:
            return mapping, skipped, False
        for row in reader:
            seq_raw = (row.get("Sequence") or "").strip()
            name = (row.get(label_column) or "").strip()
            if not seq_raw or not name:
                skipped += 1
                continue
            try:
                seq = int(seq_raw)
            except ValueError:
                skipped += 1
                continue
            mapping[seq] = name
    return mapping, skipped, True


def is_valid_name(name: str) -> bool:
    """Filters out artifacts like '0', '123', '', whitespace-only strings."""
    return any(c.isalpha() for c in name)


def scan_for_unknown_names(part_dirs, label_column: str, known_names: set):
    """Walk parts to find module names that aren't in the canonical mapping."""
    discovered = set()
    notes = []
    for d in part_dirs:
        raw = find_raw_csv(d)
        if raw is None:
            csvs = list(d.glob("*.csv"))
            if len(csvs) == 0:
                notes.append((d, "no CSV"))
            else:
                notes.append((d, f"ambiguous CSVs: {[p.name for p in csvs]}"))
            continue
        try:
            mapping, _, col_present = read_raw_csv_seq_to_name(raw, label_column)
        except Exception as e:
            notes.append((d, f"read failed: {e}"))
            continue
        if not col_present:
            notes.append((d, f"column '{label_column}' missing in {raw.name}"))
            continue
        if not mapping:
            notes.append((d, f"no usable rows in {raw.name}"))
            continue
        discovered.update(v for v in mapping.values() if is_valid_name(v))
    unknown = sorted(discovered - known_names)
    return unknown, notes


# ============================================================================
# === Point cloud I/O ========================================================
def read_ply_xyz_normals(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    fields = v.data.dtype.names
    for req in ("x", "y", "z", "nx", "ny", "nz"):
        if req not in fields:
            raise ValueError(f"{ply_path}: missing property '{req}'.  Found: {fields}")
    return np.stack([v["x"], v["y"], v["z"], v["nx"], v["ny"], v["nz"]],
                    axis=1).astype(np.float32)


def read_idx(idx_path: Path) -> np.ndarray:
    return np.loadtxt(str(idx_path), dtype=np.int64)


# ============================================================================
# === Core transforms ========================================================
def remap_labels(seq_per_point: np.ndarray, seq_to_id: dict,
                 ignore_value: int = -1) -> tuple:
    out = np.full_like(seq_per_point, ignore_value)
    unmapped: list = []
    for seq in np.unique(seq_per_point):
        seq_int = int(seq)
        if seq_int == ignore_value:
            continue
        if seq_int in seq_to_id:
            out[seq_per_point == seq] = seq_to_id[seq_int]
        else:
            unmapped.append(seq_int)
    return out, unmapped


def unit_sphere_normalize(xyz_normals: np.ndarray) -> np.ndarray:
    out = xyz_normals.copy()
    xyz = out[:, :3]
    xyz -= xyz.mean(axis=0, keepdims=True)
    max_dist = np.linalg.norm(xyz, axis=1).max()
    if max_dist > 0:
        xyz /= max_dist
    out[:, :3] = xyz
    return out


def class_color(class_id: int) -> tuple:
    if class_id < 0:
        return UNMAPPED_RGB
    return COLORMAP.get(class_id, UNMAPPED_RGB)


# ============================================================================
# === Output writers =========================================================
def write_xyz(path: Path, xyz_normals: np.ndarray) -> None:
    np.savetxt(path, xyz_normals, fmt="%.6f")


def write_seg(path: Path, labels: np.ndarray) -> None:
    np.savetxt(path, labels, fmt="%d")


def write_ply(path: Path, xyz_normals: np.ndarray, labels: np.ndarray) -> None:
    rgb = np.array([class_color(int(l)) for l in labels], dtype=np.uint8)
    vertex = np.empty(
        len(xyz_normals),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = (
        xyz_normals[:, 0], xyz_normals[:, 1], xyz_normals[:, 2])
    vertex["nx"], vertex["ny"], vertex["nz"] = (
        xyz_normals[:, 3], xyz_normals[:, 4], xyz_normals[:, 5])
    vertex["red"], vertex["green"], vertex["blue"] = (
        rgb[:, 0], rgb[:, 1], rgb[:, 2])
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


# ============================================================================
# === Per-part driver ========================================================
def process_part(part_dir: Path, n_points: int, name_to_id: dict, out_dir: Path,
                 normalize: bool = True, write_ply_too: bool = True) -> dict:
    name = part_dir.name

    raw_csv = find_raw_csv(part_dir)
    if raw_csv is None:
        csvs = list(part_dir.glob("*.csv"))
        if len(csvs) == 0:
            return {"part": name, "status": "skip", "reason": "no CSV"}
        return {"part": name, "status": "error",
                "reason": f"ambiguous CSVs: {[p.name for p in csvs]}"}
    file_stem = raw_csv.stem

    vis_dir = part_dir / "visualizations"
    if not vis_dir.exists():
        return {"part": name, "status": "skip", "reason": "no visualizations/ folder"}

    suffix_ply = f"_AreaColor_pointcloud_n{n_points}.ply"
    suffix_idx = f"_AreaColor_pointcloud_n{n_points}.idx"
    ply_path = vis_dir / f"{file_stem}{suffix_ply}"
    idx_path = vis_dir / f"{file_stem}{suffix_idx}"

    if not ply_path.exists():
        ms = sorted(vis_dir.glob(f"*{suffix_ply}"))
        if len(ms) == 1:
            ply_path = ms[0]
        elif len(ms) == 0:
            return {"part": name, "status": "skip",
                    "reason": f"no PLY at n={n_points}"}
        else:
            return {"part": name, "status": "error",
                    "reason": f"multiple PLY candidates: {[p.name for p in ms]}"}

    if not idx_path.exists():
        ms = sorted(vis_dir.glob(f"*{suffix_idx}"))
        if len(ms) == 1:
            idx_path = ms[0]
        elif len(ms) == 0:
            return {"part": name, "status": "skip",
                    "reason": f"no IDX at n={n_points}"}
        else:
            return {"part": name, "status": "error",
                    "reason": f"multiple IDX candidates: {[p.name for p in ms]}"}

    try:
        seq_to_name, csv_skipped, col_present = read_raw_csv_seq_to_name(
            raw_csv, LABEL_COLUMN)
    except Exception as e:
        return {"part": name, "status": "error", "reason": f"CSV read: {e}"}
    if not col_present:
        return {"part": name, "status": "skip",
                "reason": f"column '{LABEL_COLUMN}' missing in {raw_csv.name}"}
    if not seq_to_name:
        return {"part": name, "status": "skip",
                "reason": f"no usable {LABEL_COLUMN} rows in {raw_csv.name}"}

    seq_to_id: dict = {}
    unknown_names: set = set()
    for seq, mod_name in seq_to_name.items():
        if mod_name in name_to_id:
            seq_to_id[seq] = name_to_id[mod_name]
        else:
            unknown_names.add(mod_name)

    try:
        xyz_normals = read_ply_xyz_normals(ply_path)
        seq_per_point = read_idx(idx_path)
    except Exception as e:
        return {"part": name, "status": "error", "reason": f"PLY/IDX: {e}"}

    if len(xyz_normals) != len(seq_per_point):
        return {"part": name, "status": "error",
                "reason": f"PLY/IDX length mismatch: "
                          f"{len(xyz_normals)} vs {len(seq_per_point)}"}

    labels, unmapped_seqs = remap_labels(seq_per_point, seq_to_id)

    if normalize:
        xyz_normals = unit_sphere_normalize(xyz_normals)

    xyz_dir = out_dir / "xyz"
    seg_dir = out_dir / "seg"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    if write_ply_too:
        ply_dir = out_dir / "ply"
        ply_dir.mkdir(parents=True, exist_ok=True)

    base = f"{name}_{LABEL_COLUMN}_pointcloud_n{n_points}"
    write_xyz(xyz_dir / f"{base}.xyz", xyz_normals)
    write_seg(seg_dir / f"{base}.seg", labels)
    if write_ply_too:
        write_ply(ply_dir / f"{base}.ply", xyz_normals, labels)

    valid = labels >= 0
    valid_labels = labels[valid]
    if len(valid_labels) > 0:
        unique_ids, counts = np.unique(valid_labels, return_counts=True)
        per_class_points = {int(i): int(c) for i, c in zip(unique_ids, counts)}
    else:
        per_class_points = {}

    return {
        "part": name,
        "status": "ok",
        "n_points": int(len(labels)),
        "n_valid": int(valid.sum()),
        "n_unmapped_points": int((~valid).sum()),
        "unique_labels": sorted(per_class_points.keys()),
        "per_class_points": per_class_points,
        "unknown_names": sorted(unknown_names),
        "unmapped_sequences": unmapped_seqs,
        "csv_rows_skipped": csv_skipped,
        "raw_csv": raw_csv.name,
    }


# ============================================================================
# === Main ===================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-normalize", action="store_true",
                    help="Skip the unit-sphere normalization step.")
    ap.add_argument("--no-ply", action="store_true",
                    help="Skip writing the QA .ply files.")
    ap.add_argument("--parts", nargs="*", default=None,
                    help="Only process these part folder names (subset of SRC).")
    ap.add_argument("--drop-parts", nargs="*", default=None,
                    help="Folder names to skip entirely.")
    args = ap.parse_args()

    src = Path(SRC).expanduser().resolve()
    out = Path(OUT).expanduser().resolve()

    if not src.is_dir():
        sys.exit(f"ERROR: SRC not found: {src}")

    if args.parts:
        part_dirs = [src / p for p in args.parts]
    else:
        part_dirs = sorted(p for p in src.iterdir() if p.is_dir())

    drop_set = set(args.drop_parts or [])
    if drop_set:
        before = len(part_dirs)
        part_dirs = [p for p in part_dirs if p.name not in drop_set]
        print(f"Dropped {before - len(part_dirs)} part(s) via --drop-parts: "
              f"{sorted(drop_set)}")

    print(f"SRC          : {src}")
    print(f"OUT          : {out}")
    print(f"Label column : {LABEL_COLUMN}")
    print(f"N points     : {N_POINTS}")
    print(f"Normalize    : {not args.no_normalize}")
    print(f"Write PLY    : {not args.no_ply}")
    print(f"Parts found  : {len(part_dirs)}")
    print(f"Canonical mapping ({len(NAME_TO_ID)} classes):")
    for n, i in sorted(NAME_TO_ID.items(), key=lambda x: x[1]):
        rgb = COLORMAP.get(i, UNMAPPED_RGB)
        print(f"    {i}  {n:<14}  RGB={rgb}  hex=#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")

    # Sanity scan: find any names in CSVs that aren't in the canonical mapping
    print("-" * 70)
    print(f"Scanning raw CSVs for {LABEL_COLUMN} values not in canonical mapping...")
    unknown, scan_notes = scan_for_unknown_names(part_dirs, LABEL_COLUMN,
                                                  known_names=set(NAME_TO_ID.keys()))
    if unknown:
        print(f"  WARNING: {len(unknown)} name(s) in data NOT in canonical mapping "
              f"(will be labeled -1): {unknown}")
    else:
        print("  All discovered names match the canonical mapping.")
    if scan_notes:
        print(f"  Parts contributing nothing: {len(scan_notes)}")
        for d, msg in scan_notes:
            print(f"    - {d.name}: {msg}")

    out.mkdir(parents=True, exist_ok=True)

    # Save the canonical mapping (output identical in format to the ToolType script)
    mapping_path = out / f"_global_mapping_{LABEL_COLUMN}.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "label": LABEL_COLUMN,
            "n_classes": len(NAME_TO_ID),
            "name_to_id": NAME_TO_ID,
            "id_to_name": {v: k for k, v in NAME_TO_ID.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"  Mapping saved to: {mapping_path}")

    # ----- main pass -----
    print("-" * 70)
    results = []
    for d in part_dirs:
        r = process_part(d, n_points=N_POINTS, name_to_id=NAME_TO_ID, out_dir=out,
                         normalize=not args.no_normalize,
                         write_ply_too=not args.no_ply)
        results.append(r)
        if r["status"] == "ok":
            extras = []
            if r["unmapped_sequences"]:
                extras.append(f"unmapped seqs: {r['unmapped_sequences']}")
            if r["unknown_names"]:
                extras.append(f"names not in mapping: {r['unknown_names']}")
            if r.get("csv_rows_skipped", 0) > 0:
                extras.append(f"{r['csv_rows_skipped']} CSV rows skipped")
            extra = ("   " + " | ".join(extras)) if extras else ""
            print(f"  v {r['part']:<45}  classes={r['unique_labels']}  "
                  f"valid={r['n_valid']}/{r['n_points']}{extra}")
        else:
            print(f"  x {r['part']:<45}  [{r['status']}] {r['reason']}")

    ok = [r for r in results if r["status"] == "ok"]
    print("-" * 70)
    print(f"Processed successfully : {len(ok)} / {len(results)}")
    if ok:
        all_labels = sorted({l for r in ok for l in r["unique_labels"]})
        total_pts = sum(r["n_points"] for r in ok)
        total_unmapped = sum(r["n_unmapped_points"] for r in ok)
        print(f"Classes appearing in data : {all_labels}  "
              f"({len(all_labels)} of {len(NAME_TO_ID)} possible)")
        print(f"Total points              : {total_pts:,}")
        print(f"Unmapped points           : {total_unmapped:,}  "
              f"({100 * total_unmapped / total_pts:.2f} %)")

    summary_path = out / f"_preprocessing_summary_{LABEL_COLUMN}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary written to        : {summary_path}")

    # ----- inventory -----
    points_per_class: dict = {}
    parts_per_class: dict = {}
    for r in ok:
        for cid, n in r.get("per_class_points", {}).items():
            points_per_class[cid] = points_per_class.get(cid, 0) + n
            parts_per_class[cid] = parts_per_class.get(cid, 0) + 1

    id_to_name = {v: k for k, v in NAME_TO_ID.items()}
    rows = []
    for cid in sorted(id_to_name.keys()):
        r_rgb = COLORMAP.get(cid, UNMAPPED_RGB)
        rows.append({
            "id": cid,
            "name": id_to_name[cid],
            "rgb": list(r_rgb),
            "hex": "#{:02X}{:02X}{:02X}".format(*r_rgb),
            "n_points": points_per_class.get(cid, 0),
            "n_parts": parts_per_class.get(cid, 0),
        })
    rows.sort(key=lambda r: -r["n_points"])

    inv_json = {
        "modality": LABEL_COLUMN,
        "n_classes": len(rows),
        "color_source": "matplotlib tab10 (C0..C5), hardcoded in script",
        "classes": rows,
    }
    inv_json_path = out / f"_class_inventory_{LABEL_COLUMN}.json"
    with open(inv_json_path, "w", encoding="utf-8") as f:
        json.dump(inv_json, f, indent=2, ensure_ascii=False)

    name_w = max((len(r["name"]) for r in rows), default=10)
    name_w = max(name_w, 4)
    header = (f"{'ID':>4}  {'Name':<{name_w}}  "
              f"{'Hex':<8}  {'RGB':<15}  "
              f"{'Points':>12}  {'Parts':>6}")
    lines = [
        f"Modality: {LABEL_COLUMN}   |   {len(rows)} classes   "
        f"|   {len(ok)} parts processed",
        f"Colors  : {inv_json['color_source']}",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for r in rows:
        rgb_str = f"({r['rgb'][0]},{r['rgb'][1]},{r['rgb'][2]})"
        lines.append(
            f"{r['id']:>4}  {r['name']:<{name_w}}  "
            f"{r['hex']:<8}  {rgb_str:<15}  "
            f"{r['n_points']:>12,}  {r['n_parts']:>6}")
    inv_txt_path = out / f"_class_inventory_{LABEL_COLUMN}.txt"
    with open(inv_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Class inventory           : {inv_json_path}")
    print(f"                          : {inv_txt_path}")


if __name__ == "__main__":
    main()
