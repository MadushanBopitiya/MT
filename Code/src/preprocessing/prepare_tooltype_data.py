#!/usr/bin/env python3
"""
prepare_tooltype_data.py

Converts AreaColor-labeled point clouds into target-modality-labeled point
clouds in the format expected by Phase 1 ThesisDataset (.xyz + .seg), plus
an RGB-colored .ply for visual QA.

Per part:
    1. Read the raw CSV  ->  {Sequence: tool_name}
    2. Translate tool_name -> global integer ID using a mapping built in a
       pre-pass over ALL parts.  This bypasses the *_encoded.csv files,
       which in this dataset use inconsistent per-part integer assignments
       and cannot be used as a global vocabulary.
    3. Read the AreaColor .idx and .ply.
    4. Remap each point's Sequence -> global tool ID.
    5. Unit-sphere normalize XYZ.
    6. Write .xyz, .seg, .ply for the target modality.

Output layout (matches Phase 1 ThesisDataset):
    <out>/
      xyz/   <part>_<Modality>_pointcloud_n<N>.xyz
      seg/   <part>_<Modality>_pointcloud_n<N>.seg
      ply/   <part>_<Modality>_pointcloud_n<N>.ply           (QA only)
      _global_mapping_<Modality>.json
      _class_inventory_<Modality>.{json,txt}
      _preprocessing_summary_<Modality>.json

Example:
    python prepare_tooltype_data.py \
        --root  /data/raw/Machining_Tools \
        --out   /data/processed/Machining_Tools/train \
        --label ToolType \
        --n-points 4096
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


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def find_raw_csv(part_dir: Path) -> Optional[Path]:
    """Return the single raw CSV in part_dir (any *.csv NOT ending with
    _encoded.csv).  Returns None if zero or >1 candidates.
    """
    csvs = list(part_dir.glob("*.csv"))
    raws = [p for p in csvs if not p.name.endswith("_encoded.csv")]
    if len(raws) != 1:
        return None
    return raws[0]


def read_raw_csv_seq_to_name(
    csv_path: Path, label_column: str
) -> tuple[dict, int, bool]:
    """Read raw CSV -> {Sequence (int): tool_name (str)}.

    Returns (mapping, skipped_rows, column_present).
    Blank rows and rows with empty Sequence or label cells are skipped.
    """
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


def build_global_label_mapping(
    part_dirs, label_column: str
) -> tuple[dict, list]:
    """Walk every part, collect every distinct value in label_column from
    the raw CSV, assign each a stable integer ID (alphabetical).

    Values whose string contains no alphabetic characters (e.g. literal
    "0", "", "123") are treated as invalid and excluded.  This is a
    defence against corrupt CSV cells; legitimate tool names always
    contain letters.

    Returns (name_to_id, notes) where notes is a list of (part_dir, message)
    pairs describing parts that contributed nothing.
    """
    all_names: set = set()
    notes: list = []
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
        valid_names = {v for v in mapping.values() if is_valid_tool_name(v)}
        all_names.update(valid_names)
    name_to_id = {name: i for i, name in enumerate(sorted(all_names))}
    return name_to_id, notes


def is_valid_tool_name(name: str) -> bool:
    """A legitimate tool name contains at least one alphabetic character.
    Filters out artifacts like '0', '123', '', whitespace-only strings.
    """
    return any(c.isalpha() for c in name)


# ---------------------------------------------------------------------------
# Point cloud I/O
# ---------------------------------------------------------------------------
def read_ply_xyz_normals(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    fields = v.data.dtype.names
    for req in ("x", "y", "z", "nx", "ny", "nz"):
        if req not in fields:
            raise ValueError(
                f"{ply_path}: missing property '{req}'.  Found: {fields}"
            )
    return np.stack(
        [v["x"], v["y"], v["z"], v["nx"], v["ny"], v["nz"]], axis=1
    ).astype(np.float32)


def read_idx(idx_path: Path) -> np.ndarray:
    return np.loadtxt(str(idx_path), dtype=np.int64)


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------
def remap_labels(
    seq_per_point: np.ndarray,
    seq_to_id: dict,
    ignore_value: int = -1,
) -> tuple:
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


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
TAB20_RGB = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]
UNMAPPED_RGB = (128, 128, 128)


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def class_color(class_id: int, colormap=None) -> tuple:
    if class_id < 0:
        return UNMAPPED_RGB
    if colormap is not None and class_id in colormap:
        return colormap[class_id]
    return TAB20_RGB[class_id % len(TAB20_RGB)]


def load_colormap(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict = {}
    for k, v in raw.items():
        try:
            cid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, str) and v.startswith("#"):
            out[cid] = _hex_to_rgb(v)
        elif isinstance(v, (list, tuple)) and len(v) == 3:
            out[cid] = (int(v[0]), int(v[1]), int(v[2]))
        elif isinstance(v, dict):
            if ("rgb" in v and isinstance(v["rgb"], (list, tuple))
                    and len(v["rgb"]) == 3):
                out[cid] = (int(v["rgb"][0]), int(v["rgb"][1]),
                            int(v["rgb"][2]))
            elif "hex" in v and isinstance(v["hex"], str):
                out[cid] = _hex_to_rgb(v["hex"])
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_xyz(path: Path, xyz_normals: np.ndarray) -> None:
    np.savetxt(path, xyz_normals, fmt="%.6f")


def write_seg(path: Path, labels: np.ndarray) -> None:
    np.savetxt(path, labels, fmt="%d")


def write_ply(path: Path, xyz_normals: np.ndarray,
              labels: np.ndarray, colormap=None) -> None:
    rgb = np.array(
        [class_color(int(l), colormap) for l in labels], dtype=np.uint8
    )
    vertex = np.empty(
        len(xyz_normals),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = (
        xyz_normals[:, 0], xyz_normals[:, 1], xyz_normals[:, 2]
    )
    vertex["nx"], vertex["ny"], vertex["nz"] = (
        xyz_normals[:, 3], xyz_normals[:, 4], xyz_normals[:, 5]
    )
    vertex["red"], vertex["green"], vertex["blue"] = (
        rgb[:, 0], rgb[:, 1], rgb[:, 2]
    )
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


# ---------------------------------------------------------------------------
# Per-part driver
# ---------------------------------------------------------------------------
def process_part(part_dir: Path, n_points: int, label_column: str,
                 name_to_id: dict, out_dir: Path, normalize: bool = True,
                 write_ply_too: bool = True, colormap=None) -> dict:
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
        return {"part": name, "status": "skip",
                "reason": "no visualizations/ folder"}

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
            raw_csv, label_column
        )
    except Exception as e:
        return {"part": name, "status": "error", "reason": f"CSV read: {e}"}
    if not col_present:
        return {"part": name, "status": "skip",
                "reason": f"column '{label_column}' missing in {raw_csv.name}"}
    if not seq_to_name:
        return {"part": name, "status": "skip",
                "reason": f"no usable {label_column} rows in {raw_csv.name}"}

    seq_to_id: dict = {}
    unknown_names: set = set()
    for seq, tool_name in seq_to_name.items():
        if tool_name in name_to_id:
            seq_to_id[seq] = name_to_id[tool_name]
        else:
            unknown_names.add(tool_name)

    try:
        xyz_normals = read_ply_xyz_normals(ply_path)
        seq_per_point = read_idx(idx_path)
    except Exception as e:
        return {"part": name, "status": "error", "reason": f"PLY/IDX: {e}"}

    if len(xyz_normals) != len(seq_per_point):
        return {"part": name, "status": "error",
                "reason": (f"PLY/IDX length mismatch: "
                           f"{len(xyz_normals)} vs {len(seq_per_point)}")}

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

    base = f"{name}_{label_column}_pointcloud_n{n_points}"
    write_xyz(xyz_dir / f"{base}.xyz", xyz_normals)
    write_seg(seg_dir / f"{base}.seg", labels)
    if write_ply_too:
        write_ply(ply_dir / f"{base}.ply", xyz_normals, labels, colormap)

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--label", default="ToolType",
                    choices=["ToolType", "ModuleType",
                             "FunctionType", "StrategyType"])
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--no-ply", action="store_true")
    ap.add_argument("--colormap", type=Path, default=None)
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--drop-parts", nargs="*", default=None,
                    help="Folder names to skip entirely (e.g. b06_kon0220w026-g01_p2).")
    args = ap.parse_args()

    if args.parts:
        part_dirs = [args.root / p for p in args.parts]
    else:
        part_dirs = sorted(p for p in args.root.iterdir() if p.is_dir())

    # Apply explicit drop list, if any.
    drop_set = set(args.drop_parts or [])
    if drop_set:
        before = len(part_dirs)
        part_dirs = [p for p in part_dirs if p.name not in drop_set]
        print(f"Dropped {before - len(part_dirs)} part(s) via --drop-parts: "
              f"{sorted(drop_set)}")

    print(f"Root         : {args.root}")
    print(f"Output       : {args.out}")
    print(f"Label column : {args.label}")
    print(f"N points     : {args.n_points}")
    print(f"Normalize    : {not args.no_normalize}")
    print(f"Write PLY    : {not args.no_ply}")
    print(f"Parts found  : {len(part_dirs)}")

    colormap = None
    if args.colormap is not None:
        if not args.colormap.exists():
            sys.exit(f"ERROR: --colormap file not found: {args.colormap}")
        try:
            colormap = load_colormap(args.colormap)
            print(f"Colormap     : {args.colormap}  ({len(colormap)} entries)")
        except Exception as e:
            sys.exit(f"ERROR: failed to parse --colormap: {e}")
    else:
        print("Colormap     : tab20 (default, indexed by global class ID)")

    # ----- pre-pass: build global mapping from raw CSVs -----
    print("-" * 70)
    print(f"Pre-scanning raw CSVs to build global {args.label} -> ID mapping...")
    name_to_id, scan_notes = build_global_label_mapping(part_dirs, args.label)
    print(f"  Unique {args.label} values found: {len(name_to_id)}")
    for n, i in sorted(name_to_id.items(), key=lambda x: x[1]):
        print(f"    {i:>3}  {n}")
    if scan_notes:
        print(f"  Parts contributing nothing to mapping: {len(scan_notes)}")
        for d, msg in scan_notes:
            print(f"    - {d.name}: {msg}")

    args.out.mkdir(parents=True, exist_ok=True)
    mapping_path = args.out / f"_global_mapping_{args.label}.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "label": args.label,
            "n_classes": len(name_to_id),
            "name_to_id": name_to_id,
            "id_to_name": {v: k for k, v in name_to_id.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"  Mapping saved to: {mapping_path}")

    if not name_to_id:
        sys.exit("ERROR: global mapping is empty -- no usable raw CSV data found.")

    # ----- main pass -----
    print("-" * 70)
    results = []
    for d in part_dirs:
        r = process_part(d, n_points=args.n_points, label_column=args.label,
                         name_to_id=name_to_id, out_dir=args.out,
                         normalize=not args.no_normalize,
                         write_ply_too=not args.no_ply, colormap=colormap)
        results.append(r)
        if r["status"] == "ok":
            extras = []
            if r["unmapped_sequences"]:
                extras.append(f"unmapped seqs: {r['unmapped_sequences']}")
            if r["unknown_names"]:
                extras.append(f"names not in global map: {r['unknown_names']}")
            if r.get("csv_rows_skipped", 0) > 0:
                extras.append(f"{r['csv_rows_skipped']} CSV rows skipped")
            extra = ("   " + " | ".join(extras)) if extras else ""
            print(f"  v {r['part']:<45}  classes={r['unique_labels']}  "
                  f"valid={r['n_valid']}/{r['n_points']}{extra}")
            print(f"      |- {d}")
        else:
            print(f"  x {r['part']:<45}  [{r['status']}] {r['reason']}")
            print(f"      |- {d}")

    ok = [r for r in results if r["status"] == "ok"]
    print("-" * 70)
    print(f"Processed successfully : {len(ok)} / {len(results)}")
    if ok:
        all_labels = sorted({l for r in ok for l in r["unique_labels"]})
        total_pts = sum(r["n_points"] for r in ok)
        total_unmapped = sum(r["n_unmapped_points"] for r in ok)
        print(f"Classes appearing in data : {all_labels}  "
              f"({len(all_labels)} of {len(name_to_id)} possible)")
        print(f"Total points              : {total_pts:,}")
        print(f"Unmapped points           : {total_unmapped:,}  "
              f"({100 * total_unmapped / total_pts:.2f} %)")

    summary_path = args.out / f"_preprocessing_summary_{args.label}.json"
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

    id_to_name = {v: k for k, v in name_to_id.items()}
    rows = []
    for cid in sorted(id_to_name.keys()):
        r_rgb = class_color(cid, colormap)
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
        "modality": args.label,
        "n_classes": len(rows),
        "color_source": (str(args.colormap) if args.colormap is not None
                         else "tab20 (default, indexed by global class ID)"),
        "classes": rows,
    }
    inv_json_path = args.out / f"_class_inventory_{args.label}.json"
    with open(inv_json_path, "w", encoding="utf-8") as f:
        json.dump(inv_json, f, indent=2, ensure_ascii=False)

    name_w = max((len(r["name"]) for r in rows), default=10)
    name_w = max(name_w, 4)
    header = (f"{'ID':>4}  {'Name':<{name_w}}  "
              f"{'Hex':<8}  {'RGB':<15}  "
              f"{'Points':>12}  {'Parts':>6}")
    lines = [
        f"Modality: {args.label}   |   {len(rows)} classes   "
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
            f"{r['n_points']:>12,}  {r['n_parts']:>6}"
        )
    inv_txt_path = args.out / f"_class_inventory_{args.label}.txt"
    with open(inv_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Class inventory           : {inv_json_path}")
    print(f"                          : {inv_txt_path}")


if __name__ == "__main__":
    main()