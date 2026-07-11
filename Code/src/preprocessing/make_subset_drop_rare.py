"""
make_subset_drop_rare.py

Creates a new copy of the preprocessed Machining_Tools dataset with rare
classes masked to -1 in the .seg files, and regenerates .ply files so
dropped points render gray (signaling "ignored") rather than their
original class color.

Edit the variables in the EDIT THIS BLOCK section below, then run:

    python make_subset_drop_rare.py

The script DOES NOT modify the source directory.  It writes a fresh
copy to DST (which must be new or empty).  After running, point your
slurm_phase2.sh's DATA_ROOT to the new path.

Class IDs are NOT renumbered.  Anbohrer stays 0, Torusfraeser stays 12,
etc.  This keeps the colormap, existing checkpoints, and the model's
17-output head compatible.  The dropped classes simply have no positive
labels in the new data.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

# ============================================================================
# === EDIT THIS BLOCK ========================================================

SRC       = r"D:\MASTER THESIS\Data\processed\Machining_Tools\Industrial_Dataset\train"
DST       = r"D:\MASTER THESIS\Data\processed\Machining_Tools\Industrial_Dataset_drop10\train"
MIN_PARTS = 10                  # drop classes with fewer than this many parts

# Optional explicit override.  If non-empty, used INSTEAD of MIN_PARTS.
#   DROP_IDS = [1, 3, 5, 6, 8, 10, 11, 13, 14, 15, 16]
DROP_IDS  = []

# Colormap JSON for PLY recoloring.  Searched relative to SRC first,
# then alongside this script, then as an absolute path.  Set to "" to
# use matplotlib's tab20 fallback automatically.
COLORMAP_PATH = "tooltype_colormap.json"

# ============================================================================


def find_inventory(src_dir):
    candidates = list(Path(src_dir).glob("_class_inventory_*.json"))
    return candidates[0] if candidates else None


def find_mapping(src_dir):
    candidates = list(Path(src_dir).glob("_global_mapping_*.json"))
    return candidates[0] if candidates else None


def load_inventory(path):
    """Normalize inventory to list of {id, name, part_count, point_count}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "classes" in data:
        entries = data["classes"]
    else:
        entries = []
        for k, v in data.items():
            entry = dict(v) if isinstance(v, dict) else {"name": str(v)}
            entry.setdefault("id", int(k))
            entries.append(entry)
    out = []
    for e in entries:
        out.append({
            "id":          int(e.get("id", -1)),
            "name":        str(e.get("name", f"class_{e.get('id', '?')}")),
            "part_count":  int(e.get("part_count", e.get("parts", 0))),
            "point_count": int(e.get("point_count", e.get("points", 0))),
        })
    return out


# Hardcoded fallback palette — used if no colormap JSON is found and
# matplotlib is unavailable.  17 perceptually distinct colors for the
# 17 ToolType classes.
FALLBACK_PALETTE = {
    0:  (220,  50,  47),   # Anbohrer               - red
    1:  ( 38, 139, 210),   # Gewindebohrer          - blue
    2:  (203,  75,  22),   # Kegelfraeser           - orange
    3:  (133, 153,   0),   # Kegelsenker            - olive
    4:  (108, 113, 196),   # Kugelfraeser           - violet
    5:  (181, 137,   0),   # Messerkopf             - yellow
    6:  ( 42, 161, 152),   # Rueckwaertskegelsenker - cyan/teal
    7:  (211,  54, 130),   # Schaftfraeser          - magenta
    8:  (147, 161, 161),   # Scheibenfraeser        - slate
    9:  (139,  69,  19),   # Spiralbohrer           - saddle brown
    10: ( 50, 205,  50),   # Stufenbohrer           - lime green
    11: ( 75,   0, 130),   # Tonnenfraeser          - indigo
    12: (255, 140,   0),   # Torusfraeser           - dark orange
    13: ( 32, 178, 170),   # Viertelkreisfraeser    - teal
    14: (165,  42,  42),   # Wendeplattenfraeser    - brown
    15: (154, 205,  50),   # Zapfensenker           - yellow-green
    16: (100, 149, 237),   # Zentrierbohrer         - cornflower blue
}


def load_colormap(path_hint, src_dir, name_to_id):
    """
    Resolve a colormap dict {class_id: (R, G, B)} from disk, with
    several fallbacks.  Diagnostic output is verbose so the user can
    see exactly what happened.

    Accepts these JSON formats:
        {"0": [r, g, b], ...}                       # id-keyed
        {"Anbohrer": [r, g, b], ...}                # name-keyed
        {"Anbohrer": {"color": [r, g, b], ...}, ...}  # name-keyed nested
        {"0": {"color": [r, g, b], ...}, ...}       # id-keyed nested
        [{"id": 0, "color": [r, g, b], ...}, ...]   # list-of-dicts
        {"colormap": <one of the above>}            # wrapped
    Color values may be 0..255 ints or 0..1 floats.
    """
    # Search several locations
    candidates = []
    if path_hint:
        p = Path(path_hint)
        candidates += [
            src_dir / p,                       # next to the data
            src_dir.parent / p,                # one level up from data
            Path(__file__).parent / p,         # next to this script
            Path(__file__).parent.parent / p,  # one level up from script
            Path.cwd() / p,                    # current working directory
            p,                                 # absolute / relative-to-cwd
        ]

    cmap_path = None
    for c in candidates:
        if c.is_file():
            cmap_path = c
            break

    if cmap_path is None:
        print(f"  Colormap file not found.  Tried these locations:")
        for c in candidates:
            print(f"    - {c}")
        # Try matplotlib tab20
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            cm = plt.get_cmap("tab20")
            colors = {}
            for cid in range(20):
                r, g, b, _ = cm(cid / 19.0)
                colors[cid] = (int(r * 255), int(g * 255), int(b * 255))
            print(f"  Using matplotlib tab20 fallback.")
            return colors
        except ImportError:
            print(f"  matplotlib not installed; using hardcoded 17-color palette.")
            return dict(FALLBACK_PALETTE)

    # File found — try to parse
    with open(cmap_path, encoding="utf-8") as f:
        data = json.load(f)

    # Unwrap if there's a top-level "colormap" key
    if isinstance(data, dict) and "colormap" in data:
        data = data["colormap"]

    def normalize_rgb(c):
        if isinstance(c, dict):
            return None  # signals "not a color"
        try:
            c = list(c)[:3]
        except TypeError:
            return None
        if len(c) < 3:
            return None
        # 0..1 floats vs 0..255 ints
        if any(isinstance(x, float) and x <= 1.0 for x in c):
            return tuple(int(max(0, min(255, x * 255))) for x in c)
        return tuple(int(max(0, min(255, x))) for x in c)

    def hex_to_rgb(s):
        """Convert "#1F77B4" or "1F77B4" to (R, G, B)."""
        if not isinstance(s, str):
            return None
        s = s.strip().lstrip("#")
        if len(s) == 6:
            try:
                return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
            except ValueError:
                return None
        return None

    def extract_color(value):
        """value may be a list/tuple of RGB, a hex string, or a dict with
        a 'color' / 'rgb' / 'hex' key."""
        if isinstance(value, dict):
            if "color" in value:
                return normalize_rgb(value["color"])
            if "rgb" in value:
                return normalize_rgb(value["rgb"])
            if "hex" in value:
                return hex_to_rgb(value["hex"])
            return None
        if isinstance(value, str):
            return hex_to_rgb(value)
        return normalize_rgb(value)

    colors = {}
    if isinstance(data, list):
        for e in data:
            if not isinstance(e, dict):
                continue
            if "id" in e:
                col = extract_color(e.get("color") or e.get("rgb") or e)
                if col is not None:
                    colors[int(e["id"])] = col
    elif isinstance(data, dict):
        for k, v in data.items():
            col = extract_color(v)
            if col is None:
                continue
            # Key may be a class ID or class name
            try:
                cid = int(k)
                colors[cid] = col
            except (ValueError, TypeError):
                cid = name_to_id.get(k)
                if cid is not None:
                    colors[cid] = col

    if not colors:
        print(f"  WARNING: colormap file at {cmap_path} parsed to 0 colors.")
        print(f"           Using hardcoded 17-color palette instead.")
        return dict(FALLBACK_PALETTE)

    print(f"  Colormap: loaded {len(colors)} colors from {cmap_path}")
    print(f"            class IDs with color: {sorted(colors)}")
    missing = [c for c in range(17) if c not in colors]
    if missing:
        print(f"            class IDs without color (will use palette fallback): {missing}")
        for c in missing:
            colors[c] = FALLBACK_PALETTE.get(c, (128, 128, 128))
    return colors


def write_ply(path, points_xyz, colors_rgb):
    """ASCII PLY with per-point color."""
    n = len(points_xyz)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points_xyz, colors_rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def main():
    src = Path(SRC).expanduser().resolve()
    dst = Path(DST).expanduser().resolve()

    if not src.is_dir():
        sys.exit(f"ERROR: source not found: {src}")
    if not (src / "xyz").is_dir() or not (src / "seg").is_dir():
        sys.exit(f"ERROR: source missing xyz/ or seg/: {src}")
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"ERROR: destination exists and is non-empty: {dst}\n"
                 f"       Remove it or change DST in the script.")

    # Resolve drop set
    if DROP_IDS:
        drop_set = set(int(c) for c in DROP_IDS)
        print(f"\nDropping (explicit DROP_IDS): {sorted(drop_set)}")
    else:
        # Count parts per class directly from the .seg files (robust to
        # inventory format variations).  A "part" counts for a class if at
        # least one of its points has that class ID.
        seg_paths = sorted((src / "seg").glob("*.seg"))
        print(f"\nCounting parts per class from {len(seg_paths)} .seg files...")
        class_to_parts = {}
        for sf in seg_paths:
            labels = np.loadtxt(sf, dtype=np.int64)
            for cid in np.unique(labels):
                cid_int = int(cid)
                if cid_int < 0:
                    continue
                class_to_parts.setdefault(cid_int, set()).add(sf.stem)

        # Try to get class names from inventory (used only for the printed table)
        inv_path = find_inventory(src)
        name_lookup = {}
        if inv_path is not None:
            try:
                for e in load_inventory(inv_path):
                    name_lookup[e["id"]] = e["name"]
            except Exception:
                pass

        # Apply threshold based on actual counts
        all_class_ids = sorted(class_to_parts.keys())
        drop_set, keep_set = set(), set()
        print(f"\nThreshold: drop classes with < {MIN_PARTS} parts")
        print(f"  {'ID':>3}  {'Class':<28}  {'parts':>6}  {'status':<8}")
        for cid in all_class_ids:
            nparts = len(class_to_parts[cid])
            name = name_lookup.get(cid, f"class_{cid}")
            if nparts < MIN_PARTS:
                drop_set.add(cid)
                status = "DROP"
            else:
                keep_set.add(cid)
                status = "keep"
            print(f"  {cid:>3}  {name:<28}  {nparts:>6}  {status}")
        print(f"\n  Drop: {len(drop_set)} classes (IDs: {sorted(drop_set)})")
        print(f"  Keep: {len(keep_set)} classes (IDs: {sorted(keep_set)})")
    if not drop_set:
        sys.exit("ERROR: drop set is empty.")
    drop_arr = np.array(sorted(drop_set), dtype=np.int64)

    # Name <-> id (for name-keyed colormaps)
    name_to_id = {}
    map_path = find_mapping(src)
    if map_path is not None:
        with open(map_path, encoding="utf-8") as f:
            md = json.load(f)
        if isinstance(md, dict) and "name_to_id" in md:
            name_to_id = {str(k): int(v) for k, v in md["name_to_id"].items()}

    print("\nLoading colormap...")
    GRAY = (128, 128, 128)
    colors = load_colormap(COLORMAP_PATH, src, name_to_id)

    print(f"\nCreating {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "xyz").mkdir(exist_ok=True)
    (dst / "seg").mkdir(exist_ok=True)
    (dst / "ply").mkdir(exist_ok=True)

    seg_files = sorted((src / "seg").glob("*.seg"))
    print(f"\nProcessing {len(seg_files)} parts (xyz, seg, ply)...")

    total_pts_before, total_pts_after = {}, {}
    parts_fully_dropped = []

    for sf in seg_files:
        stem = sf.stem
        xyz_path = src / "xyz" / f"{stem}.xyz"
        if not xyz_path.is_file():
            print(f"  WARNING: no xyz file for {stem} — skipping")
            continue

        xyz_data = np.loadtxt(xyz_path, dtype=np.float32)
        labels   = np.loadtxt(sf, dtype=np.int64)

        for cid, cnt in zip(*np.unique(labels, return_counts=True)):
            total_pts_before[int(cid)] = total_pts_before.get(int(cid), 0) + int(cnt)

        mask = np.isin(labels, drop_arr)
        labels[mask] = -1

        for cid, cnt in zip(*np.unique(labels, return_counts=True)):
            total_pts_after[int(cid)] = total_pts_after.get(int(cid), 0) + int(cnt)

        if (labels == -1).all():
            parts_fully_dropped.append(stem)

        # Write outputs
        shutil.copy2(xyz_path, dst / "xyz" / xyz_path.name)
        np.savetxt(dst / "seg" / sf.name, labels, fmt="%d")
        point_colors = [GRAY if l == -1 else colors.get(int(l), GRAY) for l in labels]
        write_ply(dst / "ply" / f"{stem}.ply", xyz_data[:, 0:3], point_colors)

    # Copy splits/
    src_splits = src / "splits"
    if src_splits.is_dir():
        print(f"\nCopying splits/ directory...")
        shutil.copytree(src_splits, dst / "splits")
    else:
        print(f"\nWARNING: no splits/ in {src}.  Run make_cv_splits.py on the new dataset.")

    # Copy global mapping
    if map_path is not None:
        shutil.copy2(map_path, dst / map_path.name)
        print(f"Copied {map_path.name}")

    # Updated inventory
    inv_path = find_inventory(src)
    if inv_path is not None:
        inventory = load_inventory(inv_path)
        new_inventory = []
        for e in inventory:
            cid = e["id"]
            new_inventory.append({
                "id":              cid,
                "name":            e["name"],
                "part_count":      e["part_count"],
                "point_count":     total_pts_after.get(cid, 0),
                "dropped":         cid in drop_set,
                "original_points": e["point_count"],
            })
        with open(dst / inv_path.name, "w", encoding="utf-8") as f:
            json.dump(new_inventory, f, indent=2)
        print(f"Wrote updated inventory: {inv_path.name}")

    # Provenance
    summary = {
        "source":                  str(src),
        "destination":             str(dst),
        "drop_ids":                sorted(drop_set),
        "min_parts_threshold":     None if DROP_IDS else MIN_PARTS,
        "n_parts":                 len(seg_files),
        "n_parts_fully_dropped":   len(parts_fully_dropped),
        "parts_fully_dropped":     parts_fully_dropped,
        "points_per_class_before": dict(sorted(total_pts_before.items())),
        "points_per_class_after":  dict(sorted(total_pts_after.items())),
    }
    with open(dst / "_subset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote provenance: _subset_summary.json")

    # Verification
    print("\n" + "=" * 72)
    print("VERIFICATION — points per class before vs after")
    print("=" * 72)
    all_ids = sorted(set(total_pts_before) | set(total_pts_after))
    name_lookup = {e["id"]: e["name"] for e in (load_inventory(inv_path) if inv_path else [])}
    name_w = max((len(name_lookup.get(c, f"class_{c}")) for c in all_ids), default=12)
    print(f"  {'ID':>4}  {'Class':<{name_w}}  {'before':>10}  {'after':>10}  {'status':<10}")
    print("  " + "-" * (4 + 2 + name_w + 2 + 10 + 2 + 10 + 2 + 10))
    for cid in all_ids:
        name = name_lookup.get(cid, f"class_{cid}" if cid >= 0 else "(unmapped)")
        before = total_pts_before.get(cid, 0)
        after  = total_pts_after.get(cid, 0)
        if cid == -1:
            status = "ignore"
        elif cid in drop_set:
            status = "DROPPED"
        else:
            status = "kept"
        print(f"  {cid:>4}  {name:<{name_w}}  {before:>10,}  {after:>10,}  {status}")
    print()
    if parts_fully_dropped:
        print(f"WARNING: {len(parts_fully_dropped)} part(s) became 100% -1:")
        for p in parts_fully_dropped[:10]:
            print(f"  {p}")
        if len(parts_fully_dropped) > 10:
            print(f"  ...and {len(parts_fully_dropped) - 10} more")
        print("These parts will be loaded but contribute nothing to training/eval.")
    print()
    print("DONE. New dataset at:")
    print(f"  {dst}")
    print()
    print("Next: in slurm_phase2.sh, set")
    print(f"  DATA_ROOT=\"{dst}\"")
    print("  IGNORE_CLASSES=\"\"     # the drop is baked into the data")


if __name__ == "__main__":
    main()
