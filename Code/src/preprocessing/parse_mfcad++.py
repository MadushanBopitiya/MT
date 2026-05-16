import os
import sys
import glob
import re
import numpy as np
import trimesh

# PythonOCC imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location

# --- CONFIGURATION ---
TOTAL_POINTS = 4096
# Ensure this points to the split you want (train or test)
OUTPUT_DIR = r"D:\MASTER THESIS\Data\processed\MFCAD++\test"
INPUT_DIR = r"D:\MASTER THESIS\Data\raw\MFCAD++\MFCAD++_dataset\step\test"

# --- 1. CLASS MAPPING ---
CLASS_MAP = {
    "chamfer": 0,
    "through_hole": 1,
    "triangular_passage": 2,
    "rectangular_passage": 3,
    "6_sided_passage": 4,
    "triangular_through_slot": 5,
    "rectangular_through_slot": 6,
    "circular_through_slot": 7,
    "rectangular_through_step": 8,
    "2_sided_through_step": 9,
    "slanted_through_step": 10,
    "o_ring": 11,
    "blind_hole": 12,
    "triangular_pocket": 13,
    "rectangular_pocket": 14,
    "6_sided_pocket": 15,
    "circular_end_pocket": 16,
    "rectangular_blind_slot": 17,
    "vertical_circular_end_blind_slot": 18,
    "horizontal_circular_end_blind_slot": 19,
    "triangular_blind_step": 20,
    "circular_blind_step": 21,
    "rectangular_blind_step": 22,
    "round": 23,
    "stock": 24
}

# --- 2. HIGH-CONTRAST PALETTE (25 Classes) ---
# Distinct colors manually chosen for visibility
COLOR_PALETTE = np.array([
    [128, 128, 128], # 0: Chamfer (Grey)
    [255, 0, 0],     # 1: Through Hole (Red)
    [0, 255, 0],     # 2: Tri Passage (Green)
    [0, 0, 255],     # 3: Rect Passage (Blue)
    [255, 255, 0],   # 4: 6-Sided Passage (Yellow)
    [255, 0, 255],   # 5: Tri Thru Slot (Magenta)
    [0, 255, 255],   # 6: Rect Thru Slot (Cyan)
    [255, 128, 0],   # 7: Circ Thru Slot (Orange)
    [128, 0, 128],   # 8: Rect Thru Step (Purple)
    [0, 128, 128],   # 9: 2-Sided Step (Teal)
    [128, 128, 0],   # 10: Slanted Step (Olive)
    [255, 192, 203], # 11: O-Ring (Pink)
    [139, 0, 0],     # 12: Blind Hole (Dark Red)
    [0, 100, 0],     # 13: Tri Pocket (Dark Green)
    [0, 0, 139],     # 14: Rect Pocket (Dark Blue)
    [255, 215, 0],   # 15: 6-Sided Pocket (Gold)
    [75, 0, 130],    # 16: Circ End Pocket (Indigo)
    [255, 160, 122], # 17: Rect Blind Slot (Salmon)
    [32, 178, 170],  # 18: Vert Blind Slot (Light Sea Green)
    [240, 230, 140], # 19: Horiz Blind Slot (Khaki)
    [210, 105, 30],  # 20: Tri Blind Step (Chocolate)
    [188, 143, 143], # 21: Circ Blind Step (Rosy Brown)
    [220, 20, 60],   # 22: Rect Blind Step (Crimson)
    [0, 250, 154],   # 23: Round (Medium Spring Green)
    [50, 50, 50]     # 24: Stock (Dark Grey/Black)
])

def get_class_id(raw_name):
    """Robustly handles both numeric IDs and text names."""
    if not raw_name: return 24
    if raw_name.isdigit(): return int(raw_name)
    
    s = raw_name.lower().replace("-", "_").replace(" ", "_")
    if s in CLASS_MAP: return CLASS_MAP[s]
    
    for key in CLASS_MAP.keys():
        if key in ["face", "stock", "feature"]: continue
        if s.startswith(key): return CLASS_MAP[key]
            
    if "stock" in s: return 24
    if "face" in s: return 0
    return 24

def parse_step_regex_labels(step_filename):
    """Extracts the sequence of face names/IDs from the STEP file text."""
    labels = []
    try:
        with open(step_filename, 'r', errors='ignore') as f:
            content = f.read()
        pattern = re.compile(r"ADVANCED_FACE\s*\(\s*'([^']+)'")
        matches = pattern.findall(content)
        for m in matches:
            labels.append(get_class_id(m))
    except Exception as e:
        print(f"Error parsing text: {e}")
    return labels

def occ_face_to_trimesh(occ_face):
    """Convert PythonOCC face to Trimesh."""
    BRepMesh_IncrementalMesh(occ_face, 0.1)
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(occ_face, loc)
    if triangulation is None: return None

    nodes = []
    for i in range(1, triangulation.NbNodes() + 1):
        pnt = triangulation.Node(i).Transformed(loc.Transformation())
        nodes.append([pnt.X(), pnt.Y(), pnt.Z()])
        
    triangles = []
    for i in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(i)
        triangles.append([tri.Value(1)-1, tri.Value(2)-1, tri.Value(3)-1])
        
    if len(triangles) == 0: return None
    return trimesh.Trimesh(vertices=nodes, faces=triangles)

def process_step_file(filepath):
    name_stem = os.path.basename(filepath).split('.')[0]
    # print(f"[{name_stem}] Parsing...")
    
    # 1. Get Labels
    file_labels = parse_step_regex_labels(filepath)

    # 2. Load Geometry
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filepath)
    if status != 1: return

    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    all_points, all_normals, all_label_ids = [], [], []
    face_idx = 0
    
    while explorer.More():
        face = explorer.Current()
        lbl = file_labels[face_idx] if face_idx < len(file_labels) else 24

        tm = occ_face_to_trimesh(face)
        if tm is not None:
            n_samples = int(max(10, tm.area * 5000)) 
            pts, idxs = trimesh.sample.sample_surface(tm, n_samples)
            nrms = tm.face_normals[idxs]
            
            all_points.append(pts)
            all_normals.append(nrms)
            all_label_ids.append(np.full(len(pts), lbl))
        
        face_idx += 1
        explorer.Next()

    if not all_points: return
    
    # 3. Aggregate & Save
    full_pts = np.vstack(all_points)
    full_nrms = np.vstack(all_normals)
    full_lbls = np.concatenate(all_label_ids)
    
    # Downsample
    choice = np.random.choice(len(full_pts), TOTAL_POINTS, replace=(len(full_pts) < TOTAL_POINTS))
    final_pts = full_pts[choice]
    final_nrms = full_nrms[choice]
    final_lbls = full_lbls[choice]
    
    # Normalize
    final_pts -= np.mean(final_pts, axis=0)
    final_pts /= np.max(np.sqrt(np.sum(final_pts**2, axis=1)))

    # --- SAVE (FLAT STRUCTURE) ---
    xyz_dir = os.path.join(OUTPUT_DIR, "xyz")
    seg_dir = os.path.join(OUTPUT_DIR, "seg")
    ply_dir = os.path.join(OUTPUT_DIR, "ply")
    
    os.makedirs(xyz_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    # Save Data
    np.savetxt(os.path.join(xyz_dir, f"{name_stem}.xyz"), np.hstack((final_pts, final_nrms)), fmt='%.6f')
    np.savetxt(os.path.join(seg_dir, f"{name_stem}.seg"), final_lbls, fmt='%d')
    
    # Save PLY
    safe_lbls = np.clip(final_lbls, 0, len(COLOR_PALETTE)-1)
    colors = COLOR_PALETTE[safe_lbls]
    pcd = trimesh.points.PointCloud(vertices=final_pts, colors=colors)
    pcd.export(os.path.join(ply_dir, f"{name_stem}.ply"))
    
    print(f" > Processed: {name_stem}")

if __name__ == "__main__":
    # Ensure directories exist
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    files = glob.glob(os.path.join(INPUT_DIR, "*.stp")) + glob.glob(os.path.join(INPUT_DIR, "*.step"))
    print(f"Found {len(files)} STEP files in {INPUT_DIR}")
    
    for f in files:
        process_step_file(f)