import os
import sys
import glob
import json
import numpy as np
import trimesh
from tqdm import tqdm

# PythonOCC imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location

# --- CONFIGURATION ---
TOTAL_POINTS = 4096

# Fixed Palette
PALETTE = np.array([
    [255, 0, 0],     # 0: Red - ExtrudeSide          
    [0, 255, 0],     # 1: Lime Green - ExtrudeEnd  
    [0, 0, 255],     # 2: Blue - CutSide         
    [255, 255, 0],   # 3: Yellow - CutEnd       
    [255, 0, 255],   # 4: Magenta - Fillet
    [0, 255, 255],   # 5: Cyan - Chamfer         
    [255, 128, 0],   # 6: Orange - RevolveSide        
    [128, 0, 128],   # 7: Purple - RevolveEnd       
    [192, 192, 192]  # 8+: Light Grey   (Background/Other)
])

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

def parse_seg_file(seg_path):
    labels = []
    if not os.path.exists(seg_path): return []
    try:
        with open(seg_path, 'r') as f:
            lines = f.readlines()
            labels = [int(line.strip()) for line in lines if line.strip().isdigit()]
    except: pass
    return labels

def process_single_file(file_id, dirs, output_dir):
    # 1. Path Construction
    stp_path = os.path.join(dirs['stp'], f"{file_id}.stp")
    seg_path = os.path.join(dirs['seg'], f"{file_id}.seg")
    
    if not os.path.exists(stp_path):
        alt_path = os.path.join(dirs['stp'], f"{file_id}.step")
        if os.path.exists(alt_path): stp_path = alt_path
        else: return

    # 2. Load Labels
    face_labels = parse_seg_file(seg_path)
    if not face_labels: return 

    # 3. Load Geometry
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(stp_path)
    if status != 1: return

    step_reader.TransferRoots()
    try: shape = step_reader.OneShape()
    except AssertionError: return

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    all_points, all_normals, all_labels = [], [], []
    face_idx = 0 
    
    while explorer.More():
        face = explorer.Current()
        label = face_labels[face_idx] if face_idx < len(face_labels) else 8
            
        tm = occ_face_to_trimesh(face)
        if tm is not None:
            n_samples = int(max(10, tm.area * 5000))
            pts, idxs = trimesh.sample.sample_surface(tm, n_samples)
            nrms = tm.face_normals[idxs]
            
            all_points.append(pts)
            all_normals.append(nrms)
            all_labels.append(np.full(len(pts), label))
        
        face_idx += 1
        explorer.Next()

    if not all_points: return
    
    full_pts = np.vstack(all_points)
    full_nrms = np.vstack(all_normals)
    full_lbls = np.concatenate(all_labels)
    
    # Downsample
    if len(full_pts) >= TOTAL_POINTS:
        choice = np.random.choice(len(full_pts), TOTAL_POINTS, replace=False)
    else:
        choice = np.random.choice(len(full_pts), TOTAL_POINTS, replace=True)
        
    final_pts = full_pts[choice]
    final_nrms = full_nrms[choice]
    final_lbls = full_lbls[choice]
    
    # Normalize
    final_pts -= np.mean(final_pts, axis=0)
    final_pts /= np.max(np.sqrt(np.sum(final_pts**2, axis=1)))

    # --- 4. SAVE (UPDATED: FLAT STRUCTURE) ---
    xyz_dir = os.path.join(output_dir, "xyz")
    seg_dir = os.path.join(output_dir, "seg")
    ply_dir = os.path.join(output_dir, "ply")
    
    # Create folders (exist_ok=True handles the race condition safely)
    os.makedirs(xyz_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    # Save Files Directly
    np.savetxt(os.path.join(xyz_dir, f"{file_id}.xyz"), np.hstack((final_pts, final_nrms)), fmt='%.6f')
    np.savetxt(os.path.join(seg_dir, f"{file_id}.seg"), final_lbls, fmt='%d')
    
    # PLY Visualization
    safe_lbls = np.clip(final_lbls, 0, len(PALETTE)-1)
    colors = PALETTE[safe_lbls]
    pcd = trimesh.points.PointCloud(vertices=final_pts, colors=colors)
    pcd.export(os.path.join(ply_dir, f"{file_id}.ply"))

if __name__ == "__main__":
    RAW_ROOT = r"D:\MASTER THESIS\Data\raw\Fusion360-Segmentation\s2.0.1\breps"
    PROCESSED_ROOT = r"D:\MASTER THESIS\Data\processed\Fusion360"
    SPLIT_JSON = os.path.join(RAW_ROOT, "train_test.json")

    dirs = {'stp': os.path.join(RAW_ROOT, 'step'), 'seg': os.path.join(RAW_ROOT, 'seg')}

    if os.path.exists(SPLIT_JSON):
        with open(SPLIT_JSON, 'r') as f: splits = json.load(f)
        
        for split_name in ['train', 'test']:
            if split_name not in splits: continue
            
            output_dir = os.path.join(PROCESSED_ROOT, split_name)
            
            # Note: We don't create output_dir itself, the function creates subfolders
            for file_id in tqdm(splits[split_name], desc=f"Converting {split_name}"):
                clean_id = file_id.split('.')[0]
                process_single_file(clean_id, dirs, output_dir)
        print("\n✅ Fusion360 Processing Complete (Flat Structure).")
    else:
        print(f"❌ Error: Could not find {SPLIT_JSON}")