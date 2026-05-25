import os
import glob
import json
import csv
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
TOTAL_POINTS = 2048

# Our 18-to-14 Tool Class Compression Dictionary
TOOL_LABEL_MAPPING = {
    0: 0, 1: 1, 2: 2, 3: 3, 5: 3, 4: 4, 6: 5, 7: 6, 8: 7, 
    10: 7, 9: 8, 11: 9, 13: 9, 12: 10, 14: 11, 17: 11, 15: 12, 16: 13
}

# 14 Distinct Colors for the 14 Tool Classes + 1 Grey for Unmapped (-1)
PALETTE = np.array([
    [255, 0, 0],     # 0: Anbohrer (Red)
    [0, 255, 0],     # 1: Kegelfraeser (Green)
    [0, 0, 255],     # 2: Kegelsenker (Blue)
    [255, 255, 0],   # 3: Kugelfraeser (Yellow)
    [255, 0, 255],   # 4: Messerkopf (Magenta)
    [0, 255, 255],   # 5: Schaftfraeser (Cyan)
    [255, 128, 0],   # 6: Rueckwaertskegelsenker (Orange)
    [128, 0, 128],   # 7: Spiralbohrer (Purple)
    [128, 128, 0],   # 8: Stufenbohrer (Olive)
    [0, 128, 128],   # 9: Torusfraeser (Teal)
    [128, 0, 0],     # 10: Viertelkreisfraeser (Maroon)
    [0, 128, 0],     # 11: Zentrierbohrer (Dark Green)
    [0, 0, 128],     # 12: Wendeplattenfraeser (Navy)
    [255, 105, 180], # 13: Zapfensenker (Hot Pink)
    [192, 192, 192]  # Default/Unmapped (Light Grey) -> For label -1
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

def create_face_to_label_map(json_path, csv_path):
    """Bridges JSON Face IDs to CSV Sequence, then compresses to PyTorch Class (0-13)"""
    seq_to_class = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                if 'Sequence' in row and 'ToolType' in row:
                    seq_id = int(row['Sequence'])
                    tool_id = int(row['ToolType'])
                    seq_to_class[seq_id] = TOOL_LABEL_MAPPING.get(tool_id, -1)
            
    face_to_class = {}
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            face_types = json.load(f)
            for face_id, data_array in face_types.items():
                if face_id == "keys": continue 
                sequence_id = data_array[5] # AreaColor / Sequence is index 5
                face_to_class[str(face_id)] = seq_to_class.get(sequence_id, -1)
                
    return face_to_class

def process_single_part(folder_path, output_dir):
    """Processes a single CAD part folder."""
    part_name = os.path.basename(folder_path)
    
    # 1. Locate the trio of files
    stp_path = os.path.join(folder_path, f"{part_name}_Dst.stp")
    csv_path = os.path.join(folder_path, f"{part_name}_encoded.csv")
    json_path = os.path.join(folder_path, f"{part_name}_face_types.json")
    
    if not os.path.exists(stp_path) or not os.path.exists(json_path):
        return # Skip if missing core files

    # 2. Build the label mapping
    face_to_label = create_face_to_label_map(json_path, csv_path)

    # 3. Load Geometry
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(stp_path)
    if status != 1: return

    step_reader.TransferRoots()
    try: shape = step_reader.OneShape()
    except AssertionError: return

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    all_points, all_normals, all_labels = [], [], []
    
    # In datasets exported to STEP, topological iteration often matches the assigned integer Face ID
    face_idx = 0 
    
    while explorer.More():
        face = explorer.Current()
        
        # Try to map by string index. If it wasn't machined, it defaults to -1.
        label = face_to_label.get(str(face_idx), -1)
            
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
    
    # 4. Downsample/Upsample to 4096
    if len(full_pts) >= TOTAL_POINTS:
        choice = np.random.choice(len(full_pts), TOTAL_POINTS, replace=False)
    else:
        choice = np.random.choice(len(full_pts), TOTAL_POINTS, replace=True)
        
    final_pts = full_pts[choice]
    final_nrms = full_nrms[choice]
    final_lbls = full_lbls[choice]
    
    # 5. Normalize (Unit sphere)
    final_pts -= np.mean(final_pts, axis=0)
    final_pts /= np.max(np.sqrt(np.sum(final_pts**2, axis=1)))

    # --- 6. SAVE OUTPUTS ---
    xyz_dir = os.path.join(output_dir, "xyz")
    seg_dir = os.path.join(output_dir, "seg")
    ply_dir = os.path.join(output_dir, "ply")
    
    os.makedirs(xyz_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    # Save XYZ (Points + Normals) and SEG (Labels)
    np.savetxt(os.path.join(xyz_dir, f"{part_name}.xyz"), np.hstack((final_pts, final_nrms)), fmt='%.6f')
    np.savetxt(os.path.join(seg_dir, f"{part_name}.seg"), final_lbls, fmt='%d')
    
    # PLY Visualization
    # For PLY viewing, we map the -1 (unmapped) labels to index 14 (Grey color)
    vis_lbls = np.where(final_lbls == -1, 14, final_lbls)
    colors = PALETTE[vis_lbls]
    
    pcd = trimesh.points.PointCloud(vertices=final_pts, colors=colors)
    pcd.export(os.path.join(ply_dir, f"{part_name}.ply"))

if __name__ == "__main__":
    # Point this directly to the root containing your 75 part folders
    DATASET_ROOT = r"D:\MASTER THESIS\Data\raw\Machining_Tools" 
    
    # Output directory for the finished point clouds
    PROCESSED_ROOT = r"D:\MASTER THESIS\Data\processed\2048\Machining_Tools"

    # Find all subdirectories
    part_folders = [f.path for f in os.scandir(DATASET_ROOT) if f.is_dir()]
    
    print(f"Found {len(part_folders)} part folders. Beginning conversion...\n")
    
    for folder in tqdm(part_folders, desc="Processing Parts"):
        process_single_part(folder, PROCESSED_ROOT)
        
    print("\n✅ Dataset Conversion Complete.")