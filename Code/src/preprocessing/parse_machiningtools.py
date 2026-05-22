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

# ==========================================
# --- 1. CONFIGURATION ---
# ==========================================
TOTAL_POINTS = 2048  # Updated to match the \2048\ output directory

RAW_DATASET_DIR = r"D:\MASTER THESIS\Data\raw\Machining_Tools"
OUTPUT_DIR = r"D:\MASTER THESIS\Data\processed\2048\Machining_Tools"

# Class ID to assign to faces that do not require machining (e.g., untouched stock)
# Make sure this doesn't overlap with a real ToolType ID from the CSV
UNMACHINED_CLASS = 0 

# Dynamically generate a large color palette for up to 100 tool types for PLY viewing
np.random.seed(42)
COLOR_PALETTE = np.random.randint(0, 255, size=(100, 3))
COLOR_PALETTE[UNMACHINED_CLASS] = [100, 100, 100] # Set unmachined faces to Gray


# ==========================================
# --- 2. HELPER FUNCTIONS ---
# ==========================================
def occ_face_to_trimesh(occ_face):
    """Convert a PythonOCC face to a Trimesh object."""
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

def build_face_to_tool_map(json_path, csv_path):
    """
    Creates a direct dictionary mapping: PythonOCC Face Index -> Target ToolType
    """
    # 1. Map Sequence ID -> ToolType from CSV
    seq_to_tool = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                seq_id = int(row['Sequence'])
                tool_type = int(row['ToolType'])
                seq_to_tool[seq_id] = tool_type

    # 2. Map Face Index -> Sequence ID -> ToolType from JSON
    face_idx_to_tool = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            face_types = json.load(f)
            
        for key, value in face_types.items():
            if key == "keys": continue # Skip the metadata key
            
            face_idx = int(key)
            sequence_id = int(value[5]) # The 6th element is the Sequence ID
            
            # Look up the ToolType. If sequence isn't in CSV, assume it's unmachined
            target_tool = seq_to_tool.get(sequence_id, UNMACHINED_CLASS)
            face_idx_to_tool[face_idx] = target_tool
            
    return face_idx_to_tool


# ==========================================
# --- 3. MAIN PROCESSING PIPELINE ---
# ==========================================
def process_cam_folder(folder_path):
    """Processes a single part folder dynamically using the sidecar JSON/CSV methodology."""
    
    # 1. Dynamically find the STEP file in the main folder
    stp_files = glob.glob(os.path.join(folder_path, "*.stp")) + glob.glob(os.path.join(folder_path, "*.step"))
    if not stp_files: 
        return # Skip if no CAD file exists in this folder
        
    stp_path = stp_files[0]
    
    # Extract the true base name (e.g., "bearing_holder" from "bearing_holder_Dst.stp")
    raw_stem = os.path.basename(stp_path).split('.')[0]
    base_name = raw_stem.replace("_Dst", "") 
    
    # 2. Build paths to the sidecar metadata
    json_path = os.path.join(folder_path, f"{base_name}_face_types.json")
    csv_path = os.path.join(folder_path, f"{base_name}_encoded.csv")

    # 3. Build the exact Face -> Tool label map
    face_to_tool_map = build_face_to_tool_map(json_path, csv_path)

    # 4. Load Geometry
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(stp_path)
    if status != 1: return

    step_reader.TransferRoots()
    try: 
        shape = step_reader.OneShape()
    except AssertionError: 
        return

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    all_points, all_normals, all_labels = [], [], []
    
    # Track the sequence perfectly with PythonOCC's traversal
    face_idx = 0 
    
    while explorer.More():
        face = explorer.Current()
        
        # Get the target ToolType. Default to unmachined if face wasn't in the JSON
        label = face_to_tool_map.get(face_idx, UNMACHINED_CLASS)
            
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
    
    # Normalize to Unit Sphere (Crucial for PointNet++ / KPConv pipelines)
    final_pts -= np.mean(final_pts, axis=0)
    final_pts /= np.max(np.sqrt(np.sum(final_pts**2, axis=1)))

    # --- 5. SAVE (FLAT STRUCTURE FOR thesis_dataset.py) ---
    xyz_dir = os.path.join(OUTPUT_DIR, "xyz")
    seg_dir = os.path.join(OUTPUT_DIR, "seg")
    ply_dir = os.path.join(OUTPUT_DIR, "ply")
    
    os.makedirs(xyz_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    # Save Files Directly using base_name to keep outputs clean
    np.savetxt(os.path.join(xyz_dir, f"{base_name}.xyz"), np.hstack((final_pts, final_nrms)), fmt='%.6f')
    np.savetxt(os.path.join(seg_dir, f"{base_name}.seg"), final_lbls, fmt='%d')
    
    # PLY Visualization
    safe_lbls = np.clip(final_lbls, 0, len(COLOR_PALETTE)-1)
    colors = COLOR_PALETTE[safe_lbls]
    pcd = trimesh.points.PointCloud(vertices=final_pts, colors=colors)
    pcd.export(os.path.join(ply_dir, f"{base_name}.ply"))


# ==========================================
# --- 4. EXECUTION ---
# ==========================================
if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all subfolders inside the raw dataset directory
    part_folders = [f.path for f in os.scandir(RAW_DATASET_DIR) if f.is_dir()]
    
    print(f"Found {len(part_folders)} part folders to process in {RAW_DATASET_DIR}")
    
    for folder_path in tqdm(part_folders, desc="Processing CAM Dataset"):
        process_cam_folder(folder_path)
        
    print("\n✅ CAM Tooling Dataset Processing Complete.")
    print(f"Check your output files at: {OUTPUT_DIR}")