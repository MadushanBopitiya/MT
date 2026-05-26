import os
import json
import csv
import numpy as np
import trimesh
from tqdm import tqdm

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location

# --- CONFIGURATION ---
TOTAL_POINTS = 2048

# --- TOOL CLASS DICTIONARY ---
TOOL_LABEL_MAPPING = {
    0: 0,   # Anbohrer (Spot drill)
    1: 1,   # Kegelfraeser (Chamfer mill)
    2: 2,   # Kegelsenker (Countersink)
    3: 3,   # Kugelfraeser (Ball nose end mill)
    5: 3,   # Kugelfraeser (Duplicate)
    4: 4,   # Messerkopf (Face mill)
    6: 5,   # Schaftfraeser (Flat end mill)
    7: 6,   # Rueckwaertskegelsenker (Back countersink)
    8: 7,   # Spiralbohrer (Twist drill)
    10: 7,  # Spiralbohrer (Duplicate)
    9: 8,   # Stufenbohrer (Step drill)
    11: 9,  # Torusfraeser (Bull-nose end mill)
    13: 9,  # Torusfraeser (Duplicate)
    12: 10, # Viertelkreisfraeser (Corner rounding cutter)
    14: 11, # Zentrierbohrer (Center drill)
    17: 11, # Zentrierbohrer (Duplicate)
    15: 12, # Wendeplattenfraeser (Indexable insert mill)
    16: 13  # Zapfensenker (Counterbore)
}

# --- VISUALIZATION PALETTE ---
PALETTE = np.array([
    [255, 0, 0],     # Class 0:  Anbohrer                  (Red)
    [0, 255, 0],     # Class 1:  Kegelfraeser              (Green)
    [0, 0, 255],     # Class 2:  Kegelsenker               (Blue)
    [255, 255, 0],   # Class 3:  Kugelfraeser              (Yellow)
    [255, 0, 255],   # Class 4:  Messerkopf                (Magenta)
    [0, 255, 255],   # Class 5:  Schaftfraeser             (Cyan)
    [255, 128, 0],   # Class 6:  Rueckwaertskegelsenker    (Orange)
    [128, 0, 128],   # Class 7:  Spiralbohrer              (Purple)
    [128, 128, 0],   # Class 8:  Stufenbohrer              (Olive)
    [0, 128, 128],   # Class 9:  Torusfraeser              (Teal)
    [128, 0, 0],     # Class 10: Viertelkreisfraeser       (Maroon)
    [0, 128, 0],     # Class 11: Zentrierbohrer            (Dark Green)
    [0, 0, 128],     # Class 12: Wendeplattenfraeser       (Navy)
    [255, 105, 180], # Class 13: Zapfensenker              (Hot Pink)
    [192, 192, 192]  # Class -1: Background / Unmachined   (Grey) -> Mapped to Index 14 for PLY
])

def occ_face_to_trimesh(occ_face):
    BRepMesh_IncrementalMesh(occ_face, 0.5) 
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(occ_face, loc)
    if triangulation is None: return None

    nodes = [[triangulation.Node(i).Transformed(loc.Transformation()).X(),
              triangulation.Node(i).Transformed(loc.Transformation()).Y(),
              triangulation.Node(i).Transformed(loc.Transformation()).Z()] 
             for i in range(1, triangulation.NbNodes() + 1)]
        
    triangles = [[triangulation.Triangle(i).Value(1)-1, 
                  triangulation.Triangle(i).Value(2)-1, 
                  triangulation.Triangle(i).Value(3)-1] 
                 for i in range(1, triangulation.NbTriangles() + 1)]
        
    if not triangles: return None
    return trimesh.Trimesh(vertices=nodes, faces=triangles)

def find_core_files(folder_path):
    stp_file, json_file, csv_file = None, None, None
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        lower_name = filename.lower()
        
        if lower_name.endswith('.stp') or lower_name.endswith('.step'): stp_file = filepath
        elif lower_name.endswith('.json'):
            if 'face_types' in lower_name: json_file = filepath
            elif not json_file:
                try:
                    with open(filepath, 'r') as f:
                        if "keys" in json.load(f): json_file = filepath
                except: pass

    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    for f in csv_files:
        if '_encoded' in f.lower():
            csv_file = f
            break
            
    if not csv_file:
        for f in csv_files:
            try:
                with open(f, 'r', encoding='utf-8-sig') as file:
                    reader = csv.reader(file, delimiter=';')
                    header = [h.strip() for h in next(reader, [])]
                    if 'ToolType' in header:
                        tool_idx = header.index('ToolType')
                        first_data_row = next(reader, None)
                        if first_data_row and len(first_data_row) > tool_idx and first_data_row[tool_idx].strip().isdigit():
                            csv_file = f
                            break
            except: pass
            
    return stp_file, json_file, csv_file

def create_face_to_label_map(json_path, csv_path):
    seq_to_class = {}
    if csv_path:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=';')
            header = [h.strip() for h in next(reader, [])] 
            
            if 'Sequence' in header and 'ToolType' in header:
                seq_idx = header.index('Sequence')
                tool_idx = header.index('ToolType')
                for row in reader:
                    if len(row) > tool_idx:
                        seq_str = row[seq_idx].strip()
                        tool_str = row[tool_idx].strip()
                        if seq_str.isdigit():
                            tool_id = int(tool_str) if tool_str.isdigit() else -1
                            seq_to_class[int(seq_str)] = TOOL_LABEL_MAPPING.get(tool_id, -1)
    
    face_to_class = {}
    if json_path:
        with open(json_path, 'r', encoding='utf-8') as f:
            face_types = json.load(f)
            for face_id, data_array in face_types.items():
                if face_id != "keys":
                    seq_id = data_array[5]
                    face_to_class[str(face_id)] = seq_to_class.get(seq_id, -1)
                
    return face_to_class

def process_single_part(folder_path, output_dir):
    part_name = os.path.basename(folder_path)
    stp_path, json_path, csv_path = find_core_files(folder_path)
    
    if not stp_path or not json_path: return
    
    face_to_label = create_face_to_label_map(json_path, csv_path)

    step_reader = STEPControl_Reader()
    try:
        if step_reader.ReadFile(stp_path) != 1: return
        step_reader.TransferRoots()
        shape = step_reader.OneShape()
    except: return

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    all_points, all_normals, all_labels = [], [], []
    face_idx = 0 
    
    while explorer.More():
        face = explorer.Current()
        label = face_to_label.get(str(face_idx), -1)
            
        tm = occ_face_to_trimesh(face)
        if tm is not None and tm.area > 0:
            # --- THE ONLY MATH CHANGE: Fixed RAM Freeze for Millimeters (* 50) ---
            n_samples = int(max(10, tm.area * 50))
            
            pts, idxs = trimesh.sample.sample_surface(tm, n_samples)
            all_points.append(pts)
            all_normals.append(tm.face_normals[idxs])
            all_labels.append(np.full(len(pts), label))
            
        face_idx += 1
        explorer.Next()

    if not all_points: return

    full_pts = np.vstack(all_points)
    full_nrms = np.vstack(all_normals)
    full_lbls = np.concatenate(all_labels)
    
    pool_size = len(full_pts)

    # --- EXACT FUSION360/MFCAD++ BLIND CROP LOGIC WITH CONSOLE LOGGING ---
    if pool_size >= TOTAL_POINTS:
        tqdm.write(f" 🟢 [{part_name}] Pool Size: {pool_size} (GREATER than or equal to {TOTAL_POINTS}) -> Cropping normally.")
        choice = np.random.choice(pool_size, TOTAL_POINTS, replace=False)
    else:
        tqdm.write(f" 🔴 [{part_name}] Pool Size: {pool_size} (LESS than {TOTAL_POINTS}) -> Forcing duplication.")
        choice = np.random.choice(pool_size, TOTAL_POINTS, replace=True)
    # ---------------------------------------------------------------------

    final_pts = full_pts[choice]
    final_nrms = full_nrms[choice]
    final_lbls = full_lbls[choice]

    final_pts -= np.mean(final_pts, axis=0)
    max_dist = np.max(np.sqrt(np.sum(final_pts**2, axis=1)))
    if max_dist > 0: final_pts /= max_dist

    for d in ["xyz", "seg", "ply"]: os.makedirs(os.path.join(output_dir, d), exist_ok=True)
    
    np.savetxt(os.path.join(output_dir, "xyz", f"{part_name}.xyz"), np.hstack((final_pts, final_nrms)), fmt='%.6f')
    np.savetxt(os.path.join(output_dir, "seg", f"{part_name}.seg"), final_lbls, fmt='%d')
    
    vis_lbls = np.where(final_lbls == -1, 14, final_lbls)
    trimesh.points.PointCloud(vertices=final_pts, colors=PALETTE[vis_lbls]).export(os.path.join(output_dir, "ply", f"{part_name}.ply"))

if __name__ == "__main__":
    DATASET_ROOT = r"D:\MASTER THESIS\Data\raw\Machining_Tools" 
    PROCESSED_ROOT = r"D:\MASTER THESIS\Data\processed\2048\Machining_Tools"

    part_folders = [f.path for f in os.scandir(DATASET_ROOT) if f.is_dir()]
    print(f"\n🚀 Processing {len(part_folders)} folders using Standard Random Sampling...\n")
    
    for folder in tqdm(part_folders, desc="Processing Parts"):
        process_single_part(folder, PROCESSED_ROOT)
        
    print("\n✅ Dataset Fully Processed!")