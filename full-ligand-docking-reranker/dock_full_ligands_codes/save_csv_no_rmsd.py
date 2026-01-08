import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def get_pdb_id_from_dir_name(dir_path):
    dir_name = dir_path.name
    if len(dir_name) >= 4:
        return dir_name[:4]
    else:
        return None

def get_affinities_from_sdf(sdf_path):
    affinities = []
    with open(sdf_path, 'r') as f:
        content = f.read()
    
    molecules = content.split('$$$$')
    
    for mol_block in molecules:
        lines = mol_block.split('\n')
        for line in lines:
            if '<minimizedAffinity>' in line:
                idx = lines.index(line)
                if idx + 1 < len(lines):
                    try:
                        affinity = float(lines[idx + 1].strip())
                        affinities.append(affinity)
                    except ValueError:
                        print(f"Could not parse affinity: {lines[idx + 1].strip()}")
                break
    
    return affinities

def process_single_directory(dir_path):
    dir_path = Path(dir_path)
    full_dir_name = dir_path.name
    
    output_file = dir_path / "exhaust50_dock.sdf"
    ligand_autobox_file = dir_path / f"{full_dir_name}_ligand.sdf"
    
    print(f"Checking files in {dir_path}:")
    print(f"  Full directory name: {full_dir_name}")
    print(f"  Output file: {output_file} - exists: {output_file.exists()}")
    print(f"  Ligand autobox file: {ligand_autobox_file} - exists: {ligand_autobox_file.exists()}")
    
    if not (output_file.exists()):
        print(f"Warning: Required files not found in {dir_path}, skipping...")
        return None
    
    print(f"Processing directory: {dir_path}")
    
    affinities = get_affinities_from_sdf(str(output_file))
    print(f"Found {len(affinities)} affinities in {output_file}")
    
    if len(affinities) > 0:
        sorted_affinities = sorted(affinities)
        lowest_affinity = min(affinities)
        
        # Get sorted indices by affinity (to create ranks)
        sorted_indices_by_affinity = sorted(range(len(affinities)), key=lambda i: affinities[i])
        
        print(f"PDB ID {full_dir_name}:")
        print(f"  Lowest affinity: {lowest_affinity:.3f}")
        print(f"  Total poses: {len(affinities)}")
        
        # Create pose data with empty RMSD values
        pose_data = []
        for i in range(len(affinities)):
            vina_score = affinities[i]
            vina_rank = sorted_indices_by_affinity.index(i) + 1
            
            pose_data.append({
                'PDB_ID': full_dir_name,
                'RMSD': None,  # Empty RMSD values
                'Vina_Score': round(vina_score, 3),
                'Vina_Rank': vina_rank
            })
        
        return pose_data
    else:
        print(f"No affinities found for {full_dir_name}")
        return None

def main():
    current_dir = Path('.')
    depth1_dirs = [d for d in current_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(depth1_dirs)} directories to process")
    
    all_pose_data = []
    for dir_path in depth1_dirs:
        result = process_single_directory(dir_path)
        if result is not None:
            all_pose_data.extend(result)
    
    # Create DataFrame with all pose data
    df = pd.DataFrame(all_pose_data)
    
    # Save to CSV with no RMSD values
    df.to_csv("no_rmsd.csv", index=False)
    print(f"\nResults saved to 'no_rmsd.csv'")
    
    print(f"\nTotal number of poses processed: {len(df)}")
    print(f"Unique PDB IDs: {df['PDB_ID'].nunique()}")
    
    # Print summary
    print("\nSummary by PDB ID:")
    for pdb_id in df['PDB_ID'].unique():
        pdb_data = df[df['PDB_ID'] == pdb_id]
        print(f"  {pdb_id}: {len(pdb_data)} poses")

if __name__ == "__main__":
    main()
