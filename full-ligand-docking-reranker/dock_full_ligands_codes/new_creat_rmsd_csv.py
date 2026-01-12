import os
import sys
from pathlib import Path
import spyrmsd
from spyrmsd import rmsd
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import numpy as np
import pandas as pd

def compute_symmetry_corrected_rmsds_with_rdkit(docked_path_str: str, reference_path_str: str):
    try:
        ref_supplier = Chem.SDMolSupplier(reference_path_str, removeHs=False)
        ref_mols = []
        for mol in ref_supplier:
            if mol is not None:
                ref_mols.append(mol)
                break 
        if not ref_mols:
            print(f"Error: No valid molecules found in the reference file '{reference_path_str}'.")
            return None
        
        ref_mol = ref_mols[0]
        print(f"Loaded reference molecule: {ref_mol.GetProp('_Name') if ref_mol.HasProp('_Name') else 'Unknown'}")

        from spyrmsd.molecule import Molecule
        ref_spy_mol = Molecule.from_rdkit(ref_mol)
        if ref_spy_mol is None:
            print("Error: Could not convert reference molecule to spyrmsd format using from_rdkit.")
            return None

        ref_spy_mol.strip()
        print(f"Stripped Hs from reference. Remaining atoms: {ref_spy_mol.natoms}")

        docked_supplier = Chem.SDMolSupplier(docked_path_str, removeHs=False)
        docked_mols = []
        for mol in docked_supplier:
            if mol is not None:
                docked_mols.append(mol)
        if not docked_mols:
            print(f"Error: No valid molecules found in the docked file '{docked_path_str}'.")
            return None
        
        print(f"Loaded {len(docked_mols)} docked poses.")

        ref_coords = ref_spy_mol.coordinates
        ref_atomicnums = ref_spy_mol.atomicnums
        ref_adjacency_matrix = ref_spy_mol.adjacency_matrix

        rmsd_values = []

        for i, docked_rdkit_mol in enumerate(docked_mols):
            try:
                docked_spy_mol = Molecule.from_rdkit(docked_rdkit_mol)
                if docked_spy_mol is None:
                    print(f"Warning: Skipping pose {i+1}, could not convert to spyrmsd format using from_rdkit.")
                    rmsd_values.append(float('nan'))
                    continue

                docked_spy_mol.strip()
                print(f"  Processing Pose {i+1} ({docked_rdkit_mol.GetProp('_Name') if docked_rdkit_mol.HasProp('_Name') else f'Pose_{i+1}'}). Stripped Hs. Remaining atoms: {docked_spy_mol.natoms}")

                docked_coords = docked_spy_mol.coordinates
                docked_atomicnums = docked_spy_mol.atomicnums
                docked_adjacency_matrix = docked_spy_mol.adjacency_matrix

                if ref_coords.shape != docked_coords.shape or \
                   ref_atomicnums.shape != docked_atomicnums.shape or \
                   ref_adjacency_matrix.shape != docked_adjacency_matrix.shape:
                    print(f"    Error: Atom count mismatch after stripping Hs.")
                    print(f"      Ref: coords {ref_coords.shape}, atomicnums {ref_atomicnums.shape}, adj {ref_adjacency_matrix.shape}")
                    print(f"      Docked: coords {docked_coords.shape}, atomicnums {docked_atomicnums.shape}, adj {docked_adjacency_matrix.shape}")
                    rmsd_values.append(float('nan'))
                    continue

                sc_rmsd = rmsd.symmrmsd(
                    ref_coords,          
                    docked_coords,       
                    ref_atomicnums,      
                    docked_atomicnums,   
                    ref_adjacency_matrix, 
                    docked_adjacency_matrix, 
                )
                rmsd_values.append(sc_rmsd)
                mol_name = docked_rdkit_mol.GetProp('_Name') if docked_rdkit_mol.HasProp('_Name') else f'Pose_{i+1}'
                print(f"Pose {i+1} ({mol_name}): Symmetry-corrected RMSD = {sc_rmsd:.4f} Å")

            except Exception as e:
                mol_name = docked_rdkit_mol.GetProp('_Name') if docked_rdkit_mol.HasProp('_Name') else f'Pose_{i+1}'
                print(f"Error calculating RMSD for pose {i+1} ({mol_name}): {e}")
                import traceback
                traceback.print_exc()
                rmsd_values.append(float('nan'))

        return rmsd_values

    except Exception as e:
        print(f"An unexpected error occurred while processing the SDF files: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    
    if not (output_file.exists() and ligand_autobox_file.exists()):
        print(f"Warning: Required files not found in {dir_path}, skipping...")
        return None
    
    print(f"Processing directory: {dir_path}")
    print(f"Computing symmetry-corrected RMSDs for {full_dir_name}...")
    
    affinities = get_affinities_from_sdf(str(output_file))
    print(f"Found {len(affinities)} affinities in {output_file}")
    
    rmsd_values = compute_symmetry_corrected_rmsds_with_rdkit(str(output_file), str(ligand_autobox_file))
    
    if rmsd_values is not None:
        valid_rmsds = [r for r in rmsd_values if r is not None and not (isinstance(r, float) and np.isnan(r))]
        
        if valid_rmsds and len(affinities) > 0:
            sorted_rmsds = sorted(valid_rmsds)
            
            top1_rmsd = sorted_rmsds[0] if len(sorted_rmsds) > 0 else None
            
            rmsds_less_than_2 = [r for r in valid_rmsds if r < 2.0]
            percentage_less_than_2 = (len(rmsds_less_than_2) / len(valid_rmsds)) * 100 if valid_rmsds else 0
            
            sorted_affinities = sorted(affinities)
            
            best_affinity_idx = affinities.index(min(affinities))
            best_affinity_rmsd = rmsd_values[best_affinity_idx] if best_affinity_idx < len(rmsd_values) else None
            lowest_affinity = min(affinities)
            
            rmsd_indices = [(i, rmsd_values[i]) for i in range(len(rmsd_values))]
            sorted_by_rmsd = sorted(rmsd_indices, key=lambda x: x[1])
            
            poses_with_rmsd_less_than_2_by_rmsd = []
            for idx, (original_idx, rmsd_val) in enumerate(sorted_by_rmsd):
                if rmsd_val < 2.0:
                    pose_rank_by_affinity = sorted(range(len(affinities)), key=lambda i: affinities[i]).index(original_idx) + 1
                    poses_with_rmsd_less_than_2_by_rmsd.append(pose_rank_by_affinity)
            
            lowest_rmsd_idx = valid_rmsds.index(top1_rmsd) if top1_rmsd is not None else -1
            lowest_rmsd_pose_rank = sorted(range(len(affinities)), key=lambda i: affinities[i]).index(lowest_rmsd_idx) + 1 if lowest_rmsd_idx != -1 else -1
            
            is_best_affinity_lowest_rmsd = (best_affinity_rmsd == top1_rmsd)
            
            details_for_valid_poses_by_rmsd = []
            for idx, (original_idx, rmsd_val) in enumerate(sorted_by_rmsd):
                if rmsd_val < 2.0:
                    pose_rank_by_affinity = sorted(range(len(affinities)), key=lambda i: affinities[i]).index(original_idx) + 1
                    details_for_valid_poses_by_rmsd.append((pose_rank_by_affinity, rmsd_val, affinities[original_idx]))
            
            print(f"PDB ID {full_dir_name}:")
            print(f"  Lowest RMSD: {top1_rmsd:.4f} Å")
            print(f"  Lowest affinity RMSD: {best_affinity_rmsd:.4f} Å")
            print(f"  Lowest affinity: {lowest_affinity:.3f}")
            print(f"  Pose rank of lowest RMSD: {lowest_rmsd_pose_rank}")
            print(f"  Best affinity pose RMSD > 2 Å: {best_affinity_rmsd > 2.0 if best_affinity_rmsd is not None else 'N/A'}")
            print(f"  Best affinity pose is lowest RMSD: {is_best_affinity_lowest_rmsd}")
            print(f"  Poses with RMSD < 2 Å (sorted by RMSD): {poses_with_rmsd_less_than_2_by_rmsd}")
            print(f"  Details for poses with RMSD < 2 Å (sorted by RMSD): {[(p, f'{r:.4f}', f'{a:.3f}') for p, r, a in details_for_valid_poses_by_rmsd]}")
            
            return {
                'pdb_id': full_dir_name,
                'top1_rmsd': top1_rmsd,
                'best_affinity_rmsd': best_affinity_rmsd,
                'lowest_affinity': lowest_affinity,
                'percentage_less_than_2': percentage_less_than_2,
                'total_poses': len(valid_rmsds),
                'valid_poses': len(rmsds_less_than_2),
                'affinities': affinities,
                'sorted_affinities': sorted_affinities,
                'poses_with_rmsd_less_than_2_by_rmsd': poses_with_rmsd_less_than_2_by_rmsd,
                'lowest_rmsd_pose_rank': lowest_rmsd_pose_rank,
                'is_best_affinity_lowest_rmsd': is_best_affinity_lowest_rmsd,
                'details_for_valid_poses_by_rmsd': details_for_valid_poses_by_rmsd,
                'rmsd_values': rmsd_values
            }
        else:
            print(f"No valid RMSD values or affinities computed for {full_dir_name}")
            return None
    else:
        print(f"Failed to compute RMSDs for {full_dir_name}")
        return None

def main():
    current_dir = Path('.')
    depth1_dirs = [d for d in current_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(depth1_dirs)} directories to process")
    
    results = []
    for dir_path in depth1_dirs:
        result = process_single_directory(dir_path)
        if result is not None:
            results.append(result)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    for result in results:
        print(f"PDB ID: {result['pdb_id']}")
        print(f"  Lowest RMSD: {result['top1_rmsd']:.4f} Å")
        print(f"  Lowest affinity RMSD: {result['best_affinity_rmsd']:.4f} Å")
        print(f"  Lowest affinity: {result['lowest_affinity']:.3f}")
        print(f"  Pose rank of lowest RMSD: {result['lowest_rmsd_pose_rank']}")
        print(f"  Best affinity pose RMSD > 2 Å: {result['best_affinity_rmsd'] > 2.0 if result['best_affinity_rmsd'] is not None else 'N/A'}")
        print(f"  Best affinity pose is lowest RMSD: {result['is_best_affinity_lowest_rmsd']}")
        print(f"  Poses with RMSD < 2 Å (sorted by RMSD): {result['poses_with_rmsd_less_than_2_by_rmsd']}")
        print(f"  Details for poses with RMSD < 2 Å (sorted by RMSD): {[(p, f'{r:.4f}', f'{a:.3f}') for p, r, a in result['details_for_valid_poses_by_rmsd']]}")
        print("-" * 50)
    
    # Count PDB IDs with lowest RMSD < 2.0 Å
    pdb_ids_with_low_rmsd = [r for r in results if r['top1_rmsd'] is not None and r['top1_rmsd'] < 2.0]
    num_pdb_ids_with_low_rmsd = len(pdb_ids_with_low_rmsd)
    
    best_affinity_rmsd_over_2 = sum(1 for r in results if r['best_affinity_rmsd'] is not None and r['best_affinity_rmsd'] > 2.0)
    not_lowest_rmsd = sum(1 for r in results if r['is_best_affinity_lowest_rmsd'] == False)
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total systems processed: {len(results)}")
    print(f"Number of systems where best affinity pose RMSD > 2 Å: {best_affinity_rmsd_over_2}")
    print(f"Number of systems where best affinity pose is NOT the lowest RMSD: {not_lowest_rmsd}")
    print(f"Number of PDB IDs with lowest RMSD < 2.0 Å: {num_pdb_ids_with_low_rmsd}")
    print(f"Percentage of systems where best affinity pose RMSD > 2 Å: {best_affinity_rmsd_over_2/len(results)*100:.2f}%")
    print(f"Percentage of systems where best affinity pose is NOT the lowest RMSD: {not_lowest_rmsd/len(results)*100:.2f}%")
    print(f"Percentage of PDB IDs with lowest RMSD < 2.0 Å: {num_pdb_ids_with_low_rmsd/len(results)*100:.2f}%")
    
    # Create detailed CSV with PDB_ID, RMSD, Vina_Score, Vina_Rank (removed Pose column)
    csv_rows = []
    for result in results:
        pdb_id = result['pdb_id']
        affinities = result['affinities']
        rmsd_values = result['rmsd_values']
        
        # Get sorted indices by affinity (to create ranks)
        sorted_indices_by_affinity = sorted(range(len(affinities)), key=lambda i: affinities[i])
        
        for i in range(len(affinities)):
            vina_score = affinities[i]
            vina_rank = sorted_indices_by_affinity.index(i) + 1
            rmsd_value = rmsd_values[i] if i < len(rmsd_values) else float('nan')
            
            csv_rows.append({
                'PDB_ID': pdb_id,
                'RMSD': round(rmsd_value, 4) if not np.isnan(rmsd_value) else None,
                'Vina_Score': round(vina_score, 3),
                'Vina_Rank': vina_rank
            })
    
    df = pd.DataFrame(csv_rows)
    df.to_csv("exhaust50_detailed_results.csv", index=False)
    print(f"\nDetailed results saved to 'exhaust50_detailed_results.csv'")

if __name__ == "__main__":
    main()
