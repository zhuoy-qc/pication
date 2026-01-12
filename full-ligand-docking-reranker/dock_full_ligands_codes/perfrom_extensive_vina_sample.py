import os
import subprocess
import sys
from pathlib import Path
import numpy as np

def get_pdb_id_from_dir_name(dir_path):
    """Extract PDB ID from directory name (first 4 characters)"""
    dir_name = dir_path.name  # Use .name instead of rstrip('/')
    if len(dir_name) >= 4:
        return dir_name[:4]
    else:
        return None

def process_single_directory(dir_path):
    """Process a single directory with smina"""
    dir_path = Path(dir_path)
    full_dir_name = dir_path.name  # e.g., "7Q5I_I0F"

    protein_file = dir_path / f"{full_dir_name}_protein_protonated.pdb"
    ligand_start_file = dir_path / f"{full_dir_name}_ligand_start_conf.sdf"
    ligand_autobox_file = dir_path / f"{full_dir_name}_ligand.sdf"
    output_file = dir_path / "exhaust50_dock.sdf"  # Changed output file name


    # Run smina command with additional parameters
    cmd = [
        "smina",
        "-r", f"{full_dir_name}_protein_protonated.pdb",
        "-l", f"{full_dir_name}_ligand.sdf",
        "--autobox_ligand", f"{full_dir_name}_ligand.sdf",
        "--num_modes", "40",
        "--energy_range", "10",
        "--exhaustiveness", "50",
        "-o", "exhaust50_dock.sdf",  # output file name
        "--seed", "1"
    ]

    try:
        result = subprocess.run(cmd, cwd=dir_path, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running smina in {dir_path}: {result.stderr}")
            print(f"Command: {' '.join(cmd)}")
            return None
        print(f"smina completed successfully for {full_dir_name}")
    except Exception as e:
        print(f"Error running smina in {dir_path}: {e}")
        print(f"Command: {' '.join(cmd)}")
        return None

    # Return basic information without RMSD computation
    print(f"Processing completed for {full_dir_name}")
    
    return {
        'pdb_id': full_dir_name,
        'success': True
    }

def main():
    # Get all directories at depth=1 from current directory
    current_dir = Path('.')
    depth1_dirs = [d for d in current_dir.iterdir() if d.is_dir()]

    print(f"Found {len(depth1_dirs)} directories to process")

    # Process directories sequentially (single CPU)
    results = []
    for dir_path in depth1_dirs:
        result = process_single_directory(dir_path)
        if result is not None:
            results.append(result)

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for result in results:
        print(f"PDB ID: {result['pdb_id']}")
        print(f"  Status: {'Success' if result['success'] else 'Failed'}")
        print("-" * 50)

    # Save results to a single text file
    with open("autobox4_results_summary.txt", "w") as f:
        f.write("EXHAUST50 DOCKING RESULTS SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Total directories processed: {len(results)}\n")
        f.write(f"Directories with successful completion: {len([r for r in results if r is not None and r['success']])}\n")
        f.write("="*70 + "\n")

        for result in results:
            f.write(f"PDB ID: {result['pdb_id']}\n")
            f.write(f"  Status: {'Success' if result['success'] else 'Failed'}\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    main()
