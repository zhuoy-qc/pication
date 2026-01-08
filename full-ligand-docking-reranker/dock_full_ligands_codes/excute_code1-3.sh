#!/usr/bin/env python3
"""
Script to execute the molecular docking pipeline in sequence:
1. perfrom_extensive_vina_sample.py
2. new_creat_rmsd_csv.py
3. predict_interaction_energies.py
4. run_rerank.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name, description=""):
    """Execute a Python script and handle errors."""
    print(f"Running {script_name} {description}...")
    try:
        result = subprocess.run([sys.executable, script_name], 
                              check=True, 
                              capture_output=True, 
                              text=True)
        print(f"✓ Successfully completed {script_name}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}:")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ Script not found: {script_name}")
        return False

def main():
    # Define the scripts in execution order
    scripts = [
        ("perfrom_extensive_vina_sample.py", "(Step 1: Extensive Vina sampling)"),
        ("new_creat_rmsd_csv.py", "(Step 2: Create RMSD CSV)"),
        ("predict_interaction_energies.py", "(Step 3: Predict interaction energies)"),
        ("run_rerank.py", "(Step 4: Run re-ranking)")
    ]
    
    print("Starting molecular docking pipeline execution...")
    print("=" * 50)
    
    # Check if all scripts exist before starting
    missing_scripts = []
    for script, _ in scripts:
        if not Path(script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"Error: The following scripts were not found:")
        for script in missing_scripts:
            print(f"  - {script}")
        print("Please ensure all scripts are in the current directory.")
        return 1
    
    # Execute scripts in order
    for script_name, description in scripts:
        print(f"\n{'='*20}")
        success = run_script(script_name, description)
        if not success:
            print(f"Pipeline stopped due to error in {script_name}")
            return 1
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
