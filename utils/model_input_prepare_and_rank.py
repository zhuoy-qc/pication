#run this code in the generated dir e.g 3FX4_prepared
import os
import sys
import glob
import csv
import math
import numpy as np
import pandas as pd
import joblib
from decimal import Decimal
from multiprocessing import Pool, cpu_count, set_start_method
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit import RDLogger
from plip.structure.preparation import PDBComplex
from pathlib import Path
from tqdm import tqdm
import tempfile
import shutil
import warnings

# Filter out all warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

# Set RDKit log level to suppress unnecessary warnings
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.info')

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProteinSelect(Select):
    def accept_model(self, model):
        """Accept all models"""
        return True

    def accept_chain(self, chain):
        """Accept all chains"""
        return True

    def accept_residue(self, residue):
        """Accept only standard amino acid residues"""
        return is_aa(residue, standard=True)

def clean_pdb(input_file, output_file='pure_protein.pdb'):
    """
    Read PDB file, remove all non-protein residues, and save clean protein structure.

    Parameters:
        input_file (str): Input PDB file path
        output_file (str): Output clean protein PDB file path (default 'pure_protein.pdb')
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', input_file)

        io = PDBIO()
        io.set_structure(structure)
        io.save(output_file, ProteinSelect())
        logger.info(f"✅ Clean protein structure saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"❌ Failed to clean PDB file {input_file}: {e}")
        return None

def find_protein_files(directory):
    """
    Find all *_protein_H.pdb files in the specified directory.
    """
    protein_files = glob.glob(os.path.join(directory, "*_protein_H.pdb"))
    if protein_files:
        logger.info(f"   Found {len(protein_files)} protein_H files in {os.path.basename(directory)}")
        for f in protein_files:
            logger.info(f"     - {os.path.basename(f)}")
        return protein_files
    else:
        logger.info(f"   No protein_H files found in {os.path.basename(directory)}")
        return []

def find_docked_sdf_files(directory):
    """
    Find all docked_ring_poses.sdf files in the specified directory
    """
    sdf_files = glob.glob(os.path.join(directory, "docked_ring_poses.sdf"))
    if sdf_files:
        logger.info(f"   Found {len(sdf_files)} docked ring poses SDF files in {os.path.basename(directory)}")
        for f in sdf_files:
            logger.info(f"     - {os.path.basename(f)}")
    return sdf_files

def create_purified_protein(protein_file, purified_protein_file):
    """
    Create a purified protein file by removing non-protein residues
    """
    try:
        purified = clean_pdb(protein_file, purified_protein_file)
        if purified:
            logger.info(f"✅ Purified protein created: {purified_protein_file}")
            return purified_protein_file
        else:
            logger.error(f"❌ Failed to create purified protein: {purified_protein_file}")
            return None
    except Exception as e:
        logger.error(f"❌ Error purifying protein {protein_file}: {e}")
        return None

def create_complexes_for_pair(protein_file, docked_sdf, output_dir):
    """Create complexes for a single protein and docking result"""
    os.makedirs(output_dir, exist_ok=True)

    protein_base = os.path.basename(protein_file).replace('_protein_H.pdb', '')
    docked_base = os.path.basename(docked_sdf).replace('.sdf', '')

    # Create purified protein file first
    purified_protein_file = os.path.join(output_dir, f"{protein_base}_purified.pdb")
    purified_protein = create_purified_protein(protein_file, purified_protein_file)
    
    if not purified_protein:
        logger.error(f"❌ Skipping complex creation for {protein_file} due to purification failure")
        return []

    supplier = Chem.SDMolSupplier(docked_sdf)
    successful_complexes = []

    # Add progress bar
    molecules = list(supplier)
    with tqdm(total=len(molecules), desc=f"Creating complexes {protein_base}_{docked_base}",
              unit="complex", leave=False) as pbar:
        for i, mol in enumerate(molecules):
            if mol is None:
                pbar.update(1)
                continue

            output_file = os.path.join(output_dir, f"{protein_base}_{docked_base}_complex_{i+1}.pdb")

            try:
                with open(purified_protein, 'r') as protein, open(output_file, 'w') as output:
                    for line in protein:
                        if line.startswith(('ATOM', 'HETATM')):
                            output.write(line)

                    pdb_block = Chem.MolToPDBBlock(mol)
                    output.write(pdb_block)
                    output.write("END\n")

                successful_complexes.append(output_file)
            except Exception as e:
                logger.error(f"❌ Error creating complex {i+1}: {e}")
            finally:
                pbar.update(1)

    return successful_complexes

def process_directory(directory):
    """
    Process a single directory: generate protein-ligand complexes
    """
    logger.info(f"\n🔍 Processing directory: {directory}")
    logger.info("-" * 50)

    protein_files = find_protein_files(directory)
    docked_sdf_files = find_docked_sdf_files(directory)

    if not protein_files or not docked_sdf_files:
        logger.info(f"   Skipping directory {os.path.basename(directory)}: missing protein_H or docked ring poses files")
        if not protein_files:
            logger.info("   - No protein_H files found")
        if not docked_sdf_files:
            logger.info("   - No docked ring poses SDF files found")
        return []

    complex_dirs = []

    # Add overall progress bar
    total_pairs = len(protein_files) * len(docked_sdf_files)
    with tqdm(total=total_pairs, desc=f"Processing {os.path.basename(directory)}", unit="pair") as dir_pbar:
        for protein_file in protein_files:
            protein_base = os.path.basename(protein_file).replace('_protein_H.pdb', '')

            for docked_sdf in docked_sdf_files:
                docked_base = os.path.basename(docked_sdf).replace('.sdf', '')

                output_dir = os.path.join(directory, f"complexes_{protein_base}_{docked_base}")
                complex_dirs.append(output_dir)

                logger.info(f"🔗 Creating complexes for {protein_base} + {docked_base}")

                complexes = create_complexes_for_pair(protein_file, docked_sdf, output_dir)
                logger.info(f"   Created {len(complexes)} complexes in {os.path.basename(output_dir)}")

                dir_pbar.update(1)

    return complex_dirs

def calculate_angle(ring_normal, charge_vector):
    """Calculate angle between ring normal and charge vector"""
    dot_product = Decimal(ring_normal[0]) * Decimal(charge_vector[0]) + \
                  Decimal(ring_normal[1]) * Decimal(charge_vector[1]) + \
                  Decimal(ring_normal[2]) * Decimal(charge_vector[2])
    norm_ring = Decimal(math.sqrt(sum(x**2 for x in ring_normal)))
    norm_charge = Decimal(math.sqrt(sum(x**2 for x in charge_vector)))
    cos_theta = dot_product / (norm_ring * norm_charge)
    cos_theta = max(min(float(cos_theta), 1.0), -1.0)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    # Return both original angle and adjusted angle (if >90, use 180-angle)
    original_angle = angle_deg
    adjusted_angle = 180 - angle_deg if angle_deg > 90 else angle_deg

    return original_angle, adjusted_angle

def calculate_rz(distance, offset):
    """Calculate RZ value"""
    try:
        rz_squared = float(distance)**2 - float(offset)**2
        return math.sqrt(max(rz_squared, 0))
    except (ValueError, TypeError):
        return float('nan')

def analyze_pication_interactions(pdb_file_path):
    """Analyze π-cation interactions, including ring center coordinates"""
    from io import StringIO
    import sys

    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        pdb_file = Path(pdb_file_path)
        dir_name = pdb_file.parent.name

        my_mol = PDBComplex()
        my_mol.load_pdb(str(pdb_file))
        my_mol.analyze()
        results = []

        if not hasattr(my_mol, 'interaction_sets') or not my_mol.interaction_sets:
            return results

        for bs_id, interactions in my_mol.interaction_sets.items():
            if hasattr(interactions, 'pication_laro'):
                for pication in interactions.pication_laro:
                    try:
                        ring_normal = np.array(pication.ring.normal, dtype=np.float64)
                        charge_vector = np.array(pication.charge.center, dtype=np.float64) - np.array(pication.ring.center, dtype=np.float64)
                        original_angle, adjusted_angle = calculate_angle(ring_normal, charge_vector)
                        rz = calculate_rz(pication.distance, pication.offset)

                        # Include ring center coordinate detailed data
                        interaction_data = {
                            'Directory': dir_name,
                            'PDB_File': pdb_file.name,
                            'Binding_Site': bs_id,
                            'Ligand': f"{pication.restype_l}-{pication.resnr_l}-{pication.reschain_l}",
                            'Protein': f"{pication.restype}-{pication.resnr}-{pication.reschain}",
                            'Distance': round(float(pication.distance), 2),
                            'Offset': round(float(pication.offset), 2),
                            'RZ': round(rz, 2) if not math.isnan(rz) else float('nan'),
                            'Angle': round(original_angle, 2),  # Record original angle in CSV
                            'Adjusted_Angle': round(adjusted_angle, 2),  # Add adjusted angle for reference
                            'Ring_Center_X': round(float(pication.ring.center[0]), 3),
                            'Ring_Center_Y': round(float(pication.ring.center[1]), 3),
                            'Ring_Center_Z': round(float(pication.ring.center[2]), 3),
                            'Charged_Center_X': round(float(pication.charge.center[0]), 3),
                            'Charged_Center_Y': round(float(pication.charge.center[1]), 3),
                            'Charged_Center_Z': round(float(pication.charge.center[2]), 3),
                            'Ring_Normal_X': round(float(ring_normal[0]), 3),
                            'Ring_Normal_Y': round(float(ring_normal[1]), 3),
                            'Ring_Normal_Z': round(float(ring_normal[2]), 3),
                            'Ring_Type': pication.ring.type,
                            'Atom_Indices': str(pication.ring.atoms_orig_idx),
                            'Interaction_Type': 'π-Cation'
                        }

                        results.append(interaction_data)
                    except Exception as e:
                        logger.error(f"Error processing π-cation interaction: {e}")

        sys.stderr = old_stderr
        return results

    except Exception as e:
        sys.stderr = old_stderr
        raise e

def process_single_pdb(pdb_file):
    """Process a single PDB file, only record results with interactions"""
    try:
        results = analyze_pication_interactions(pdb_file)
        if results:
            logger.info(f"✅ Processed {os.path.basename(pdb_file)}: {len(results)} π-cation interactions")
        return results
    except Exception as e:
        logger.error(f"❌ Error processing {os.path.basename(pdb_file)}: {str(e)}")
        return []

def process_all_complex_dirs(complex_dirs):
    """Process all PDB files in complex directories, only output files with interactions"""
    all_pdb_files = []

    for complex_dir in complex_dirs:
        pdb_files = glob.glob(os.path.join(complex_dir, "*_complex_*.pdb"))
        all_pdb_files.extend(pdb_files)

    if not all_pdb_files:
        logger.error("❌ No complex PDB files found!")
        return None

    logger.info(f"🔍 Found {len(all_pdb_files)} complex PDB files to process")
    num_processes = min(90, cpu_count())
    logger.info(f"  Using {num_processes} CPU cores")

    all_results = []

    # Add multiprocessing progress bar
    with Pool(processes=num_processes) as pool:
        # Use tqdm to create progress bar
        with tqdm(total=len(all_pdb_files), desc="Analyzing π-cation interactions", unit="file") as pbar:
            for results in pool.imap_unordered(process_single_pdb, all_pdb_files):
                if results:
                    all_results.extend(results)
                pbar.update(1)

    if all_results:
        # Create temporary CSV file for model input
        temp_csv_file = None
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', prefix='temp_pication_') as temp_file:
            temp_csv_file = temp_file.name
            fieldnames = list(all_results[0].keys())
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        logger.info(f"\n📊 Total interactions found across all complexes: {len(all_results)}")
        logger.info(f"💾 Temporary results saved to {temp_csv_file}")
        return temp_csv_file
    else:
        logger.info("\n  No π-cation interactions found across all complexes.")
        return None

def find_model_file(base_dir):
    """Find the model file by searching upward from base directory"""
    model_dir = None
    current_path = os.path.abspath(base_dir)
    while current_path != os.path.dirname(current_path):
        # Look for files that start with 'final_model' and end with '.pkl'
        model_candidates = glob.glob(os.path.join(current_path, 'final_model_*.pkl'))
        if model_candidates:
            # Return the most recent one if multiple exist
            model_dir = sorted(model_candidates, reverse=True)[0]
            break
        current_path = os.path.dirname(current_path)
    
    return model_dir

def run_model_prediction(input_csv_path, base_dir):
    # Find the model file automatically
    model_path = find_model_file(base_dir)
    if model_path is None:
        logger.error("❌ Could not find model file (final_model_*.pkl) in directory tree")
        return
    
    print(f"✅ Found model path: {model_path}")
    logger.info(f"✅ Found model file: {model_path}")
    output_csv_path = 'predictions_with_energy_ranked.csv'

    try:
        model = joblib.load(model_path)
        logger.info(f"✅ Successfully loaded model: {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    try:
        df = pd.read_csv(input_csv_path)
        logger.info(f"✅ Successfully read CSV file: {input_csv_path}")
        logger.info(f"📊 Data shape: {df.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to read CSV file: {e}")
        return

    # Use Adjusted_Angle for model input (180-angle if original > 90)
    required_columns_in_csv = ['Offset', 'RZ', 'Adjusted_Angle', 'Distance']
    if not all(col in df.columns for col in required_columns_in_csv):
        missing_cols = [col for col in required_columns_in_csv if col not in df.columns]
        logger.error(f"❌ CSV file is missing required columns: {missing_cols}")
        return

    # Create model input feature DataFrame (using original precision)
    X_input = df[required_columns_in_csv].copy()
    feature_mapping = {
        'Offset': 'delta_x',
        'RZ': 'delta_z',
        'Adjusted_Angle': 'dihedral_angle',  # Use adjusted angle for model input
        'Distance': 'distance'
    }
    X_input.rename(columns=feature_mapping, inplace=True)

    if X_input.isnull().any().any():
        logger.info("   Input data contains missing values, processing...")
        X_input_clean = X_input.dropna()
        logger.info(f"📊 Cleaned feature data shape: {X_input_clean.shape}")
    else:
        X_input_clean = X_input
        logger.info("✅ Input data contains no missing values.")

    try:
        logger.info("\n🔮 Making energy predictions...")
        expected_feature_order = ['delta_z', 'delta_x', 'dihedral_angle', 'distance']
        X_input_clean = X_input_clean[expected_feature_order]

        # Add prediction progress bar
        predicted_energies = []
        with tqdm(total=len(X_input_clean), desc="Predicting energies", unit="sample") as pbar:
            for i in range(0, len(X_input_clean), 1000):  # Process in batches to avoid memory issues
                batch = X_input_clean.iloc[i:i+1000]
                batch_pred = model.predict(batch)
                predicted_energies.extend(batch_pred)
                pbar.update(len(batch))

        logger.info("✅ Prediction completed!")
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        return

    # Add prediction results back to original DataFrame
    df_pred = df.loc[X_input_clean.index].copy()  # Fixed: Use square brackets instead of parentheses
    df_pred['Predicted_Energy'] = predicted_energies

    # Rank by PDB ID group (rank within each PDB file)
    df_pred['PDB_ID'] = df_pred['PDB_File'].str.extract(r'([^_]+_[^_]+)_')[0]  # Extract PDB ID

    # Rank interactions separately for each PDB ID
    df_pred['Energy_Rank'] = df_pred.groupby('PDB_ID')['Predicted_Energy'].rank(ascending=True, method='dense').astype(int)

    # Sort by PDB ID and rank
    df_pred_sorted = df_pred.sort_values(['PDB_ID', 'Energy_Rank'], ascending=[True, True])
    df_pred_sorted.reset_index(drop=True, inplace=True)

    # Save results to new CSV file (all numerical values rounded to 2 decimal places)
    try:
        # Round numerical columns
        numeric_cols = ['Distance', 'Offset', 'RZ', 'Angle', 'Adjusted_Angle', 'Predicted_Energy',
                       'Ring_Center_X', 'Ring_Center_Y', 'Ring_Center_Z',
                       'Charged_Center_X', 'Charged_Center_Y', 'Charged_Center_Z',
                       'Ring_Normal_X', 'Ring_Normal_Y', 'Ring_Normal_Z']
        for col in numeric_cols:
            if col in df_pred_sorted.columns:
                df_pred_sorted[col] = df_pred_sorted[col].round(2)

        df_pred_sorted.to_csv(output_csv_path, index=False)
        logger.info(f"💾 Results saved to: {output_csv_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save results file: {e}")

    # Display top 10 prediction results
    logger.info("\n🏆 Top 10 complexes (sorted by PDB ID and energy rank, lower is more stable):")
    cols_to_display = ['PDB_ID', 'Energy_Rank', 'Predicted_Energy', 'Ligand', 'Protein',
                      'Distance', 'Offset', 'RZ', 'Angle', 'Adjusted_Angle', 'Ring_Center_X', 'Ring_Center_Y', 'Ring_Center_Z']
    top_10 = df_pred_sorted.head(10)[cols_to_display].to_string(index=False)
    for line in top_10.split('\n'):
        logger.info(line)

    logger.info(f"\n🎉 Processing complete! Ranked results saved to {output_csv_path}")
    return output_csv_path

def display_final_results(csv_file_path):
    """Display final ranking results (grouped by PDB ID)"""
    try:
        df = pd.read_csv(csv_file_path)
        logger.info(f"\n🎯 FINAL π-CATION INTERACTION RANKING RESULTS (Grouped by PDB ID)")
        logger.info("=" * 80)

        # Display top 20 results (grouped by PDB ID)
        logger.info("\n🏆 TOP RANKED INTERACTIONS BY PDB ID:")
        logger.info("-" * 80)

        # Get the top-ranked interaction for each PDB ID
        top_interactions = df.loc[df.groupby('PDB_ID')['Energy_Rank'].idxmin()]

        display_columns = ['PDB_ID', 'Energy_Rank', 'Predicted_Energy', 'Ligand', 'Protein',
                          'Distance', 'Offset', 'RZ', 'Angle', 'Adjusted_Angle', 'Ring_Center_X', 'Ring_Center_Y', 'Ring_Center_Z']

        available_columns = [col for col in display_columns if col in df.columns]

        top_20 = top_interactions.head(20)[available_columns]
        for _, row in top_20.iterrows():
            logger.info(f"PDB: {row['PDB_ID']} | Rank {row['Energy_Rank']:2d} | Energy: {row['Predicted_Energy']:8.4f} | "
                       f"Ligand: {row['Ligand']} | Protein: {row['Protein']} | "
                       f"Dist: {row['Distance']} | Offset: {row['Offset']} | "
                       f"RZ: {row['RZ']} | Angle: {row['Angle']}° | Adj Angle: {row['Adjusted_Angle']}° | "
                       f"Ring Center: ({row['Ring_Center_X']}, {row['Ring_Center_Y']}, {row['Ring_Center_Z']})")

        # Display statistics
        logger.info("\n📊 STATISTICS:")
        logger.info("-" * 80)
        logger.info(f"Total interactions analyzed: {len(df)}")
        logger.info(f"Number of unique PDB files: {df['PDB_ID'].nunique()}")
        logger.info(f"Minimum predicted energy: {df['Predicted_Energy'].min():.4f}")
        logger.info(f"Maximum predicted energy: {df['Predicted_Energy'].max():.4f}")
        logger.info(f"Average predicted energy: {df['Predicted_Energy'].mean():.4f}")
        logger.info(f"Standard deviation: {df['Predicted_Energy'].std():.4f}")

        logger.info(f"\n💾 Complete results saved to: {os.path.abspath(csv_file_path)}")

    except Exception as e:
        logger.error(f"❌ Error displaying final results: {e}")

def cleanup_generated_files_recursive(base_dir):
    """Remove generated files in specified formats from base directory, parent directory, and all subdirectories"""
    patterns_to_remove = [
        'plipfixed.*.pdb',
        '*_complex_*_protonated.pdb',
        '*_purified.pdb'  # Add purified protein files to cleanup
    ]
    
    files_removed = 0
    
    # Clean up in base directory and all subdirectories
    for pattern in patterns_to_remove:
        # Search in base directory and all subdirectories
        pattern_path = os.path.join(base_dir, '**', pattern)
        files = glob.glob(pattern_path, recursive=True)
        for file_path in files:
            try:
                os.remove(file_path)
                logger.info(f"🗑️  Removed generated file: {file_path}")
                files_removed += 1
            except Exception as e:
                logger.warning(f"⚠️  Could not remove file {file_path}: {e}")
    
    # Also clean up in parent directory
    parent_dir = os.path.dirname(base_dir)
    for pattern in patterns_to_remove:
        pattern_path = os.path.join(parent_dir, pattern)
        files = glob.glob(pattern_path)
        for file_path in files:
            try:
                os.remove(file_path)
                logger.info(f"🗑️  Removed generated file from parent directory: {file_path}")
                files_removed += 1
            except Exception as e:
                logger.warning(f"⚠️  Could not remove file from parent directory {file_path}: {e}")
    
    if files_removed > 0:
        logger.info(f"✅ Cleaned up {files_removed} generated files from directory tree and parent directory")
    else:
        logger.info("✅ No generated files to clean up")

def print_logo():
    logo = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    RRRRRRR  IIIIIIII  NNN    NN  GGGGGGG     ║
    ║                    RR   RR     II     NNNN   NN  GG          ║
    ║                    RRRRRR      II     NN NN  NN  GG  GGG     ║
    ║                    RR  RR      II     NN  NN NN  GG    GG    ║
    ║                    RR   RR  IIIIIIII  NN   NNNN   GGGGGGG    ║
    ║                                                              ║
    ║     DDDDDDD  OOOOOOO   CCC    K   K                          ║
    ║     DD   DD OO   OO  CC       K  K                           ║
    ║     DD   DD OO   OO CC        KKK                            ║
    ║     DD   DD OO   OO CC        K  K                           ║
    ║     DDDDDDD  OOOOOOO   CCC    K   K                          ║
    ║                                                              ║
    ║  Tools to dock ligand aromatic rings fast and accruately     ║
    ║                   Version 1.0.0                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(logo)

# --- Main execution flow ---
if __name__ == "__main__":
    set_start_method('fork', force=True)

    print_logo()

    if len(sys.argv) > 1:
        base_directory = sys.argv[1]
    else:
        base_directory = os.getcwd()

    logger.info(f"📁 Base directory: {base_directory}")

    # Check what files are in the current directory
    files_in_dir = os.listdir(base_directory)
    logger.info(f"📁 Files in directory: {files_in_dir}")
    
    # Step 1: Generate complexes for the current directory
    print("  STEP 1: GENERATING PROTEIN-LIGAND COMPLEXES IN CURRENT DIRECTORY")
    print("-" * 50)
    all_complex_dirs = process_directory(base_directory)

    if not all_complex_dirs:
        logger.error("❌ No complexes were generated in the directory")
        sys.exit(1)

    # PI-CATION RANGE Filtering and model input preparation
    print("  STEP 2: PI-CATION RANGE FILTERING AND MODEL INPUT PREPARATION")
    print("-" * 50)
    temp_csv_file = process_all_complex_dirs(all_complex_dirs)

    # MODEL PREDICTION AND RANK
    final_results_file = None
    if temp_csv_file:
        print("  STEP 3: MODEL PREDICTION AND RANK")
        print("-" * 50)
        final_results_file = run_model_prediction(temp_csv_file, base_directory)
        
        # Clean up temporary file after use
        try:
            os.unlink(temp_csv_file)
            logger.info(f"🗑️  Temporary file {temp_csv_file} deleted")
        except Exception as e:
            logger.warning(f"⚠️  Could not delete temporary file {temp_csv_file}: {e}")

    # Clean up generated files in the root directory, parent directory, and all subdirectories
    cleanup_generated_files_recursive(base_directory)
 
    print("🎯 ALL STEPS COMPLETED SUCCESSFULLY!")
