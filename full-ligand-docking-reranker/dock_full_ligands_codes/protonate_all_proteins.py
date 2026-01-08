import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_protein_file(filename):
    """
    Check if the file matches either of these patterns:
    - 4-char PDB ID + '_protein.pdb' (e.g., 9B6G_protein.pdb)
    - 4-char PDB ID + _ + additional identifier + '_protein.pdb' (e.g., 9B6G_LQ7_protein.pdb)
    """
    # Pattern: 4 alphanumeric chars, optional underscore + additional identifier, followed by '_protein.pdb'
    pattern = r'^[A-Za-z0-9]{4}(?:_[A-Za-z0-9]+)?_protein\.pdb$'
    return bool(re.match(pattern, filename, re.IGNORECASE))

def protonate_single_file_with_pdb2pqr(args):
    pdb_file, subdir_path = args

    if "only" in os.path.basename(pdb_file).lower() or "protonated" in os.path.basename(pdb_file).lower():
        return ("skip", pdb_file)

    try:
        base_name = os.path.basename(pdb_file).replace('.pdb', '')
        protonated_file = os.path.join(subdir_path, f"{base_name}_protonated.pdb")

        # Check if protonated file already exists before processing
        if os.path.exists(protonated_file) and os.path.getsize(protonated_file) > 0:
            logger.debug(f"Already protonated: {protonated_file}")
            return ("skip", protonated_file)

        cmd = [
            'pdb2pqr30',
            '--ff=AMBER',
            '--with-ph=7.4',
            '--keep-chain',
            '--drop-water',
            pdb_file,
            protonated_file
        ]

        logger.debug(f"Protonating {pdb_file} with PDB2PQR")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if result.returncode == 0 and os.path.exists(protonated_file) and os.path.getsize(protonated_file) > 0:
            return ("success", protonated_file)
        else:
            error_msg = result.stderr if result.returncode != 0 else "Output file not created or empty"
            logger.debug(f"PDB2PQR failed for {pdb_file}: {error_msg}")
            return ("fail", pdb_file)

    except subprocess.TimeoutExpired:
        logger.debug(f"Timeout for {pdb_file}")
        return ("fail", pdb_file)
    except Exception as e:
        logger.debug(f"Error processing {pdb_file}: {e}")
        return ("fail", pdb_file)


def protonate_single_file_with_obabel(args):
    """
    Protonate a single PDB file using obabel as fallback.
    """
    pdb_file, subdir_path = args

    if "only" in os.path.basename(pdb_file).lower() or "protonated" in os.path.basename(pdb_file).lower():
        return ("skip", pdb_file)

    try:
        base_name = os.path.basename(pdb_file).replace('.pdb', '')
        protonated_file = os.path.join(subdir_path, f"{base_name}_protonated.pdb")

        # Check if protonated file already exists before processing
        if os.path.exists(protonated_file) and os.path.getsize(protonated_file) > 0:
            logger.debug(f"Already protonated: {protonated_file}")
            return ("skip", protonated_file)

        cmd = [
            'obabel',
            pdb_file,
            '-O', protonated_file,
            '-h',
            '--pdb'
        ]

        logger.debug(f"Protonating {pdb_file} with OBabel")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and os.path.exists(protonated_file) and os.path.getsize(protonated_file) > 0:
            logger.debug(f"Successfully protonated: {protonated_file}")
            return ("success", protonated_file)
        else:
            error_msg = result.stderr if result.returncode != 0 else "Output file not created or empty"
            logger.debug(f"OBabel failed for {pdb_file}: {error_msg}")
            return ("fail", pdb_file)

    except subprocess.TimeoutExpired:
        logger.debug(f"Timeout for {pdb_file}")
        return ("fail", pdb_file)
    except Exception as e:
        logger.debug(f"Error processing {pdb_file}: {e}")
        return ("fail", pdb_file)


def protonate_file_with_fallback(pdb_file, subdir_path):
    """
    Try protonating with PDB2PQR first, fallback to OBabel if it fails.
    """
    base_name = os.path.basename(pdb_file).replace('.pdb', '')
    protonated_file = os.path.join(subdir_path, f"{base_name}_protonated.pdb")

    # Check if already protonated
    if os.path.exists(protonated_file) and os.path.getsize(protonated_file) > 0:
        logger.debug(f"Already protonated: {protonated_file}")
        return ("skip", protonated_file)

    # First try PDB2PQR
    status, result_file = protonate_single_file_with_pdb2pqr((pdb_file, subdir_path))

    if status == "success":
        return status, result_file
    elif status == "skip":
        return status, result_file
    else:
        # Fallback to OBabel
        logger.debug(f"Falling back to OBabel for {pdb_file}")
        return protonate_single_file_with_obabel((pdb_file, subdir_path))


def process_directory(directory_path):
    """
    Process all PDB files in a directory, applying protonation with fallback.
    Only processes files that match the protein naming pattern: PDBID_protein.pdb or PDBID_ligand_protein.pdb
    """
    pdb_files = []
    for file in os.listdir(directory_path):
        if (file.lower().endswith('.pdb') and 
            is_protein_file(file) and  # Check if it matches the protein pattern
            "protonated" not in file.lower() and 
            "only" not in file.lower()):
            pdb_files.append(os.path.join(directory_path, file))

    results = []
    for pdb_file in pdb_files:
        status, result_file = protonate_file_with_fallback(pdb_file, directory_path)
        results.append((status, result_file))

    return results


def process_single_directory(directory_path):
    """
    Process a single directory and return results.
    """
    if not os.path.isdir(directory_path):
        return directory_path, [("error", f"Directory does not exist: {directory_path}")]

    return directory_path, process_directory(directory_path)


def find_pdb_directories(root_path='.'):
    """
    Recursively find all directories containing .pdb files
    """
    pdb_directories = set()
    
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith('.pdb'):
                pdb_directories.add(root)
                break  # Once we know there are PDBs in this dir, no need to check more files
    
    return list(pdb_directories)


def main():
    # Automatically detect number of CPUs available
    num_cpus = multiprocessing.cpu_count()
    # Use up to 75% of available CPUs to avoid system overload
    max_workers = min(num_cpus, max(1, int(num_cpus * 0.75)))
    
    logger.info(f"Detected {num_cpus} CPUs, using {max_workers} workers for processing")

    # Find all directories containing PDB files
    directories = find_pdb_directories('.')
    
    # Filter directories that actually contain non-protonated protein PDB files
    filtered_directories = []
    for directory in directories:
        contains_unprotonated_protein = False
        for file in os.listdir(directory):
            if (file.lower().endswith('.pdb') and 
                is_protein_file(file) and  # Check if it matches the protein pattern
                "protonated" not in file.lower() and 
                "only" not in file.lower()):
                
                # Check if corresponding protonated file doesn't exist
                base_name = file.replace('.pdb', '')
                protonated_file = os.path.join(directory, f"{base_name}_protonated.pdb")
                if not os.path.exists(protonated_file) or os.path.getsize(protonated_file) == 0:
                    contains_unprotonated_protein = True
                    break
        
        if contains_unprotonated_protein:
            filtered_directories.append(directory)

    log_entries = []

    # Use ThreadPoolExecutor with automatically determined number of workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all directory processing tasks
        future_to_dir = {
            executor.submit(process_single_directory, directory): directory
            for directory in filtered_directories
        }

        # Collect results as they complete
        for future in as_completed(future_to_dir):
            directory, results = future.result()
            log_entries.append(f"Processing directory: {directory}")

            successful_count = sum(1 for status, _ in results if status == "success")
            failed_count = sum(1 for status, _ in results if status == "fail")
            skipped_count = sum(1 for status, _ in results if status == "skip")

            log_entries.append(f"  Directory {directory}: {successful_count} successful, {failed_count} failed, {skipped_count} skipped")

            for status, result_file in results:
                log_entries.append(f"    {status.upper()}: {result_file}")

    # Write the new_continue.log file
    with open('new_continue.log', 'w') as log_file:
        for entry in log_entries:
            log_file.write(entry + '\n')

    print(f"Processing complete for {len(filtered_directories)} directories. "
          f"Using {max_workers} workers out of {num_cpus} available CPUs. "
          f"Processed only files matching PDBID_protein.pdb or PDBID_ligand_protein.pdb pattern. Check new_continue.log for details.")


if __name__ == "__main__":
    main()
