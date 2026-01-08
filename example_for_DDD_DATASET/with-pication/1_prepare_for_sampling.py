#*_*_protein.pdb in this code, change if your PDB file is named differently (line 106)
import os
import glob
import subprocess
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pickle
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
file_handler = logging.FileHandler("docking_prepare.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
def protonate_single_file_with_pdb2pqr(args):
    pdb_file, subdir_path = args

    if "only" in os.path.basename(pdb_file).lower():
        return ("skip", pdb_file)

    try:
        base_name = os.path.basename(pdb_file).replace('.pdb', '')
        protonated_file = os.path.join(subdir_path, f"{base_name}_protonated.pdb")

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

    if "only" in os.path.basename(pdb_file).lower():
        return ("skip", pdb_file)

    try:
        base_name = os.path.basename(pdb_file).replace('.pdb', '')
        protonated_file = os.path.join(subdir_path, f"{base_name}_protonated.pdb")

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


def find_and_protonate_pdb_files(base_dir):
    pdb_files_to_process = []
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        pdb_pattern = os.path.join(subdir_path, "*_*_protein.pdb")   #change to match the format of your protein PDB file format!
        pdb_files = glob.glob(pdb_pattern)
        pdb_files_to_process.extend([(pdb_file, subdir_path) for pdb_file in pdb_files])

    if not pdb_files_to_process:
        logger.debug("No PDB files found to process")
        return [], []

    logger.debug(f"Found {len(pdb_files_to_process)} PDB files to protonate")

    num_cores = max(1, int(cpu_count() * 0.75))
    logger.debug(f"Using {num_cores} cores for parallel protonation (75% of {cpu_count()} total)")

    # First pass: Try with PDB2PQR
    logger.debug("First pass: Protonating with PDB2PQR")
    with Pool(processes=num_cores, maxtasksperchild=5) as pool:
        results = list(tqdm(
            pool.imap(protonate_single_file_with_pdb2pqr, pdb_files_to_process, chunksize=1),
            total=len(pdb_files_to_process),
            desc="Protonating PDB files with PDB2PQR",
            unit="file"
        ))

    successful_files = []
    failed_files = []
    skipped_files = []

    for status, path in results:
        if status == "success":
            successful_files.append(path)
        elif status == "fail":
            failed_files.append(path)
        elif status == "skip":
            skipped_files.append(path)

    logger.debug(f"PDB2PQR results: {len(successful_files)} successful, {len(failed_files)} failed, {len(skipped_files)} skipped")

    # Second pass: Try failed files with OBabel
    if failed_files:
        logger.debug(f"Second pass: Protonating {len(failed_files)} failed files with OBabel")
        obabel_args = [(f, os.path.dirname(f)) for f in failed_files]
        
        with Pool(processes=min(len(obabel_args), num_cores), maxtasksperchild=5) as pool:
            obabel_results = list(tqdm(
                pool.imap(protonate_single_file_with_obabel, obabel_args, chunksize=1),
                total=len(obabel_args),
                desc="Protonating failed files with OBabel",
                unit="file"
            ))

        # Process OBabel results
        obabel_successful = []
        obabel_failed = []
        
        for status, path in obabel_results:
            if status == "success":
                obabel_successful.append(path)
            elif status == "fail":
                obabel_failed.append(path)
            elif status == "skip":
                skipped_files.append(path)

        # Update final results
        successful_files.extend(obabel_successful)
        failed_files = obabel_failed  # Only files that failed with both methods

        # Log OBabel results
        if obabel_successful:
            logger.debug(f"OBabel recovered {len(obabel_successful)} files")
        if obabel_failed:
            logger.debug(f"OBabel also failed for {len(obabel_failed)} files")

    return successful_files, failed_files


def classify_aromatic_rings(sdf_file, reference_ring_dir="ring_sdf_files"):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        suppl = Chem.SDMolSupplier(sdf_file)
        molecule = next(suppl, None)
        if molecule is None:
            raise ValueError(f"Could not read molecule from {sdf_file}")

        molecule = Chem.AddHs(molecule)
        Chem.SanitizeMol(molecule)
        aromatic_rings_info = []
        ring_info = molecule.GetRingInfo()
        atoms = molecule.GetAtoms()

        aromatic_rings = []
        for ring in ring_info.AtomRings():
            if all(atoms[idx].GetIsAromatic() for idx in ring):
                aromatic_rings.append(frozenset(ring))

        for ring_set in aromatic_rings:
            ring_list = list(ring_set)
            elements = [atoms[idx].GetSymbol() for idx in ring_list]
            ring_size = len(ring_list)

            if ring_size == 6:
                n_count = elements.count('N')
                if n_count == 0:
                    aromatic_rings_info.append(('Benzene', ring_list))
                elif n_count == 1:
                    aromatic_rings_info.append(('Pyridine', ring_list))
                elif n_count == 2:
                    positions = []
                    for i, idx in enumerate(ring_list):
                        if atoms[idx].GetSymbol() == 'N':
                            positions.append(i)
                    diff = abs(positions[0] - positions[1])
                    if diff == 1 or diff == 5:
                        aromatic_rings_info.append(('Pyridazine-like', ring_list))
                    elif diff == 2 or diff == 4:
                        aromatic_rings_info.append(('Pyrimidine-like', ring_list))
                    else:
                        aromatic_rings_info.append(('Pyrazine-like', ring_list))
                elif n_count >= 3:
                    aromatic_rings_info.append(('Triazine/Tetrazine', ring_list))
            elif ring_size == 5:
                heteroatoms = [e for e in elements if e != 'C']
                if not heteroatoms:
                    aromatic_rings_info.append(('Cyclopentadiene-like', ring_list))
                elif 'N' in heteroatoms and 'O' not in heteroatoms and 'S' not in heteroatoms:
                    aromatic_rings_info.append(('Pyrrole-like', ring_list))
                elif 'O' in heteroatoms:
                    aromatic_rings_info.append(('Furan-like', ring_list))
                elif 'S' in heteroatoms:
                    aromatic_rings_info.append(('Thiophene-like', ring_list))
                else:
                    aromatic_rings_info.append(('Other 5-membered heteroaromatic', ring_list))
            else:
                aromatic_rings_info.append((f'Other {ring_size}-membered aromatic ring', ring_list))

        fused_systems = []
        for i, ring_a in enumerate(aromatic_rings):
            for ring_b in aromatic_rings[i+1:]:
                if ring_a & ring_b:
                    found = False
                    for sys in fused_systems:
                        if ring_a in sys or ring_b in sys:
                            sys.update([ring_a, ring_b])
                            found = True
                            break
                    if not found:
                        fused_systems.append({ring_a, ring_b})

        bicyclic_fused_info = []
        for sys in fused_systems:
            if len(sys) == 2:
                ring_a, ring_b = list(sys)
                size_a, size_b = len(ring_a), len(ring_b)
                elements_a = [atoms[idx].GetSymbol() for idx in ring_a]
                elements_b = [atoms[idx].GetSymbol() for idx in ring_b]
                n_count = elements_a.count('N') + elements_b.count('N')

                if size_a == 6 and size_b == 6:
                    if n_count == 0:
                        bicyclic_fused_info.append(('Naphthalene-like', list(ring_a), list(ring_b)))
                    elif n_count == 1:
                        bicyclic_fused_info.append(('Quinoline/Isoquinoline-like', list(ring_a), list(ring_b)))
                    elif n_count == 2:
                        bicyclic_fused_info.append(('Cinnoline/Phthalazine-like', list(ring_a), list(ring_b)))
                    else:
                        bicyclic_fused_info.append((f'Fused 6+6 ring with {n_count} N atoms', list(ring_a), list(ring_b)))
                elif {size_a, size_b} == {5, 6}:
                    five_ring = ring_a if size_a == 5 else ring_b
                    six_ring = ring_b if size_b == 6 else ring_a
                    five_elements = [atoms[idx].GetSymbol() for idx in five_ring]
                    heteroatoms = [e for e in five_elements if e != 'C']
                    if 'N' in heteroatoms:
                        bicyclic_fused_info.append(('Indole-like', list(five_ring), list(six_ring)))
                    elif 'O' in heteroatoms:
                        bicyclic_fused_info.append(('Benzofuran-like', list(five_ring), list(six_ring)))
                    elif 'S' in heteroatoms:
                        bicyclic_fused_info.append(('Benzothiophene-like', list(five_ring), list(six_ring)))
                    else:
                        bicyclic_fused_info.append(('Other 5+6 fused system', list(five_ring), list(six_ring)))
                else:
                    bicyclic_fused_info.append((f'Other fused system ({size_a}+{size_b})', list(ring_a), list(ring_b)))

        ring_matches = []
        if reference_ring_dir and os.path.exists(reference_ring_dir):
            reference_mapping = {
                'benzene.sdf': 'Benzene',
                'pyridine.sdf': 'Pyridine',
                'pyrimidine.sdf': 'Pyrimidine',
                'pyrrole.sdf': 'Pyrrole',
                'thiophene.sdf': 'Thiophene'
            }
            for ring_type, atom_indices in aromatic_rings_info:
                matched_ref = None
                matched_path = None
                for ref_file, ref_name in reference_mapping.items():
                    if ring_type.lower().startswith(ref_name.lower()):
                        ref_path = os.path.join(reference_ring_dir, ref_file)
                        if os.path.exists(ref_path):
                            matched_ref = ref_name
                            matched_path = ref_path
                            break
                if matched_ref and matched_path:
                    ring_matches.append((ring_type, atom_indices, matched_ref, matched_path))
        return aromatic_rings_info, bicyclic_fused_info, ring_matches

    except Exception as e:
        logger.debug(f"Error in ring classification for {sdf_file}: {e}")
        return [], [], []


def find_ligand_sdf_files(directory):
    ligand_pattern = os.path.join(directory, "*_ligand.sdf")
    return glob.glob(ligand_pattern)


def setup_environment():
    # Check PDB2PQR
    try:
        result = subprocess.run(['pdb2pqr30', '--version'], capture_output=True, text=True, check=True)
        logger.debug(f"PDB2PQR available: {result.stdout.strip() or 'OK'}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("PDB2PQR is not installed or not in PATH. Install with: conda install -c conda-forge pdb2pqr")
        return False

    # Check OBabel
    try:
        result = subprocess.run(['obabel', '-V'], capture_output=True, text=True, check=True)
        logger.debug(f"OBabel available: {result.stderr.strip() or 'OK'}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("OBabel is not installed or not in PATH. Install with: conda install -c conda-forge openbabel")
        return False

    # Check RDKit
    try:
        from rdkit import Chem
        logger.debug("RDKit available for ring classification")
    except ImportError:
        logger.debug("RDKit not available. Ring classification will fail.")

    return True


def collect_all_docking_tasks(protonated_files, reference_ring_dir):
    all_docking_tasks = []
    logger.debug("Collecting all docking tasks from all proteins...")

    for protonated_file in tqdm(protonated_files, desc="Scanning proteins"):
        protein_dir = os.path.dirname(protonated_file)
        ligand_files = find_ligand_sdf_files(protein_dir)

        if not ligand_files:
            logger.debug(f"No ligand files found in {protein_dir}")
            continue

        autobox_ligand = ligand_files[0]
        all_ring_matches = []
        for ligand_file in ligand_files:
            try:
                aromatic_rings, bicyclic_fused, ring_matches = classify_aromatic_rings(ligand_file, reference_ring_dir)
                if ring_matches:
                    all_ring_matches.extend(ring_matches)
            except Exception as e:
                logger.debug(f"Error in ring matching for {ligand_file}: {e}")
                continue

        if not all_ring_matches:
            logger.debug(f"No ring matches found in {protein_dir}")
            continue

        for ring_match in all_ring_matches:
            ring_type, atom_indices, ref_name, ref_path = ring_match
            all_docking_tasks.append((
                protonated_file,
                ref_path,
                autobox_ligand,
                protein_dir,
                30
            ))

    return all_docking_tasks


def main_preparation():
    # Only log essential startup to terminal? Actually, skip even this.
    # We'll only print what's necessary at the end.

    base_dir = os.getcwd()
    reference_ring_dir = None

    current_path = os.path.abspath(base_dir)
    while current_path != os.path.dirname(current_path):
        candidate = os.path.join(current_path,  'ring_sdf_files')
        if os.path.exists(candidate):
            reference_ring_dir = candidate
            break
        current_path = os.path.dirname(current_path)

    if not setup_environment():
        print("❌ Environment setup failed. Please install required tools.")
        return

    # === STEP 1: Protonation ===
    protonated_files, failed_files = find_and_protonate_pdb_files(base_dir)

    # Print clean summary to terminal
    if failed_files:
        failed_ids = sorted({os.path.basename(os.path.dirname(f)) for f in failed_files})
        print(f"⚠️  Protonation failed for {len(failed_ids)} complexes: {', '.join(failed_ids)}")
        print(f"   These files failed with both PDB2PQR and OBabel")
    else:
        print("✅ All PDB files protonated successfully.")

    if not protonated_files:
        print("❌ No protonated PDB files found. Exiting.")
        return

    # === STEP 2: Task Collection ===
    all_docking_tasks = collect_all_docking_tasks(protonated_files, reference_ring_dir)

    if not all_docking_tasks:
        print("❌ No docking tasks found. Exiting.")
        return

    tasks_file = "docking_tasks.pkl"
    with open(tasks_file, 'wb') as f:
        pickle.dump(all_docking_tasks, f)

    print(f"✅ Preparation completed! {len(all_docking_tasks)} docking tasks saved to {tasks_file}")


if __name__ == "__main__":
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main_preparation()

  
