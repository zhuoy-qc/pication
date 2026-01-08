#example usage, python sampling_need_user_input.py 3XF4 benzene.sdf 
#Known issues with the protein cation side chain residue number not matching orginial PDB complex in the ouput csv, but did not affect the docking results
#Then select the ligand you are interested in 3XF4  by input the corresponding number 
import os
import sys
import logging
from Bio.PDB import PDBParser, PDBIO, Select, PDBList
from rdkit import Chem
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("docking_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CleanPDBSelector(Select):
    def accept_residue(self, residue):
        return residue.resname.strip() not in ["HOH", "WAT", "CL", "NA", "K", "MG", "CA"]

class LigandSelector(Select):
    def __init__(self):
        self.valid_ligands = []

    def accept_residue(self, residue):
        if residue.id[0] != " ":
            temp_file = f"temp_{residue.resname}_{residue.id[1]}.pdb"
            io = PDBIO()
            io.set_structure(residue)
            io.save(temp_file)

            mol = Chem.MolFromPDBFile(temp_file)
            os.remove(temp_file)

            if mol and mol.GetNumAtoms() >= 9:
                self.valid_ligands.append({
                    'resname': residue.resname,
                    'chain': residue.parent.id,
                    'resnum': residue.id[1],
                    'atom_count': mol.GetNumAtoms(),
                    'residue': residue
                })
        return False

def download_pdb(pdb_id):
    try:
        pdb_file = f"{pdb_id}.pdb"
        if not os.path.exists(pdb_file):
            logger.info(f"Downloading PDB file for {pdb_id}...")
            pdbl = PDBList()
            pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=".")
            downloaded_file = f"pdb{pdb_id.lower()}.ent"
            if os.path.exists(downloaded_file):
                os.rename(downloaded_file, pdb_file)
                logger.info(f"Downloaded and renamed to {pdb_file}")
            else:
                logger.error(f"Downloaded file {downloaded_file} not found.")
                return None
        else:
            logger.info(f"Using existing PDB file: {pdb_file}")
        return pdb_file
    except Exception as e:
        logger.error(f"Error downloading PDB file: {e}")
        return None

def process_pdb(pdb_file):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("input", pdb_file)
        logger.info(f"Successfully parsed PDB file: {pdb_file}")

        io = PDBIO()
        io.set_structure(structure)
        clean_file = f"{os.path.splitext(pdb_file)[0]}_protein.pdb"
        io.save(clean_file, select=CleanPDBSelector())
        logger.info(f"Saved cleaned protein to {clean_file}")

        ligand_selector = LigandSelector()
        io.set_structure(structure)
        io.save("temp_ligand_check.pdb", select=ligand_selector)
        os.remove("temp_ligand_check.pdb")

        if not ligand_selector.valid_ligands:
            logger.warning("No ligands with ≥9 atoms found")
            return None, None

        logger.info(f"Found {len(ligand_selector.valid_ligands)} valid ligands")
        return clean_file, ligand_selector.valid_ligands
    except Exception as e:
        logger.error(f"Error processing PDB file: {e}")
        return None, None

def convert_to_sdf_and_protonate(ligand_residue, resname_tag, output_dir="."):
    try:
        temp_pdb = os.path.join(output_dir, f"temp_{resname_tag}.pdb")
        io = PDBIO()
        io.set_structure(ligand_residue)
        io.save(temp_pdb)
        logger.debug(f"Saved temporary PDB for ligand: {temp_pdb}")

        mol = Chem.MolFromPDBFile(temp_pdb)
        os.remove(temp_pdb)

        if not mol:
            logger.error(f"Could not parse ligand {resname_tag} from PDB")
            return None, None

        sdf_file = os.path.join(output_dir, f"ligand_{resname_tag.replace('-', '_')}.sdf")
        protonated_sdf = os.path.join(output_dir, f"ligand_{resname_tag.replace('-', '_')}_H.sdf")

        writer = Chem.SDWriter(sdf_file)
        writer.write(mol)
        writer.close()
        logger.info(f"Converted {resname_tag} to SDF format: {sdf_file}")

        # Check if obabel is available
        if os.system("which obabel > /dev/null 2>&1") != 0:
            logger.error("OpenBabel (obabel) is not installed or not in PATH")
            return sdf_file, None

        obabel_cmd = f"obabel {sdf_file} -O {protonated_sdf} -p 2>/dev/null"
        exit_code = os.system(obabel_cmd)
        
        if exit_code == 0 and os.path.exists(protonated_sdf):
            logger.info(f"Protonated ligand saved to: {protonated_sdf}")
            return sdf_file, protonated_sdf
        else:
            logger.error(f"Failed to protonate ligand using obabel (exit code: {exit_code})")
            return sdf_file, None
    except Exception as e:
        logger.error(f"Error converting/protonating ligand: {e}")
        return None, None

def protonate_protein(pdb_file, ligand_info, output_dir="."):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein_with_other_ligands", pdb_file)

        class ProteinWithOtherLigandsSelector(Select):
            def accept_residue(self, residue):
                if residue.id[0] == " ":
                    return True
                if (residue.resname == ligand_info['resname'] and
                    residue.parent.id == ligand_info['chain'] and
                    residue.id[1] == ligand_info['resnum']):
                    return False
                return True

        temp_file = os.path.join(output_dir, "temp_protein_with_other_ligands.pdb")
        io = PDBIO()
        io.set_structure(structure)
        io.save(temp_file, select=ProteinWithOtherLigandsSelector())
        logger.debug(f"Created temporary protein file: {temp_file}")

        # Check if obabel is available
        if os.system("which obabel > /dev/null 2>&1") != 0:
            logger.error("OpenBabel (obabel) is not installed or not in PATH")
            return None

        protonated_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdb_file))[0]}_H.pdb")
        obabel_cmd = f"obabel {temp_file} -O {protonated_file} -p 2>/dev/null"
        exit_code = os.system(obabel_cmd)
        os.remove(temp_file)

        if exit_code == 0 and os.path.exists(protonated_file):
            logger.info(f"Protonated protein saved to {protonated_file}")
            return protonated_file
        else:
            logger.error(f"Failed to protonate protein using obabel (exit code: {exit_code})")
            return None
    except Exception as e:
        logger.error(f"Error protonating protein: {e}")
        return None

def run_smina_docking(protein_file, ligand_file, autobox_ligand, output_dir="."):
    try:
        # Check if smina is available
        if os.system("which smina > /dev/null 2>&1") != 0:
            logger.error("Smina is not installed or not in PATH")
            return

        output_file = os.path.join(output_dir, "docked_ring_poses.sdf")
        cmd = (
            f"smina -r {protein_file} -l {ligand_file} "
            f"--autobox_ligand {autobox_ligand} -o {output_file} "
            "--seed 1 --scoring vina --num_modes 2000  --energy_range 20   --autobox_add 1 "
        )
        logger.info(f"Running docking command:\n{cmd}")
        
        exit_code = os.system(cmd)
        
        if exit_code == 0 and os.path.exists(output_file):
            logger.info(f"Docking results saved to {output_file}")
        else:
            logger.error(f"Docking failed with exit code: {exit_code}")
    except Exception as e:
        logger.error(f"Error running docking: {e}")

# -------------------- Main Execution --------------------
def main():
    try:
        logger.info("Script started")
        
        if len(sys.argv) < 3:
            logger.error("Usage: python prepare_docking.py <PDB_ID> <ligand_sdf_path>")
            print("Usage: python prepare_docking.py <PDB_ID> <ligand_sdf_path>")
            sys.exit(1)

        pdb_id = sys.argv[1].upper()
        user_ligand_sdf = sys.argv[2]
        logger.info(f"PDB ID: {pdb_id}, Ligand SDF: {user_ligand_sdf}")

        if not os.path.isfile(user_ligand_sdf):
            logger.error(f"Ligand SDF file '{user_ligand_sdf}' not found.")
            print(f"Error: Ligand SDF file '{user_ligand_sdf}' not found.")
            sys.exit(1)

        pdb_file = download_pdb(pdb_id)
        if not pdb_file:
            logger.error("Failed to download PDB file")
            sys.exit(1)
            
        output_dir = f"{pdb_id}_prepared"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        protein_file, ligands = process_pdb(pdb_file)
        if not ligands:
            logger.error("No valid ligands found. Exiting.")
            print("No valid ligands found. Exiting.")
            sys.exit(1)

        new_protein_file = os.path.join(output_dir, os.path.basename(protein_file))
        os.rename(protein_file, new_protein_file)
        protein_file = new_protein_file

        print("\nAvailable ligands:")
        for idx, ligand in enumerate(ligands):
            print(f"{idx+1}. {ligand['resname']}-{ligand['chain']}-{ligand['resnum']} ({ligand['atom_count']} atoms)")

        while True:
            try:
                choice = int(input(f"\nEnter the number of the ligand to process (1-{len(ligands)}): "))
                if 1 <= choice <= len(ligands):
                    selected_ligand = ligands[choice - 1]
                    break
                else:
                    print("Please enter a valid number")
            except ValueError:
                print("Please enter a number")

        resname_tag = f"{selected_ligand['resname']}-{selected_ligand['chain']}-{selected_ligand['resnum']}"
        logger.info(f"Selected ligand: {resname_tag}")

        ligand_sdf, ligand_sdf_H = convert_to_sdf_and_protonate(selected_ligand['residue'], resname_tag, output_dir)
        protonated_protein = protonate_protein(protein_file, selected_ligand, output_dir)

        print("\nProcessing complete!")
        print(f"\nFiles generated in '{output_dir}':")
        print(f"- Cleaned protein: {os.path.basename(protein_file)}")
        print(f"- Protonated protein: {os.path.basename(protonated_protein) if protonated_protein else '[FAILED]'}")
        print(f"- Ligand in SDF format: {os.path.basename(ligand_sdf) if ligand_sdf else '[FAILED]'}")
        print(f"- Protonated ligand: {os.path.basename(ligand_sdf_H) if ligand_sdf_H else '[FAILED]'}")

        # Run smina docking if all required files exist
        if protonated_protein and ligand_sdf_H:
            run_smina_docking(
                protein_file=protonated_protein,
                ligand_file=user_ligand_sdf,
                autobox_ligand=ligand_sdf_H,
                output_dir=output_dir
            )
        else:
            logger.warning("Cannot run docking - missing required files")
            print("\nCannot run docking - missing required files")
            
        logger.info("Script finished successfully")
        
    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    main()
