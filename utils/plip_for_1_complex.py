#!/usr/bin/env python3
import os
import sys
from Bio.PDB import PDBParser, PDBIO, Select, PDBList
from rdkit import Chem
import numpy as np
from plip.structure.preparation import PDBComplex

# -------------------- Selectors --------------------

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

# -------------------- Core Functions --------------------

def download_pdb(pdb_id):
    pdb_file = f"{pdb_id}.pdb"
    if not os.path.exists(pdb_file):
        print(f"Downloading PDB file for {pdb_id}...")
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=".")
        downloaded_file = f"pdb{pdb_id.lower()}.ent"
        if os.path.exists(downloaded_file):
            os.rename(downloaded_file, pdb_file)
        else:
            print(f"Error: Downloaded file {pdb_file} not found.")
            sys.exit(1)
    return pdb_file

def compute_dihedral_angle(p1, p2, p3, n2):
    """
    Computes the dihedral angle between two planes.

    Parameters:
    - p1, p2, p3: 3D points (as lists or tuples) defining the first plane.
    - n2: Normal vector (as a list or tuple) of the second plane.

    Returns:
    - Dihedral angle in degrees (0 <= angle <= 90).
    """
    # Convert inputs to numpy arrays
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    n2 = np.array(n2, dtype=float)

    # Calculate normal vector for the first plane using cross product
    v1 = p2 - p1
    v2 = p3 - p1
    n1 = np.cross(v1, v2)
    norm_n1 = np.linalg.norm(n1)

    # Check for collinearity of the three points
    if norm_n1 < 1e-8:
        raise ValueError("The three points defining the first plane are collinear.")

    n1 = n1 / norm_n1  # Normalize the first normal vector

    # Normalize the second normal vector
    norm_n2 = np.linalg.norm(n2)
    if norm_n2 < 1e-8:
        raise ValueError("The normal vector of the second plane is zero.")
    n2 = n2 / norm_n2

    # Calculate the angle between the two normal vectors
    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
    angle_rad = np.arccos(np.abs(cos_angle))  # Acute angle in radians
    return np.degrees(angle_rad)

def calculate_rz(distance, offset):
    """Calculate rz value from distance and offset"""
    return np.sqrt(distance**2 - offset**2)

def analyze_pication_interactions(pdb_file):
    """Analyze π-cation (laro) interactions in a PDB complex file with precise geometry"""
    my_mol = PDBComplex()
    my_mol.load_pdb(pdb_file)
    my_mol.analyze()
    
    results = []
    
    for _, interactions in my_mol.interaction_sets.items():
        for pication in interactions.all_pication_laro:
            # Get the ring normal vector from PLIP
            ring_normal = np.array(pication.ring.normal, dtype=np.float64)
            
            # Get the coordinates of the charged atoms
            charge_atom_coords = [atom.coords for atom in pication.charge.atoms]
            
            # We need at least 3 non-collinear points to define a plane
            if len(charge_atom_coords) < 3:
                continue  # Skip if not enough atoms
            
            # Take the first 3 charged atom coordinates to define the plane
            p1 = charge_atom_coords[0]
            p2 = charge_atom_coords[1]
            p3 = charge_atom_coords[2]
            
            # Calculate the dihedral angle between the plane defined by the 3 charged atoms and the ring normal
            try:
                dihedral_angle = compute_dihedral_angle(p1, p2, p3, ring_normal)
            except ValueError as e:
                print(f"Skipping interaction due to error: {e}")
                continue
            
            # Calculate other geometric parameters
            charge_vector = np.array(pication.charge.center, dtype=np.float64) - np.array(pication.ring.center, dtype=np.float64)
            angle = np.arccos(np.clip(np.dot(ring_normal, charge_vector) / (np.linalg.norm(ring_normal) * np.linalg.norm(charge_vector)), -1.0, 1.0))
            angle_deg = np.degrees(angle)
            rz = calculate_rz(pication.distance, pication.offset)
            
            results.append({
                'PDB_File': pdb_file,
                'Ligand': f"{pication.restype_l}-{pication.resnr_l}-{pication.reschain_l}",
                'Protein': f"{pication.restype}-{pication.resnr}-{pication.reschain}",
                'Distance': format(float(pication.distance), '.16g'),
                'Offset': format(float(pication.offset), '.16g'),
                'RZ': format(rz, '.16g'),
                'Angle': format(angle_deg, '.16g'),  # Original angle
                'Dihedral_Angle': format(dihedral_angle, '.16g'),  # New dihedral angle between ring normal and charged plane
                'Ring_Center_X': format(float(pication.ring.center[0]), '.16g'),
                'Ring_Center_Y': format(float(pication.ring.center[1]), '.16g'),
                'Ring_Center_Z': format(float(pication.ring.center[2]), '.16g'),
                'Charged_Center_X': format(float(pication.charge.center[0]), '.16g'),
                'Charged_Center_Y': format(float(pication.charge.center[1]), '.16g'),
                'Charged_Center_Z': format(float(pication.charge.center[2]), '.16g'),
                'Ring_Normal_X': format(float(ring_normal[0]), '.16g'),
                'Ring_Normal_Y': format(float(ring_normal[1]), '.16g'),
                'Ring_Normal_Z': format(float(ring_normal[2]), '.16g'),
                'Ring_Type': pication.ring.type,
                'Atom_Indices': str(pication.ring.atoms_orig_idx)
            })
    
    return results

def get_raw_coordinates_for_pication_interactions(pdb_file):
    """
    Access the raw coordinates used to compute charge centers and ring centers
    """
    my_mol = PDBComplex()
    my_mol.load_pdb(pdb_file)
    my_mol.analyze()
    
    raw_data = []
    
    for ligand_id, interactions in my_mol.interaction_sets.items():
        for pication in interactions.all_pication_laro:
            # Get raw coordinates for the ring atoms
            ring_atom_coords = [atom.coords for atom in pication.ring.atoms]
            
            # Get raw coordinates for the charged atoms
            charge_atom_coords = [atom.coords for atom in pication.charge.atoms]
            
            raw_data.append({
                'Ligand_ID': ligand_id,
                'PDB_File': pdb_file,
                'Ligand': f"{pication.restype_l}-{pication.resnr_l}-{pication.reschain_l}",
                'Protein': f"{pication.restype}-{pication.resnr}-{pication.reschain}",
                'Ring_Atom_Coords': ring_atom_coords,  # Raw coordinates for ring atoms
                'Ring_Center': pication.ring.center,   # Computed centroid
                'Charge_Atom_Coords': charge_atom_coords,  # Raw coordinates for charged atoms
                'Charge_Center': pication.charge.center,    # Computed centroid
                'Distance': pication.distance,
                'Offset': pication.offset
            })
    
    return raw_data

def print_raw_charged_atom_coordinates(pdb_file):
    """
    Print the raw coordinates for charged atoms involved in pi-cation interactions
    """
    print(f"Analyzing pi-cation interactions in {pdb_file}")
    print("="*60)
    
    my_mol = PDBComplex()
    my_mol.load_pdb(pdb_file)
    my_mol.analyze()
    
    interaction_count = 0
    
    for ligand_id, interactions in my_mol.interaction_sets.items():
        pications = interactions.all_pication_laro
        if pications:
            print(f"Ligand: {ligand_id}")
            for pication in pications:
                interaction_count += 1
                print(f"  Interaction #{interaction_count}:")
                print(f"    Protein residue: {pication.restype}-{pication.resnr}-{pication.reschain}")
                print(f"    Ligand residue: {pication.restype_l}-{pication.resnr_l}-{pication.reschain_l}")
                print(f"    Distance: {pication.distance:.3f} Å")
                print(f"    Offset: {pication.offset:.3f} Å")
                
                # Get the ring normal vector from PLIP
                ring_normal = np.array(pication.ring.normal, dtype=np.float64)
                print(f"    Ring normal vector: ({ring_normal[0]:.3f}, {ring_normal[1]:.3f}, {ring_normal[2]:.3f})")
                
                # Get the coordinates of the ring atoms
                ring_atom_coords = [atom.coords for atom in pication.ring.atoms]
                
                # Print raw coordinates for ring atoms
                print(f"    Raw ring atom coordinates:")
                for i, coord in enumerate(ring_atom_coords):
                    print(f"      Atom {i+1}: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})")
                
                # Print the computed ring center
                print(f"    Ring center (centroid): ({pication.ring.center[0]:.3f}, {pication.ring.center[1]:.3f}, {pication.ring.center[2]:.3f})")
                
                # Print raw coordinates for charged atoms
                charge_atom_coords = [atom.coords for atom in pication.charge.atoms]
                print(f"    Raw charged atom coordinates:")
                for i, coord in enumerate(charge_atom_coords):
                    print(f"      Atom {i+1}: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})")
                
                # Print the computed charge center
                print(f"    Charge center (centroid): ({pication.charge.center[0]:.3f}, {pication.charge.center[1]:.3f}, {pication.charge.center[2]:.3f})")
                
                # Calculate dihedral angle if we have at least 3 charged atoms
                if len(charge_atom_coords) >= 3:
                    p1 = charge_atom_coords[0]
                    p2 = charge_atom_coords[1]
                    p3 = charge_atom_coords[2]
                    try:
                        dihedral_angle = compute_dihedral_angle(p1, p2, p3, ring_normal)
                        print(f"    Dihedral angle (between plane of first 3 charged atoms and ring normal): {dihedral_angle:.3f}°")
                    except ValueError as e:
                        print(f"    Could not calculate dihedral angle: {e}")
                
                # Calculate the angle between ring normal and charge vector
                charge_vector = np.array(pication.charge.center, dtype=np.float64) - np.array(pication.ring.center, dtype=np.float64)
                cos_angle = np.dot(ring_normal, charge_vector) / (np.linalg.norm(ring_normal) * np.linalg.norm(charge_vector))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_rad = np.arccos(np.abs(cos_angle))
                angle_deg = np.degrees(angle_rad)
                print(f"    Angle between ring normal and charge vector: {angle_deg:.3f}°")
                
                print()
    
    if interaction_count == 0:
        print("No pi-cation interactions found.")
    else:
        print(f"Total pi-cation interactions found: {interaction_count}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <PDB_ID>")
        sys.exit(1)
    
    pdb_id = sys.argv[1].upper()
    
    # Download the PDB file
    pdb_file = download_pdb(pdb_id)
    
    # Clean up the PDB file by removing water and ions
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    clean_file = f"clean_{pdb_file}"
    io = PDBIO()
    io.set_structure(structure)
    io.save(clean_file, CleanPDBSelector())
    
    # Identify valid ligands
    parser_clean = PDBParser(QUIET=True)
    structure_clean = parser_clean.get_structure("clean_protein", clean_file)
    
    ligand_selector = LigandSelector()
    io_ligand = PDBIO()
    io_ligand.set_structure(structure_clean)
    io_ligand.save(f"ligands_{pdb_file}", ligand_selector)
    
    print(f"Valid ligands found: {len(ligand_selector.valid_ligands)}")
    for ligand in ligand_selector.valid_ligands:
        print(f"  - {ligand['resname']} (Chain {ligand['chain']}, Residue {ligand['resnum']}, {ligand['atom_count']} atoms)")
    
    # Analyze pi-cation interactions
    print_raw_charged_atom_coordinates(clean_file)

if __name__ == "__main__":
    main()
