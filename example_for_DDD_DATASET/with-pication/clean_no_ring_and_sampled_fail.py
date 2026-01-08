import pandas as pd
import re

def extract_failed_pdb_ids_from_file(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    failed_pdb_ids = []
    for line in log_content.splitlines():
        if 'No ring matches found' in line:
            match = re.search(r'/([^/]+)$', line)
            if match:
                pdb_id = match.group(1)
                if len(pdb_id) >= 4 and pdb_id[0].isdigit():
                    failed_pdb_ids.append(pdb_id)

    return set(failed_pdb_ids)  # Return as set for efficient lookup

# Extract PDB IDs from docking_prepare.log
no_ring_pdbs = extract_failed_pdb_ids_from_file('docking_prepare.log')
print(f"Number of PDB IDs with no ring matches: {len(no_ring_pdbs)}")

# Extract PDB IDs with timeouts from smina_timeouts.log
timeout_pdbs = set()
with open('smina_timeouts.log', 'r') as f:
    for line in f:
        if 'TIMEOUT' in line:
            # Extract PDB ID from the protein path - looking for the directory name
            # Pattern: /.../PDBID/PDBID_protein_protonated.pdb
            match = re.search(r'/([0-9a-z]{4})/\1_protein_protonated\.pdb', line)
            if match:
                pdb_id = match.group(1).lower()
                timeout_pdbs.add(pdb_id)

print(f"Number of PDB IDs with timeouts: {len(timeout_pdbs)}")

# Combine both sets of PDB IDs to remove
all_pdbs_to_remove = no_ring_pdbs.union(timeout_pdbs)
print(f"Total number of PDB IDs to remove: {len(all_pdbs_to_remove)}")

# Read the CSV file
df = pd.read_csv('reference_experimental_pication_interactions_report.csv')
initial_count = len(df)
print(f"Initial number of rows in CSV: {initial_count}")

# Check if 'Directory' column exists
if 'Directory' not in df.columns:
    print("Error: 'Directory' column not found in the CSV file")
    print(f"Available columns: {list(df.columns)}")
else:
    # Filter out rows where the PDB ID in 'Directory' column had either no ring matches or timeout
    filtered_df = df[~df['Directory'].str.lower().isin(all_pdbs_to_remove)]

    final_count = len(filtered_df)
    print(f"Number of rows after filtering: {final_count}")
    print(f"Number of rows removed: {initial_count - final_count}")

    # Save the filtered dataframe
    filtered_df.to_csv('new_reference_experimental_pication_interactions_report.csv', index=False)
