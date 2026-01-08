import pandas as pd
import re

def extract_core_pdb_id(s):
    s = str(s)
    # Pattern: 4 chars (alphanumeric) + '_' + rest (e.g., 8KCO_N60, 1A2B_LIG)
    match = re.search(r'([A-Z0-9]{4}_[A-Z0-9]+)', s.upper())
    if match:
        return match.group(1)
    # Fallback: first token
    return s.split('_')[0].split('/')[0].split('\\')[0].strip()

def extract_residue_id(protein_str):
    s = str(protein_str).strip()
    match = re.search(r'([A-Z]{3})\D*(\d+)', s.upper())
    if match:
        res = match.group(1)
        num = match.group(2)
        if res in {'ARG', 'LYS', 'HIS'}:
            return f"{res}{num}"
    return None

# Load data
ref_df = pd.read_csv('reference_experimental_pication_interactions_report_with_pka_filtered.csv')
pred_df = pd.read_csv('model_interactions.csv')

# Build reference: group by CORE PDB ID
ref_by_pdb = {}
for _, row in ref_df.iterrows():
    core_id = extract_core_pdb_id(row['Directory'])
    resid = extract_residue_id(row['Protein'])
    if resid is None:
        continue
    if core_id not in ref_by_pdb:
        ref_by_pdb[core_id] = {'all': set(), 'ARG': set(), 'LYS': set(), 'HIS': set()}
    ref_by_pdb[core_id]['all'].add(resid)
    if resid.startswith('ARG'):
        ref_by_pdb[core_id]['ARG'].add(resid)
    elif resid.startswith('LYS'):
        ref_by_pdb[core_id]['LYS'].add(resid)
    elif resid.startswith('HIS'):
        ref_by_pdb[core_id]['HIS'].add(resid)

# Build predictions: group by CORE PDB ID
model_by_pdb_rank = {}
vina_by_pdb_rank = {}

for _, row in pred_df.iterrows():
    core_id = extract_core_pdb_id(row['PDB_ID'])
    resid = extract_residue_id(row['Protein'])
    if resid is None:
        continue
    mr = int(row['Model_Rank'])
    vr = int(row['Vina_Rank'])

    # Model
    if mr <= 8:
        if core_id not in model_by_pdb_rank:
            model_by_pdb_rank[core_id] = {}
        if mr not in model_by_pdb_rank[core_id]:
            model_by_pdb_rank[core_id][mr] = set()
        model_by_pdb_rank[core_id][mr].add(resid)

    # Vina
    if vr <= 8:
        if core_id not in vina_by_pdb_rank:
            vina_by_pdb_rank[core_id] = {}
        if vr not in vina_by_pdb_rank[core_id]:
            vina_by_pdb_rank[core_id][vr] = set()
        vina_by_pdb_rank[core_id][vr].add(resid)

# Only evaluate PDBs present in reference
common_pdbs = set(ref_by_pdb.keys()) & (set(model_by_pdb_rank.keys()) | set(vina_by_pdb_rank.keys()))

# Count totals
total_all = sum(len(d['all']) for d in ref_by_pdb.values())
total_ARG = sum(len(d['ARG']) for d in ref_by_pdb.values())
total_LYS = sum(len(d['LYS']) for d in ref_by_pdb.values())
total_HIS = sum(len(d['HIS']) for d in ref_by_pdb.values())

# Initialize accumulators
depths = [(2,1), (4,2), (6,3), (8,4)]  # (k, m)
vina_all = [0]*4; comb_all = [0]*4
vina_ARG = [0]*4; comb_ARG = [0]*4
vina_LYS = [0]*4; comb_LYS = [0]*4
vina_HIS = [0]*4; comb_HIS = [0]*4

def get_set(d, pdb, max_r):
    s = set()
    if pdb in d:
        for r in range(1, max_r+1):
            if r in d[pdb]:
                s.update(d[pdb][r])
    return s

for pdb in common_pdbs:
    ref = ref_by_pdb[pdb]
    for i, (k, m) in enumerate(depths):
        # Vina top-k
        v_set = get_set(vina_by_pdb_rank, pdb, k)
        vina_all[i] += len(ref['all'] & v_set)
        vina_ARG[i] += len(ref['ARG'] & v_set)
        vina_LYS[i] += len(ref['LYS'] & v_set)
        vina_HIS[i] += len(ref['HIS'] & v_set)

        # Combined: model top-m + vina top-m
        c_set = get_set(model_by_pdb_rank, pdb, m)
        c_set.update(get_set(vina_by_pdb_rank, pdb, m))
        comb_all[i] += len(ref['all'] & c_set)
        comb_ARG[i] += len(ref['ARG'] & c_set)
        comb_LYS[i] += len(ref['LYS'] & c_set)
        comb_HIS[i] += len(ref['HIS'] & c_set)

# Output
def r(num, den): return f"{num}/{den} ({100*num/den:.1f}%)" if den > 0 else "–"
print(f"Evaluated {len(common_pdbs)} complexes (out of {len(ref_by_pdb)} reference)")
print(f"Total interactions: {total_all} (ARG: {total_ARG}, LYS: {total_LYS}, HIS: {total_HIS})\n")

print(f"{'Rank':<8} {'Metric':<8} {'Vina':<18} {'Combined':<18}")
print("-"*52)
for i, lab in enumerate(["Top-2","Top-4","Top-6","Top-8"]):
    print(f"{lab:<8} {'All':<8} {r(vina_all[i], total_all):<18} {r(comb_all[i], total_all):<18}")
    print(f"{'':<8} {'ARG':<8} {r(vina_ARG[i], total_ARG):<18} {r(comb_ARG[i], total_ARG):<18}")
    print(f"{'':<8} {'LYS':<8} {r(vina_LYS[i], total_LYS):<18} {r(comb_LYS[i], total_LYS):<18}")
    print(f"{'':<8} {'HIS':<8} {r(vina_HIS[i], total_HIS):<18} {r(comb_HIS[i], total_HIS):<18}")
    if i < 3: print()
