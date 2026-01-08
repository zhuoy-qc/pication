#!/usr/bin/env python3
import pandas as pd
import re

def extract_core_pdb_id(s):
    """Extract core PDB-ligand ID like '8KCO_N60' from strings like 'complexes_8KCO_N60_exhaust50'."""
    s = str(s)
    match = re.search(r'([A-Z0-9]{4}_[A-Z0-9]+)', s.upper())
    if match:
        return match.group(1)
    # Fallback: first underscore-separated token
    return s.split('_')[0].strip()

def extract_residue_id(protein_str):
    """Extract normalized residue ID: 'ARG-102-A' → 'ARG102'."""
    s = str(protein_str).strip()
    match = re.search(r'([A-Z]{3})\D*(\d+)', s.upper())
    if match:
        res = match.group(1)
        num = match.group(2)
        if res in {'ARG', 'LYS', 'HIS'}:
            return f"{res}{num}"
    return None

def apply_ranking(df_pred):
    """Filter predictions using dihedral-based ranking for ARG, top-% for HIS/LYS."""
    df = df_pred.copy()
    df['_ResType'] = df['Protein'].apply(lambda x: extract_residue_id(x)[:3] if extract_residue_id(x) else None)

    filtered_rows = []

    # ARG: dihedral-angle binning (2° bins, 45 bins)
    arg_subset = df[df['_ResType'] == 'ARG'].copy()
    if not arg_subset.empty and 'Dihedral_Angle' in df.columns:
        bin_size = 2
        top_percent_per_bin = [
            1, 16, 3, 21, 17, 9, 22, 11, 4, 3, 5, 14, 2, 8, 2, 4, 2, 6, 3, 1,
            9, 1, 1, 4, 5, 11, 10, 8, 1, 9, 1, 21, 2, 12, 1, 12, 5, 5, 8, 1,
            3, 10, 2, 1, 6
        ]
        for bin_idx in range(len(top_percent_per_bin)):
            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size if bin_idx < len(top_percent_per_bin) - 1 else 90
            in_bin = arg_subset[
                (arg_subset['Dihedral_Angle'] >= bin_start) &
                (arg_subset['Dihedral_Angle'] <= bin_end)
            ].copy()
            if not in_bin.empty:
                pct = top_percent_per_bin[bin_idx]
                n_keep = max(1, min(int(len(in_bin) * pct / 100), len(in_bin)))
                filtered_rows.append(in_bin.nsmallest(n_keep, 'Energy_Rank'))
    else:
        if not arg_subset.empty:
            filtered_rows.append(arg_subset)

    # HIS: top 50%, LYS: top 5%
    for res, pct in [('HIS', 50), ('LYS', 5)]:
        subset = df[df['_ResType'] == res].copy()
        if not subset.empty:
            n_keep = max(1, int(len(subset) * pct / 100))
            filtered_rows.append(subset.nsmallest(n_keep, 'Energy_Rank'))

    return pd.concat(filtered_rows, ignore_index=True) if filtered_rows else df.iloc[0:0]

def calculate_metrics(report_file, predictions_file):
    # Load data
    df_report = pd.read_csv(report_file)
    df_pred = pd.read_csv(predictions_file)

    # Normalize identifiers
    df_report['_CorePDB'] = df_report['Directory'].apply(extract_core_pdb_id)
    df_pred['_CorePDB'] = df_pred['PDB_ID'].apply(extract_core_pdb_id)
    df_report['_ResID'] = df_report['Protein'].apply(extract_residue_id)
    df_pred['_ResID'] = df_pred['Protein'].apply(extract_residue_id)
    df_report['_ResType'] = df_report['_ResID'].str[:3]
    df_pred['_ResType'] = df_pred['_ResID'].str[:3]

    # Keep only valid π-cation residues and non-null IDs
    valid_report = df_report.dropna(subset=['_CorePDB', '_ResID', '_ResType'])
    valid_pred = df_pred.dropna(subset=['_CorePDB', '_ResID', '_ResType'])

    # Build reference interaction set: {(CorePDB, ResID)}
    reference_set = set(zip(valid_report['_CorePDB'], valid_report['_ResID']))
    ref_by_type = {
        'ARG': set(zip(valid_report[valid_report['_ResType'] == 'ARG']['_CorePDB'],
                       valid_report[valid_report['_ResType'] == 'ARG']['_ResID'])),
        'LYS': set(zip(valid_report[valid_report['_ResType'] == 'LYS']['_CorePDB'],
                       valid_report[valid_report['_ResType'] == 'LYS']['_ResID'])),
        'HIS': set(zip(valid_report[valid_report['_ResType'] == 'HIS']['_CorePDB'],
                       valid_report[valid_report['_ResType'] == 'HIS']['_ResID'])),
        'ALL': reference_set
    }

    # Build prediction set (before filtering)
    pred_set_before = set(zip(valid_pred['_CorePDB'], valid_pred['_ResID']))
    pred_by_type_before = {
        'ARG': set(zip(valid_pred[valid_pred['_ResType'] == 'ARG']['_CorePDB'],
                       valid_pred[valid_pred['_ResType'] == 'ARG']['_ResID'])),
        'LYS': set(zip(valid_pred[valid_pred['_ResType'] == 'LYS']['_CorePDB'],
                       valid_pred[valid_pred['_ResType'] == 'LYS']['_ResID'])),
        'HIS': set(zip(valid_pred[valid_pred['_ResType'] == 'HIS']['_CorePDB'],
                       valid_pred[valid_pred['_ResType'] == 'HIS']['_ResID']))
    }

    # Compute BEFORE-filter recovery
    metrics = {}
    for typ in ['ALL', 'ARG', 'LYS', 'HIS']:
        ref_set = ref_by_type[typ]
        pred_set = pred_by_type_before[typ] if typ != 'ALL' else pred_set_before
        matched = ref_set & pred_set
        metrics[f'{typ.lower()}_total'] = len(ref_set)
        metrics[f'{typ.lower()}_recovered_before'] = len(matched)
        metrics[f'{typ.lower()}_rate_before'] = 100 * len(matched) / len(ref_set) if ref_set else 0.0

    # Apply ranking filter
    df_pred_filtered = apply_ranking(valid_pred)
    pred_set_after = set(zip(df_pred_filtered['_CorePDB'], df_pred_filtered['_ResID']))
    pred_by_type_after = {
        'ARG': set(zip(df_pred_filtered[df_pred_filtered['_ResType'] == 'ARG']['_CorePDB'],
                       df_pred_filtered[df_pred_filtered['_ResType'] == 'ARG']['_ResID'])),
        'LYS': set(zip(df_pred_filtered[df_pred_filtered['_ResType'] == 'LYS']['_CorePDB'],
                       df_pred_filtered[df_pred_filtered['_ResType'] == 'LYS']['_ResID'])),
        'HIS': set(zip(df_pred_filtered[df_pred_filtered['_ResType'] == 'HIS']['_CorePDB'],
                       df_pred_filtered[df_pred_filtered['_ResType'] == 'HIS']['_ResID']))
    }

    # Compute AFTER-filter recovery
    for typ in ['ALL', 'ARG', 'LYS', 'HIS']:
        ref_set = ref_by_type[typ]
        pred_set = pred_by_type_after[typ] if typ != 'ALL' else pred_set_after
        matched = ref_set & pred_set
        metrics[f'{typ.lower()}_recovered_after'] = len(matched)
        metrics[f'{typ.lower()}_rate_after'] = 100 * len(matched) / len(ref_set) if ref_set else 0.0

    # Compute false positive reduction (only for ARG/LYS)
    for typ in ['ARG', 'LYS']:
        fp_before = len(pred_by_type_before[typ] - ref_by_type[typ])
        fp_after = len(pred_by_type_after[typ] - ref_by_type[typ])
        metrics[f'{typ.lower()}_fp_before'] = fp_before
        metrics[f'{typ.lower()}_fp_after'] = fp_after
        metrics[f'{typ.lower()}_fp_removed'] = fp_before - fp_after
        metrics[f'{typ.lower()}_fp_reduction_rate'] = 100 * (fp_before - fp_after) / fp_before if fp_before > 0 else 0.0

    metrics['df_pred_filtered'] = df_pred_filtered
    return metrics

def main():
    report_file = "reference_experimental_pication_interactions_report_with_pka_filtered.csv"
    predictions_file = "new_sample_with_energy_predicted.csv"

    results = calculate_metrics(report_file, predictions_file)

    # Print results
    print("=== π-Cation Recovery (Per-Complex, Residue-Instance Level) ===\n")

    print("Reference interaction counts:")
    print(f"  Overall: {results['all_total']}")
    print(f"  ARG:     {results['arg_total']}")
    print(f"  LYS:     {results['lys_total']}")
    print(f"  HIS:     {results['his_total']}\n")

    print("Recovery rates (before ranking filter):")
    print(f"  Overall: {results['all_recovered_before']}/{results['all_total']} ({results['all_rate_before']:.2f}%)")
    print(f"  ARG:     {results['arg_recovered_before']}/{results['arg_total']} ({results['arg_rate_before']:.2f}%)")
    print(f"  LYS:     {results['lys_recovered_before']}/{results['lys_total']} ({results['lys_rate_before']:.2f}%)")
    if results['his_total'] > 0:
        print(f"  HIS:     {results['his_recovered_before']}/{results['his_total']} ({results['his_rate_before']:.2f}%)")
    print()

    print("Recovery rates (after ranking filter):")
    print(f"  Overall: {results['all_recovered_after']}/{results['all_total']} ({results['all_rate_after']:.2f}%)")
    print(f"  ARG:     {results['arg_recovered_after']}/{results['arg_total']} ({results['arg_rate_after']:.2f}%)")
    print(f"  LYS:     {results['lys_recovered_after']}/{results['lys_total']} ({results['lys_rate_after']:.2f}%)")
    if results['his_total'] > 0:
        print(f"  HIS:     {results['his_recovered_after']}/{results['his_total']} ({results['his_rate_after']:.2f}%)")
    print()

    print("False positive reduction (ARG/LYS only):")
    print(f"  ARG: {results['arg_fp_removed']}/{results['arg_fp_before']} removed ({results['arg_fp_reduction_rate']:.2f}%)")
    print(f"  LYS: {results['lys_fp_removed']}/{results['lys_fp_before']} removed ({results['lys_fp_reduction_rate']:.2f}%)")

    # Save filtered results
    results['df_pred_filtered'].to_csv("tight_remained_predictions.csv", index=False)
    print("\nFiltered predictions saved to 'tight_remained_predictions.csv'")

if __name__ == "__main__":
    main()
