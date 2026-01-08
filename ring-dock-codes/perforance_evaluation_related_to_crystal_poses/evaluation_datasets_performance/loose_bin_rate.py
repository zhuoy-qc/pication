import pandas as pd
import re

def extract_residue_id(protein_str):
    if pd.isna(protein_str):
        return None
    s = str(protein_str).strip()
    match = re.search(r'([A-Za-z]{3})\W*(\d+)', s)
    if match:
        res_code = match.group(1).upper()
        number = match.group(2)
        if res_code in {'ARG', 'HIS', 'LYS'}:
            return f"{res_code}{number}"
    return None

def extract_residue_type(protein_str):
    """Extract just the residue type: 'ARG', 'HIS', or 'LYS'."""
    rid = extract_residue_id(protein_str)
    if rid:
        return rid[:3]
    return None

def apply_ranking(df_pred):
    """
    Filter ARG predictions based on dihedral angle bins with percentage-based ranking.
    Uses fixed parameters: bin size 2°, 45 bins with top percentages [60, 55, 50, ..., 10]
    Returns filtered DataFrame.
    """
    df = df_pred.copy()
    df['ResidueType'] = df['Protein'].apply(extract_residue_type)

    filtered_rows = []

    # Handle ARG residues with dihedral-specific ranking
    arg_subset = df[df['ResidueType'] == 'ARG'].copy()
    if not arg_subset.empty and 'Dihedral_Angle' in df.columns:
        bin_size = 2
        dihedral_bins = 45
        top_percent_per_bin = [60, 55, 50, 45, 40, 35, 30, 25, 20, 15] + [10] * 35  # 45 bins total

        for bin_idx in range(dihedral_bins):
            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size if bin_idx < dihedral_bins - 1 else 90

            bin_subset = arg_subset[
                (arg_subset['Dihedral_Angle'] >= bin_start) &
                (arg_subset['Dihedral_Angle'] < bin_end if bin_idx < dihedral_bins - 1 else arg_subset['Dihedral_Angle'] <= bin_end)
            ].copy()

            if not bin_subset.empty:
                pct = top_percent_per_bin[bin_idx]
                n_total = len(bin_subset)
                n_keep = max(1, min(int(n_total * (pct / 100)), len(bin_subset)))
                bin_sorted = bin_subset.sort_values('Energy_Rank')
                filtered_rows.append(bin_sorted.head(n_keep))
    else:
        # If no ARG or no dihedral, keep all original ARG if exists
        if not arg_subset.empty:
            filtered_rows.append(arg_subset)

    # Handle HIS and LYS normally (top 50% for HIS, top 12% for LYS — as in your code)
    for res_type, pct in [('HIS', 50), ('LYS', 12)]:
        subset = df[df['ResidueType'] == res_type].copy()
        if not subset.empty:
            n_total = len(subset)
            n_keep = max(1, int(n_total * (pct / 100)))
            subset_sorted = subset.sort_values('Energy_Rank')
            filtered_rows.append(subset_sorted.head(n_keep))

    if filtered_rows:
        return pd.concat(filtered_rows, ignore_index=True)
    else:
        return df.iloc[0:0]

def calculate_metrics(report_file, predictions_file):
    df_report = pd.read_csv(report_file)
    df_pred = pd.read_csv(predictions_file)

    # === FIXED: Normalize and group by (CorePDB, ResidueID) ===
    def extract_core_pdb_id(s):
        s = str(s)
        match = re.search(r'([A-Z0-9]{4}_[A-Z0-9]+)', s.upper())
        return match.group(1) if match else s.split('_')[0].strip()

    df_report['_CorePDB'] = df_report['Directory'].apply(extract_core_pdb_id)
    df_pred['_CorePDB'] = df_pred['PDB_ID'].apply(extract_core_pdb_id)
    df_report['_ResID'] = df_report['Protein'].apply(extract_residue_id)
    df_pred['_ResID'] = df_pred['Protein'].apply(extract_residue_id)
    df_report['_ResType'] = df_report['_ResID'].str[:3]
    df_pred['_ResType'] = df_pred['_ResID'].str[:3]

    # Keep only valid interactions
    df_report = df_report.dropna(subset=['_CorePDB', '_ResID', '_ResType'])
    df_pred = df_pred.dropna(subset=['_CorePDB', '_ResID', '_ResType'])

    # Build reference and prediction sets as (CorePDB, ResID) tuples
    ref_set_all = set(zip(df_report['_CorePDB'], df_report['_ResID']))
    pred_set_all = set(zip(df_pred['_CorePDB'], df_pred['_ResID']))

    ### FIXED: Replace all global 'Protein' set operations with per-complex (PDB,ResID) sets

    # Overall
    matched_before_all = ref_set_all & pred_set_all
    overall_recovery_rate_before = (len(matched_before_all) / len(ref_set_all) * 100) if len(ref_set_all) > 0 else 0
    overall_recovery_count_before = len(matched_before_all)
    overall_total_count = len(ref_set_all)

    # ARG-specific
    ref_ARG = set(zip(df_report[df_report['_ResType'] == 'ARG']['_CorePDB'],
                      df_report[df_report['_ResType'] == 'ARG']['_ResID']))
    pred_ARG = set(zip(df_pred[df_pred['_ResType'] == 'ARG']['_CorePDB'],
                       df_pred[df_pred['_ResType'] == 'ARG']['_ResID']))
    arg_matched_before = ref_ARG & pred_ARG
    arg_recovery_rate_before = (len(arg_matched_before) / len(ref_ARG) * 100) if len(ref_ARG) > 0 else 0
    arg_recovery_count_before = len(arg_matched_before)
    arg_total_count = len(ref_ARG)

    # LYS-specific
    ref_LYS = set(zip(df_report[df_report['_ResType'] == 'LYS']['_CorePDB'],
                      df_report[df_report['_ResType'] == 'LYS']['_ResID']))
    pred_LYS = set(zip(df_pred[df_pred['_ResType'] == 'LYS']['_CorePDB'],
                       df_pred[df_pred['_ResType'] == 'LYS']['_ResID']))
    lys_matched_before = ref_LYS & pred_LYS
    lys_recovery_rate_before = (len(lys_matched_before) / len(ref_LYS) * 100) if len(ref_LYS) > 0 else 0
    lys_recovery_count_before = len(lys_matched_before)
    lys_total_count = len(ref_LYS)

    # HIS-specific
    ref_HIS = set(zip(df_report[df_report['_ResType'] == 'HIS']['_CorePDB'],
                      df_report[df_report['_ResType'] == 'HIS']['_ResID']))
    pred_HIS = set(zip(df_pred[df_pred['_ResType'] == 'HIS']['_CorePDB'],
                       df_pred[df_pred['_ResType'] == 'HIS']['_ResID']))
    his_matched_before = ref_HIS & pred_HIS
    his_recovery_rate_before = (len(his_matched_before) / len(ref_HIS) * 100) if len(ref_HIS) > 0 else 0
    his_recovery_count_before = len(his_matched_before)
    his_total_count = len(ref_HIS)

    # Apply ranking filter
    df_pred_filtered = apply_ranking(df_pred)

    # Rebuild filtered prediction sets (with same normalization)
    df_pred_filtered['_CorePDB'] = df_pred_filtered['PDB_ID'].apply(extract_core_pdb_id)
    df_pred_filtered['_ResID'] = df_pred_filtered['Protein'].apply(extract_residue_id)
    df_pred_filtered['_ResType'] = df_pred_filtered['_ResID'].str[:3]
    df_pred_filtered = df_pred_filtered.dropna(subset=['_CorePDB', '_ResID', '_ResType'])

    pred_set_filtered = set(zip(df_pred_filtered['_CorePDB'], df_pred_filtered['_ResID']))
    matched_after_all = ref_set_all & pred_set_filtered
    overall_recovery_rate_after = (len(matched_after_all) / len(ref_set_all) * 100) if len(ref_set_all) > 0 else 0
    overall_recovery_count_after = len(matched_after_all)

    # ARG after
    pred_ARG_filt = set(zip(df_pred_filtered[df_pred_filtered['_ResType'] == 'ARG']['_CorePDB'],
                            df_pred_filtered[df_pred_filtered['_ResType'] == 'ARG']['_ResID']))
    arg_matched_after = ref_ARG & pred_ARG_filt
    arg_recovery_rate_after = (len(arg_matched_after) / len(ref_ARG) * 100) if len(ref_ARG) > 0 else 0
    arg_recovery_count_after = len(arg_matched_after)

    # LYS after
    pred_LYS_filt = set(zip(df_pred_filtered[df_pred_filtered['_ResType'] == 'LYS']['_CorePDB'],
                            df_pred_filtered[df_pred_filtered['_ResType'] == 'LYS']['_ResID']))
    lys_matched_after = ref_LYS & pred_LYS_filt
    lys_recovery_rate_after = (len(lys_matched_after) / len(ref_LYS) * 100) if len(ref_LYS) > 0 else 0
    lys_recovery_count_after = len(lys_matched_after)

    # HIS after
    pred_HIS_filt = set(zip(df_pred_filtered[df_pred_filtered['_ResType'] == 'HIS']['_CorePDB'],
                            df_pred_filtered[df_pred_filtered['_ResType'] == 'HIS']['_ResID']))
    his_matched_after = ref_HIS & pred_HIS_filt
    his_recovery_rate_after = (len(his_matched_after) / len(ref_HIS) * 100) if len(ref_HIS) > 0 else 0
    his_recovery_count_after = len(his_matched_after)

    # False positive calculation (still based on residue ID, but now per-complex)
    fp_before_all = pred_set_all - ref_set_all
    fp_after_all = pred_set_filtered - ref_set_all

    arg_fp_before = len(pred_ARG - ref_ARG)
    arg_fp_after = len(pred_ARG_filt - ref_ARG)
    arg_fp_removed = arg_fp_before - arg_fp_after
    arg_fp_total = arg_fp_before
    arg_filtered_out_rate = (arg_fp_removed / arg_fp_before * 100) if arg_fp_before > 0 else 0.0

    lys_fp_before = len(pred_LYS - ref_LYS)
    lys_fp_after = len(pred_LYS_filt - ref_LYS)
    lys_fp_removed = lys_fp_before - lys_fp_after
    lys_fp_total = lys_fp_before
    lys_filtered_out_rate = (lys_fp_removed / lys_fp_before * 100) if lys_fp_before > 0 else 0.0

    return {
        'arg_recovery_rate_before': arg_recovery_rate_before,
        'arg_recovery_rate_after': arg_recovery_rate_after,
        'arg_filtered_out_rate': arg_filtered_out_rate,
        'lys_recovery_rate_before': lys_recovery_rate_before,
        'lys_recovery_rate_after': lys_recovery_rate_after,
        'lys_filtered_out_rate': lys_filtered_out_rate,
        'his_recovery_rate_before': his_recovery_rate_before,
        'his_recovery_rate_after': his_recovery_rate_after,
        'overall_recovery_rate_before': overall_recovery_rate_before,
        'overall_recovery_rate_after': overall_recovery_rate_after,
        'df_pred_filtered': df_pred_filtered,
        'arg_recovery_count_before': arg_recovery_count_before,
        'arg_recovery_count_after': arg_recovery_count_after,
        'arg_total_count': arg_total_count,
        'lys_recovery_count_before': lys_recovery_count_before,
        'lys_recovery_count_after': lys_recovery_count_after,
        'lys_total_count': lys_total_count,
        'his_recovery_count_before': his_recovery_count_before,
        'his_recovery_count_after': his_recovery_count_after,
        'his_total_count': his_total_count,
        'overall_recovery_count_before': overall_recovery_count_before,
        'overall_recovery_count_after': overall_recovery_count_after,
        'overall_total_count': overall_total_count,
        'arg_fp_removed': arg_fp_removed,
        'arg_fp_total': arg_fp_total,
        'lys_fp_removed': lys_fp_removed,
        'lys_fp_total': lys_fp_total
    }

def main():
    report_file = "reference_experimental_pication_interactions_report_with_pka_filtered.csv"
    predictions_file = "new_sample_with_energy_predicted.csv"

    results = calculate_metrics(report_file, predictions_file)

    print(f"ARG Recovery Rate (Before Filter): {results['arg_recovery_count_before']}/{results['arg_total_count']} ({results['arg_recovery_rate_before']:.2f}%)")
    print(f"ARG Recovery Rate (After Filter): {results['arg_recovery_count_after']}/{results['arg_total_count']} ({results['arg_recovery_rate_after']:.2f}%)")
    print(f"ARG False Positive Filtered Out Rate: {results['arg_fp_removed']}/{results['arg_fp_total']} ({results['arg_filtered_out_rate']:.2f}%)")
    print(f"LYS Recovery Rate (Before Filter): {results['lys_recovery_count_before']}/{results['lys_total_count']} ({results['lys_recovery_rate_before']:.2f}%)")
    print(f"LYS Recovery Rate (After Filter): {results['lys_recovery_count_after']}/{results['lys_total_count']} ({results['lys_recovery_rate_after']:.2f}%)")
    print(f"LYS False Positive Filtered Out Rate: {results['lys_fp_removed']}/{results['lys_fp_total']} ({results['lys_filtered_out_rate']:.2f}%)")
    print(f"HIS Recovery Rate (Before Filter): {results['his_recovery_count_before']}/{results['his_total_count']} ({results['his_recovery_rate_before']:.2f}%)")
    print(f"HIS Recovery Rate (After Filter): {results['his_recovery_count_after']}/{results['his_total_count']} ({results['his_recovery_rate_after']:.2f}%)")
    print(f"Overall Recovery Rate (Before Filter): {results['overall_recovery_count_before']}/{results['overall_total_count']} ({results['overall_recovery_rate_before']:.2f}%)")
    print(f"Overall Recovery Rate (After Filter): {results['overall_recovery_count_after']}/{results['overall_total_count']} ({results['overall_recovery_rate_after']:.2f}%)")

    # Save filtered predictions
    results['df_pred_filtered'].to_csv("loose_remained_predictions.csv", index=False)
    print("Filtered predictions saved to 'loose_remained_predictions.csv'")

if __name__ == "__main__":
    main()
