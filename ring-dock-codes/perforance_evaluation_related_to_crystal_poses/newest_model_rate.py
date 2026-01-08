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

    # Handle HIS and LYS normally (top 50% for HIS, top 5% for LYS)
    for res_type, pct in [('HIS', 50), ('LYS', 5)]:
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

    df_report['ResidueType'] = df_report['Protein'].apply(extract_residue_type)
    df_pred['ResidueType'] = df_pred['Protein'].apply(extract_residue_type)

    # Calculate before filter metrics
    unique_report_all = set(df_report['Protein'].unique())
    unique_pred_all = set(df_pred['Protein'].unique())
    matched_before_all = unique_report_all & unique_pred_all
    overall_recovery_rate_before = (len(matched_before_all) / len(unique_report_all) * 100) if len(unique_report_all) > 0 else 0

    # ARG-specific before filter metrics
    arg_report = set(df_report[df_report['ResidueType'] == 'ARG']['Protein'].unique())
    arg_pred = set(df_pred[df_pred['ResidueType'] == 'ARG']['Protein'].unique())
    arg_matched_before = arg_report & arg_pred
    arg_recovery_rate_before = (len(arg_matched_before) / len(arg_report) * 100) if len(arg_report) > 0 else 0
    arg_recovery_count_before = len(arg_matched_before)
    arg_total_count = len(arg_report)
    
    # LYS-specific before filter metrics
    lys_report = set(df_report[df_report['ResidueType'] == 'LYS']['Protein'].unique())
    lys_pred = set(df_pred[df_pred['ResidueType'] == 'LYS']['Protein'].unique())
    lys_matched_before = lys_report & lys_pred
    lys_recovery_rate_before = (len(lys_matched_before) / len(lys_report) * 100) if len(lys_report) > 0 else 0
    lys_recovery_count_before = len(lys_matched_before)
    lys_total_count = len(lys_report)
    
    # HIS-specific before filter metrics
    his_report = set(df_report[df_report['ResidueType'] == 'HIS']['Protein'].unique())
    his_pred = set(df_pred[df_pred['ResidueType'] == 'HIS']['Protein'].unique())
    his_matched_before = his_report & his_pred
    his_recovery_rate_before = (len(his_matched_before) / len(his_report) * 100) if len(his_report) > 0 else 0
    his_recovery_count_before = len(his_matched_before)
    his_total_count = len(his_report)
    
    overall_recovery_count_before = len(matched_before_all)
    overall_total_count = len(unique_report_all)

    # Apply ranking filter
    df_pred_filtered = apply_ranking(df_pred)

    # Calculate after filter metrics
    unique_pred_filtered_all = set(df_pred_filtered['Protein'].unique())
    matched_after_all = unique_report_all & unique_pred_filtered_all
    overall_recovery_rate_after = (len(matched_after_all) / len(unique_report_all) * 100) if len(unique_report_all) > 0 else 0

    # ARG-specific after filter metrics
    arg_filtered = set(df_pred_filtered[df_pred_filtered['ResidueType'] == 'ARG']['Protein'].unique())
    arg_matched_after = arg_report & arg_filtered
    arg_recovery_rate_after = (len(arg_matched_after) / len(arg_report) * 100) if len(arg_report) > 0 else 0
    arg_recovery_count_after = len(arg_matched_after)
    
    # LYS-specific after filter metrics
    lys_filtered = set(df_pred_filtered[df_pred_filtered['ResidueType'] == 'LYS']['Protein'].unique())
    lys_matched_after = lys_report & lys_filtered
    lys_recovery_rate_after = (len(lys_matched_after) / len(lys_report) * 100) if len(lys_report) > 0 else 0
    lys_recovery_count_after = len(lys_matched_after)
    
    # HIS-specific after filter metrics
    his_filtered = set(df_pred_filtered[df_pred_filtered['ResidueType'] == 'HIS']['Protein'].unique())
    his_matched_after = his_report & his_filtered
    his_recovery_rate_after = (len(his_matched_after) / len(his_report) * 100) if len(his_report) > 0 else 0
    his_recovery_count_after = len(his_matched_after)
    
    overall_recovery_count_after = len(matched_after_all)

    # Non-experimental predictions filtered out (ARG only)
    predictions_not_in_exp_before = unique_pred_all - unique_report_all
    predictions_not_in_exp_after = unique_pred_filtered_all - unique_report_all
    arg_not_in_exp_before = len(set(df_pred[df_pred['ResidueType'] == 'ARG']['Protein'].unique()) & predictions_not_in_exp_before)
    arg_not_in_exp_after = len(set(df_pred_filtered[df_pred_filtered['ResidueType'] == 'ARG']['Protein'].unique()) & predictions_not_in_exp_after)

    if arg_not_in_exp_before > 0:
        arg_filtered_out_rate = ((arg_not_in_exp_before - arg_not_in_exp_after) / arg_not_in_exp_before) * 100
        arg_fp_removed = arg_not_in_exp_before - arg_not_in_exp_after
        arg_fp_total = arg_not_in_exp_before
    else:
        arg_filtered_out_rate = 0.0
        arg_fp_removed = 0
        arg_fp_total = 0

    # Non-experimental predictions filtered out (LYS only)
    lys_not_in_exp_before = len(set(df_pred[df_pred['ResidueType'] == 'LYS']['Protein'].unique()) & predictions_not_in_exp_before)
    lys_not_in_exp_after = len(set(df_pred_filtered[df_pred_filtered['ResidueType'] == 'LYS']['Protein'].unique()) & predictions_not_in_exp_after)

    if lys_not_in_exp_before > 0:
        lys_filtered_out_rate = ((lys_not_in_exp_before - lys_not_in_exp_after) / lys_not_in_exp_before) * 100
        lys_fp_removed = lys_not_in_exp_before - lys_not_in_exp_after
        lys_fp_total = lys_not_in_exp_before
    else:
        lys_filtered_out_rate = 0.0
        lys_fp_removed = 0
        lys_fp_total = 0

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
    report_file = "newest_reference_experimental_pication_interactions_report.csv"
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
    results['df_pred_filtered'].to_csv("remained_predictions.csv", index=False)
    print("Filtered predictions saved to 'remained_predictions.csv'")

if __name__ == "__main__":
    main()
