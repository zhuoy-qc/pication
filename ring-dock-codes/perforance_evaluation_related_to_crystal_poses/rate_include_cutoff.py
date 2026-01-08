import pandas as pd
import re

# ===== CONFIGURABLE PARAMETERS =====
TOP_PERCENT_PER_RESIDUE = {
    'ARG': 10,   # top % of ARG predictions (within dihedral cutoff if applicable)
    'HIS': 50,   # top % of HIS predictions
    'LYS': 5     # top % of LYS predictions
}
DIHEDRAL_CUTOFF_ARG = 30  # degrees
# ===================================

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

def apply_ranking(df_pred, per_residue_pct, dihedral_cutoff_arg=35):
    df = df_pred.copy()
    df['ResidueType'] = df['Protein'].apply(extract_residue_type)

    filtered_rows = []
    for res_type in ['ARG', 'HIS', 'LYS']:
        subset = df[df['ResidueType'] == res_type].copy()
        if subset.empty:
            continue

        # Apply dihedral filter to ARG predictions BEFORE ranking
        if res_type == 'ARG':
            def filter_arg_row(row):
                # Use the correct column name: 'Dihedral_Angle'
                dihedral = row.get('Dihedral_Angle') 
                if pd.notna(dihedral):
                    try:
                        return float(dihedral) <= dihedral_cutoff_arg
                    except (ValueError, TypeError):
                        return False  # exclude if invalid
                else:
                    return False  # exclude if no dihedral info
            subset = subset[subset.apply(filter_arg_row, axis=1)].copy()
            if subset.empty: # Check if subset is empty after dihedral filter
                continue # Skip ranking if no ARGs meet the dihedral criteria

        pct = per_residue_pct.get(res_type, 100)
        n_total = len(subset)
        n_keep = max(1, int(n_total * (pct / 100)))
        subset_sorted = subset.sort_values('Energy_Rank')
        filtered_rows.append(subset_sorted.head(n_keep))
    if filtered_rows:
        return pd.concat(filtered_rows, ignore_index=True)
    else:
        return df.iloc[0:0]

def check_protein_matching_per_residue(report_file, predictions_file, per_residue_pct, dihedral_cutoff_arg=35):
    df_report = pd.read_csv(report_file)
    df_pred = pd.read_csv(predictions_file)

    df_report.columns = df_report.columns.str.strip()
    df_pred.columns = df_pred.columns.str.strip()

    # === Apply dihedral filter ONLY for ARG in experimental report ===
    # Assumes the report file uses 'Dihedral_Angle_°'
    def should_keep_row(row):
        res_type = extract_residue_type(row['Protein'])
        if res_type == 'ARG':
            dihedral = row.get('Dihedral_Angle_°') # Use the original column name for the report
            if pd.notna(dihedral):
                try:
                    return float(dihedral) <= dihedral_cutoff_arg
                except (ValueError, TypeError):
                    return False  # if invalid, exclude
            else:
                return False  # no dihedral info → exclude ARG
        else:
            return True  # keep all HIS/LYS regardless

    # Create a copy for dihedral-filtered report (only affects ARG)
    df_report_filtered = df_report[df_report.apply(should_keep_row, axis=1)].copy()

    # Apply ranking to predictions (with dihedral filter for ARG predictions)
    df_pred_filtered = apply_ranking(df_pred, per_residue_pct, dihedral_cutoff_arg)

    def extract_full_protein_info(protein_str):
        return str(protein_str).strip() if pd.notna(protein_str) else ""

    # Add Protein_Residue and ResidueType to all DataFrames
    for df in [df_report, df_report_filtered, df_pred, df_pred_filtered]:
        df['Protein_Residue'] = df['Protein'].apply(extract_full_protein_info)
        df['ResidueType'] = df['Protein'].apply(extract_residue_type)

    # === Sets for matching ===
    unique_report_all = set(df_report_filtered['Protein_Residue'].unique())
    unique_pred_all = set(df_pred['Protein_Residue'].unique())
    unique_pred_filtered_all = set(df_pred_filtered['Protein_Residue'].unique())

    total_report_all = len(unique_report_all)

    # === Per-residue analysis BEFORE ranking (against filtered report) ===
    per_residue_results_before = {}
    for res_type in ['ARG', 'HIS', 'LYS']:
        report_subset = df_report_filtered[df_report_filtered['ResidueType'] == res_type]
        pred_subset = df_pred[df_pred['ResidueType'] == res_type]

        unique_report_res = set(report_subset['Protein_Residue'].unique()) if not report_subset.empty else set()
        unique_pred_res = set(pred_subset['Protein_Residue'].unique()) if not pred_subset.empty else set()
        matched_res_before = unique_report_res & unique_pred_res
        total_report_res = len(unique_report_res)
        matched_count_res_before = len(matched_res_before)
        recovery_rate_res_before = (matched_count_res_before / total_report_res * 100) if total_report_res > 0 else 0

        per_residue_results_before[res_type] = {
            'total_experimental': total_report_res,
            'recovered_before_ranking': matched_count_res_before,
            'recovery_rate_before_ranking': recovery_rate_res_before
        }

    # === Per-residue analysis AFTER ranking (against filtered report) ===
    per_residue_results = {}
    for res_type in ['ARG', 'HIS', 'LYS']:
        report_subset = df_report_filtered[df_report_filtered['ResidueType'] == res_type]
        pred_filtered_subset = df_pred_filtered[df_pred_filtered['ResidueType'] == res_type]

        unique_report_res = set(report_subset['Protein_Residue'].unique()) if not report_subset.empty else set()
        unique_pred_filtered_res = set(pred_filtered_subset['Protein_Residue'].unique()) if not pred_filtered_subset.empty else set()
        matched_res = unique_report_res & unique_pred_filtered_res

        total_report_res = len(unique_report_res)
        matched_count_res = len(matched_res)
        recovery_rate_res = (matched_count_res / total_report_res * 100) if total_report_res > 0 else 0

        per_residue_results[res_type] = {
            'total_experimental': total_report_res,
            'recovered': matched_count_res,
            'recovery_rate': recovery_rate_res,
            'missed_interactions': list(unique_report_res - matched_res),
            'predictions_considered': len(pred_filtered_subset)
        }

    # === FP analysis (predictions not in experimental) ===
    predictions_not_in_exp_before = unique_pred_all - unique_report_all
    predictions_not_in_exp_after = unique_pred_filtered_all - unique_report_all

    # FP counts per residue type
    predictions_not_in_exp_before_by_residue = {}
    predictions_not_in_exp_after_by_residue = {}
    fp_filtered_rate_per_residue = {}

    for res_type in ['ARG', 'HIS', 'LYS']:
        pred_before = df_pred[df_pred['ResidueType'] == res_type]['Protein_Residue'].unique()
        pred_after = df_pred_filtered[df_pred_filtered['ResidueType'] == res_type]['Protein_Residue'].unique()

        fp_before = len(set(pred_before) & predictions_not_in_exp_before)
        fp_after = len(set(pred_after) & predictions_not_in_exp_after)

        predictions_not_in_exp_before_by_residue[res_type] = fp_before
        predictions_not_in_exp_after_by_residue[res_type] = fp_after

        if fp_before > 0:
            fp_filtered_rate = ((fp_before - fp_after) / fp_before) * 100
        else:
            fp_filtered_rate = 0.0
        fp_filtered_rate_per_residue[res_type] = fp_filtered_rate

    # Overall recovery before and after ranking (against filtered report)
    matched_all_before = unique_report_all & unique_pred_all
    total_recovered_before = len(matched_all_before)
    overall_rate_before = (total_recovered_before / total_report_all * 100) if total_report_all > 0 else 0

    matched_all_after = unique_report_all & unique_pred_filtered_all
    total_recovered_after = len(matched_all_after)
    overall_rate_after = (total_recovered_after / total_report_all * 100) if total_report_all > 0 else 0

    return {
        'overall': {
            'total_pi_cation_interactions': total_report_all,
            'recovered_interactions_before': total_recovered_before,
            'recovered_interactions_after': total_recovered_after,
            'recovery_rate_before': overall_rate_before,
            'recovery_rate_after': overall_rate_after,
            'total_predictions_considered': len(df_pred_filtered),
        },
        'per_residue_results_before_ranking': per_residue_results_before,
        'per_residue_results': per_residue_results,
        'predictions_not_in_exp_before_ranking': {
            'count': len(predictions_not_in_exp_before),
            'by_residue': predictions_not_in_exp_before_by_residue
        },
        'predictions_not_in_exp_after_ranking': {
            'count': len(predictions_not_in_exp_after),
            'by_residue': predictions_not_in_exp_after_by_residue
        },
        'fp_filtered_rate_per_residue': fp_filtered_rate_per_residue,
        'filtered_out_percentage': (len(predictions_not_in_exp_before) - len(predictions_not_in_exp_after)) / len(predictions_not_in_exp_before) * 100 if len(predictions_not_in_exp_before) > 0 else 0,
        'per_residue_percent': per_residue_pct,
    }

def main():
    report_file = "new_reference_experimental_pication_interactions_report.csv"
    predictions_file = "predictions_with_energy_ranked.csv"

    try:
        results = check_protein_matching_per_residue(
            report_file, predictions_file,
            per_residue_pct=TOP_PERCENT_PER_RESIDUE,
            dihedral_cutoff_arg=DIHEDRAL_CUTOFF_ARG
        )

        print("SAMPLING RECOVERY RATES BEFORE RANKING:")
        print("-" * 60)
        for res_type in ['ARG', 'HIS', 'LYS']:
            res_data = results['per_residue_results_before_ranking'][res_type]
            print(f"{res_type}: {res_data['recovered_before_ranking']}/{res_data['total_experimental']} ({res_data['recovery_rate_before_ranking']:.2f}%)")

        overall_before = results['overall']['recovery_rate_before']
        overall_after = results['overall']['recovery_rate_after']
        print(f"\nOverall BEFORE RANKING: {results['overall']['recovered_interactions_before']}/{results['overall']['total_pi_cation_interactions']} ({overall_before:.2f}%)")

        print(f"\nSUCCESS RECOVERY RATE AFTER model filter (top % within dihedral cutoff for ARG):")
        print("-" * 60)
        total_recovered = 0
        total_experimental = 0
        for res_type in ['ARG', 'HIS', 'LYS']:
            res_data = results['per_residue_results'][res_type]
            recovered = res_data['recovered']
            total_exp = res_data['total_experimental']
            print(f"{res_type}: {recovered}/{total_exp} ({res_data['recovery_rate']:.2f}%)")
            total_recovered += recovered
            total_experimental += total_exp

        print(f"\nOverall AFTER RANKING: {results['overall']['recovered_interactions_after']}/{results['overall']['total_pi_cation_interactions']} ({overall_after:.2f}%)")

        # False Positive Filtered rate per residue (excluding HIS)
        print(f"\nFALSE POSITIVE FILTERED RATE PER RESIDUE TYPE:")
        print("-" * 50)
        for res_type in ['ARG', 'LYS']: # Excluding HIS
            rate = results['fp_filtered_rate_per_residue'][res_type]
            print(f"{res_type}: {rate:.2f}%")

        # PER-RESIDUE MISSED DETAILS AFTER RANKING
        print(f"\nPER-RESIDUE MISSED DETAILS AFTER RANKING:")
        print("-" * 60)
        for res_type in ['ARG', 'HIS', 'LYS']:
            res_data = results['per_residue_results'][res_type]
            if res_data['missed_interactions']:
                print(f"\n{res_type} - Missed ({len(res_data['missed_interactions'])} total):")
                for i, m in enumerate(res_data['missed_interactions'], 1):
                    print(f"  {i}. {m}")
            else:
                print(f"\n{res_type} - ✅ All interactions recovered!")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
