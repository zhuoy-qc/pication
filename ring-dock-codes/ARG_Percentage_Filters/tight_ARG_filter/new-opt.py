import pandas as pd
import numpy as np
import re
import psutil
from multiprocessing import Pool, cpu_count
import time

# --- Your existing functions (keep them the same) ---
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

def apply_ranking_per_bin_fp(df_pred, bin_percentages_dict, bin_size=2):
    """
    Filter ARG predictions based on dihedral angle bins with percentage-based ranking.
    Uses a dictionary {bin_index: percentage} for each bin.
    Handles HIS and LYS normally (top 50% for HIS, top 5% for LYS).
    Returns filtered DataFrame and a dictionary of removed ARG predictions per bin.
    """
    df = df_pred.copy()
    df['ResidueType'] = df['Protein'].apply(extract_residue_type)

    filtered_rows = []
    removed_per_bin = {} # Dictionary to store removed ARG predictions for each bin

    # Handle ARG residues with dihedral-specific ranking
    arg_subset = df[df['ResidueType'] == 'ARG'].copy()
    if not arg_subset.empty and 'Dihedral_Angle' in df.columns:
        num_bins = int(np.ceil(90 / bin_size)) # Calculate number of bins

        for bin_idx in range(num_bins):
            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size if bin_idx < num_bins - 1 else 90

            bin_subset = arg_subset[
                (arg_subset['Dihedral_Angle'] >= bin_start) &
                (arg_subset['Dihedral_Angle'] <= bin_end) # Use <= to include the end for the last bin
            ].copy()

            if not bin_subset.empty:
                pct = bin_percentages_dict.get(bin_idx, 0) # Default to 0 if percentage not found
                # Ensure percentage is within 0-100 range
                pct = max(0, min(100, pct))
                n_total = len(bin_subset)
                n_keep = max(0, min(int(n_total * (pct / 100)), len(bin_subset)))
                
                if n_total > 0:
                    bin_sorted = bin_subset.sort_values('Energy_Rank')
                    kept_in_bin = bin_sorted.head(n_keep)
                    removed_in_bin = bin_sorted.tail(len(bin_sorted) - n_keep) if n_keep < len(bin_sorted) else pd.DataFrame(columns=bin_subset.columns)
                    
                    if not kept_in_bin.empty:
                        filtered_rows.append(kept_in_bin)
                    removed_per_bin[bin_idx] = removed_in_bin
                else:
                    removed_per_bin[bin_idx] = pd.DataFrame(columns=arg_subset.columns) # Empty DataFrame for bin
            else:
                removed_per_bin[bin_idx] = pd.DataFrame(columns=arg_subset.columns) # Empty DataFrame for bin
    else:
        # If no ARG or no dihedral, keep all original ARG if exists
        if not arg_subset.empty:
            filtered_rows.append(arg_subset)
        # Initialize removed_per_bin if needed
        num_bins = int(np.ceil(90 / bin_size))
        for bin_idx in range(num_bins):
             removed_per_bin[bin_idx] = pd.DataFrame(columns=df.columns) # Empty DataFrame for bin

    # Handle HIS and LYS normally (top 50% for HIS, top 5% for LYS)
    for res_type, pct in [('HIS', 50), ('LYS', 5)]:
        subset = df[df['ResidueType'] == res_type].copy()
        if not subset.empty:
            n_total = len(subset)
            n_keep = max(1, int(n_total * (pct / 100)))
            subset_sorted = subset.sort_values('Energy_Rank')
            filtered_rows.append(subset_sorted.head(n_keep))

    final_filtered_df = pd.concat(filtered_rows, ignore_index=True) if filtered_rows else df.iloc[0:0]
    return final_filtered_df, removed_per_bin


def calculate_metrics_per_bin_fp(report_file, predictions_file, bin_percentages_dict, bin_size=2):
    """
    Calculates metrics based on the provided bin_percentages_dict.
    Returns: ARG recovery rate, overall FP reduction rate, per-bin FP reduction rates, total FP removed
    """
    df_report = pd.read_csv(report_file)
    df_pred = pd.read_csv(predictions_file)

    df_report['ResidueType'] = df_report['Protein'].apply(extract_residue_type)
    df_pred['ResidueType'] = df_pred['Protein'].apply(extract_residue_type)

    # Apply the ranking filter with the current percentages
    df_pred_filtered, removed_per_bin = apply_ranking_per_bin_fp(df_pred, bin_percentages_dict, bin_size)

    # Calculate ARG-specific metrics after filter
    unique_report_all = set(df_report['Protein'].unique())
    arg_report = set(df_report[df_report['ResidueType'] == 'ARG']['Protein'].unique())
    arg_filtered = set(df_pred_filtered[df_pred_filtered['ResidueType'] == 'ARG']['Protein'].unique())

    arg_matched_after = arg_report & arg_filtered
    arg_recovery_rate_after = (len(arg_matched_after) / len(arg_report) * 100) if len(arg_report) > 0 else 0

    # Calculate FP reduction rates
    unique_pred_all = set(df_pred['Protein'].unique())
    predictions_not_in_exp_before = unique_pred_all - unique_report_all
    
    # Get original ARG predictions per bin for FP calculation
    arg_pred_original = df_pred[df_pred['ResidueType'] == 'ARG'].copy()
    num_bins = int(np.ceil(90 / bin_size))
    per_bin_fp_reduction_rates = []  # This will store REDUCTION percentages
    total_fp_removed = 0
    total_fp_before = 0

    for bin_idx in range(num_bins):
        bin_start = bin_idx * bin_size
        bin_end = (bin_idx + 1) * bin_size if bin_idx < num_bins - 1 else 90

        bin_original_arg_pred_set = set(
            arg_pred_original[
                (arg_pred_original['Dihedral_Angle'] >= bin_start) &
                (arg_pred_original['Dihedral_Angle'] <= bin_end)
            ]['Protein'].unique()
        )
        bin_original_arg_fp_set = bin_original_arg_pred_set & predictions_not_in_exp_before

        # Removed predictions for this bin (from apply_ranking_per_bin_fp)
        removed_bin_df = removed_per_bin.get(bin_idx, pd.DataFrame(columns=df_pred.columns))
        removed_bin_arg_fp_set = set(removed_bin_df[removed_bin_df['ResidueType'] == 'ARG']['Protein'].unique()) & predictions_not_in_exp_before

        n_fp_before_bin = len(bin_original_arg_fp_set)
        n_fp_removed_bin = len(removed_bin_arg_fp_set)

        if n_fp_before_bin > 0:
            # Calculate FP REDUCTION rate (what percentage of FPs were removed)
            fp_reduction_rate_bin = (n_fp_removed_bin / n_fp_before_bin) * 100
        else:
            fp_reduction_rate_bin = 0.0

        per_bin_fp_reduction_rates.append(fp_reduction_rate_bin)
        total_fp_removed += n_fp_removed_bin
        total_fp_before += n_fp_before_bin

    # Calculate overall FP REDUCTION rate
    overall_fp_reduction_rate = (total_fp_removed / total_fp_before * 100) if total_fp_before > 0 else 0.0

    return arg_recovery_rate_after, overall_fp_reduction_rate, per_bin_fp_reduction_rates, total_fp_removed


def optimize_single_bin_grid_search(args):
    """
    Perform grid search for a single bin across all integer percentages 1-100.
    Find the percentage that satisfies both constraints:
    - Recovery Rate >= 50%
    - FP Reduction Rate >= 30%
    Then maximize the sum (Recovery + FP Reduction) among valid solutions.
    """
    bin_idx, report_file, predictions_file, bin_size, min_recovery, min_fp_reduction = args
    
    best_percentage = 0
    best_recovery = 0
    best_fp_reduction = 0
    best_sum = -np.inf
    found_valid_solution = False
    
    print(f"  Optimizing bin {bin_idx}...", end=" ")
    start_time = time.time()
    
    # Test all integer percentages from 1 to 100
    for percentage in range(1, 91):
        # Create a dictionary with only this bin's percentage set
        temp_percentages = {bin_idx: percentage}
        recovery_rate, overall_fp_reduction, per_bin_fp_reductions, _ = calculate_metrics_per_bin_fp(
            report_file, predictions_file, temp_percentages, bin_size
        )
        
        # Get this bin's FP reduction rate
        current_bin_fp_reduction = per_bin_fp_reductions[bin_idx] if bin_idx < len(per_bin_fp_reductions) else 0.0
        
        # Check if this percentage satisfies BOTH constraints
        satisfies_recovery = recovery_rate >= min_recovery
        satisfies_fp_reduction = current_bin_fp_reduction >= min_fp_reduction
        
        if satisfies_recovery and satisfies_fp_reduction:
            found_valid_solution = True
            current_sum = recovery_rate + current_bin_fp_reduction
            
            if current_sum > best_sum:
                best_percentage = percentage
                best_recovery = recovery_rate
                best_fp_reduction = current_bin_fp_reduction
                best_sum = current_sum
    
    # If no percentage satisfies both constraints, relax to single constraint
    if not found_valid_solution:
        print("No percentage satisfies both constraints, relaxing...", end=" ")
        
        # First try: satisfy recovery constraint only
        for percentage in range(1, 91):
            temp_percentages = {bin_idx: percentage}
            recovery_rate, overall_fp_reduction, per_bin_fp_reductions, _ = calculate_metrics_per_bin_fp(
                report_file, predictions_file, temp_percentages, bin_size
            )
            current_bin_fp_reduction = per_bin_fp_reductions[bin_idx] if bin_idx < len(per_bin_fp_reductions) else 0.0
            
            if recovery_rate >= min_recovery:
                current_sum = recovery_rate + current_bin_fp_reduction
                if current_sum > best_sum:
                    best_percentage = percentage
                    best_recovery = recovery_rate
                    best_fp_reduction = current_bin_fp_reduction
                    best_sum = current_sum
                    found_valid_solution = True
        
        # If still no solution, satisfy FP reduction constraint only
        if not found_valid_solution:
            for percentage in range(1, 91):
                temp_percentages = {bin_idx: percentage}
                recovery_rate, overall_fp_reduction, per_bin_fp_reductions, _ = calculate_metrics_per_bin_fp(
                    report_file, predictions_file, temp_percentages, bin_size
                )
                current_bin_fp_reduction = per_bin_fp_reductions[bin_idx] if bin_idx < len(per_bin_fp_reductions) else 0.0
                
                if current_bin_fp_reduction >= min_fp_reduction:
                    current_sum = recovery_rate + current_bin_fp_reduction
                    if current_sum > best_sum:
                        best_percentage = percentage
                        best_recovery = recovery_rate
                        best_fp_reduction = current_bin_fp_reduction
                        best_sum = current_sum
                        found_valid_solution = True
        
        # If still no solution, use best available
        if not found_valid_solution:
            for percentage in range(1, 91):
                temp_percentages = {bin_idx: percentage}
                recovery_rate, overall_fp_reduction, per_bin_fp_reductions, _ = calculate_metrics_per_bin_fp(
                    report_file, predictions_file, temp_percentages, bin_size
                )
                current_bin_fp_reduction = per_bin_fp_reductions[bin_idx] if bin_idx < len(per_bin_fp_reductions) else 0.0
                
                current_sum = recovery_rate + current_bin_fp_reduction
                if current_sum > best_sum:
                    best_percentage = percentage
                    best_recovery = recovery_rate
                    best_fp_reduction = current_bin_fp_reduction
                    best_sum = current_sum
    
    end_time = time.time()
    constraint_status = "BOTH" if (best_recovery >= min_recovery and best_fp_reduction >= min_fp_reduction) else "PARTIAL"
    print(f"Completed in {end_time - start_time:.1f}s ({constraint_status})")
    
    return bin_idx, best_percentage, best_recovery, best_fp_reduction, best_sum, constraint_status


def optimize_all_bins_grid_search(report_file, predictions_file, bin_size=2, min_recovery=50.0, min_fp_reduction=30.0):
    """
    Optimize each bin independently using grid search across all integer percentages 1-100.
    Find the optimal percentage for each bin that satisfies:
    - Recovery Rate >= 50%
    - FP Reduction Rate >= 30%
    Then maximize the sum (Recovery + FP Reduction) among valid solutions.
    """
    num_bins = int(np.ceil(90 / bin_size))
    print(f"Performing grid search optimization for {num_bins} bins with bin size {bin_size}")
    print(f"Testing all integer percentages from 1% to 100%")
    print(f"CONSTRAINTS:")
    print(f"  - Minimum Recovery Rate: {min_recovery}%")
    print(f"  - Minimum FP Reduction Rate: {min_fp_reduction}%")
    print(f"Objective: Maximize (Recovery Rate + FP Reduction Rate) among valid solutions")
    print("-" * 80)
    
    # Prepare arguments for parallel processing
    args_list = [
        (bin_idx, report_file, predictions_file, bin_size, min_recovery, min_fp_reduction)
        for bin_idx in range(num_bins)
    ]
    
    # Use multiprocessing to optimize bins in parallel
    print(f"Using {cpu_count()} CPU cores for parallel grid search...")
    start_total_time = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(optimize_single_bin_grid_search, args_list)
    
    total_time = time.time() - start_total_time
    print(f"\nTotal optimization time: {total_time:.1f} seconds")
    
    # Collect results
    optimal_percentages = [0] * num_bins
    bin_recovery_rates = [0.0] * num_bins
    bin_fp_reduction_rates = [0.0] * num_bins
    bin_sums = [0.0] * num_bins
    constraint_statuses = [""] * num_bins
    
    for bin_idx, percentage, recovery, fp_reduction, bin_sum, status in results:
        optimal_percentages[bin_idx] = percentage
        bin_recovery_rates[bin_idx] = recovery
        bin_fp_reduction_rates[bin_idx] = fp_reduction
        bin_sums[bin_idx] = bin_sum
        constraint_statuses[bin_idx] = status
    
    # Calculate combined metrics using all optimal percentages
    print("\nCalculating final combined metrics...")
    optimal_percentages_dict = {i: p for i, p in enumerate(optimal_percentages)}
    final_recovery, final_overall_fp_reduction, final_per_bin_fp_reductions, final_fp_removed = calculate_metrics_per_bin_fp(
        report_file, predictions_file, optimal_percentages_dict, bin_size
    )
    
    # Print detailed results
    print(f"\n{'='*100}")
    print(f"GRID SEARCH OPTIMIZATION RESULTS")
    print(f"{'='*100}")
    print(f"Optimal Percentages per Bin (Bin Size {bin_size}°):")
    print(f"{'Bin Range':<12} {'Pct':<4} {'Recovery':<9} {'FP Reduct':<10} {'Sum':<8} {'Status':<12} {'Constraints':<20}")
    print(f"{'-'*100}")
    
    recovery_constraint_met = 0
    fp_constraint_met = 0
    both_constraints_met = 0
    
    for i, (perc, recovery, fp_reduction, bin_sum, status) in enumerate(zip(
        optimal_percentages, bin_recovery_rates, bin_fp_reduction_rates, bin_sums, constraint_statuses)):
        
        bin_start = i * bin_size
        bin_end = (i + 1) * bin_size if i < len(optimal_percentages) - 1 else 90
        
        recovery_ok = recovery >= min_recovery
        fp_ok = fp_reduction >= min_fp_reduction
        
        if recovery_ok and fp_ok:
            both_constraints_met += 1
            constraint_text = "✓ BOTH"
        elif recovery_ok:
            recovery_constraint_met += 1
            constraint_text = "✓ RECOVERY ONLY"
        elif fp_ok:
            fp_constraint_met += 1
            constraint_text = "✓ FP REDUCT ONLY"
        else:
            constraint_text = "✗ NONE"
        
        print(f"[{bin_start:2d}, {bin_end:2d})   {perc:3d}%  {recovery:6.2f}%  {fp_reduction:6.2f}%  {bin_sum:6.2f}%  {status:<12} {constraint_text}")
    
    print(f"{'-'*100}")
    
    # Print constraint summary
    print(f"\nCONSTRAINT SUMMARY:")
    print(f"  BOTH constraints satisfied: {both_constraints_met}/{num_bins} bins ({both_constraints_met/num_bins*100:.1f}%)")
    print(f"  Recovery only satisfied: {recovery_constraint_met}/{num_bins} bins")
    print(f"  FP Reduction only satisfied: {fp_constraint_met}/{num_bins} bins")
    print(f"  No constraints satisfied: {num_bins - (both_constraints_met + recovery_constraint_met + fp_constraint_met)}/{num_bins} bins")
    
    print(f"\nFINAL COMBINED RESULTS:")
    print(f"  ARG Recovery Rate: {final_recovery:.2f}%")
    print(f"  Overall ARG FP Reduction Rate: {final_overall_fp_reduction:.2f}%")
    print(f"  Sum (Recovery + FP Reduction): {final_recovery + final_overall_fp_reduction:.2f}%")
    print(f"  Total FP Removed: {final_fp_removed}")
    
    # Check if overall results meet constraints
    overall_recovery_ok = final_recovery >= min_recovery
    overall_fp_ok = final_overall_fp_reduction >= min_fp_reduction
    
    print(f"\nOVERALL CONSTRAINT CHECK:")
    print(f"  Recovery Rate >= {min_recovery}%: {final_recovery:.2f}% {'✓' if overall_recovery_ok else '✗'}")
    print(f"  FP Reduction Rate >= {min_fp_reduction}%: {final_overall_fp_reduction:.2f}% {'✓' if overall_fp_ok else '✗'}")
    
    # Save optimal percentages to file
    percentages_df = pd.DataFrame({
        'Bin_Index': range(num_bins),
        'Bin_Start': [i * bin_size for i in range(num_bins)],
        'Bin_End': [(i + 1) * bin_size if i < num_bins - 1 else 90 for i in range(num_bins)],
        'Optimal_Percentage': optimal_percentages,
        'Recovery_Rate': bin_recovery_rates,
        'FP_Reduction_Rate': bin_fp_reduction_rates,
        'Sum_Objective': bin_sums,
        'Constraint_Status': constraint_statuses
    })
    percentages_df.to_csv("optimal_percentages_grid_search_constrained.csv", index=False)
    print(f"\nOptimal percentages saved to 'optimal_percentages_grid_search_constrained.csv'")
    
    # Apply the optimal filter and save results
    df_pred = pd.read_csv(predictions_file)
    df_pred_filtered_optimal, _ = apply_ranking_per_bin_fp(df_pred, optimal_percentages_dict, bin_size)
    df_pred_filtered_optimal.to_csv("remained_predictions_optimized_constrained.csv", index=False)
    print(f"Optimized filtered predictions saved to 'remained_predictions_optimized_constrained.csv'")
    
    return optimal_percentages


def main():
    report_file = "newest_reference_experimental_pication_interactions_report.csv"
    predictions_file = "new_sample_with_energy_predicted.csv"

    # Perform grid search optimization with dual constraints
    optimal_percentages = optimize_all_bins_grid_search(
        report_file, predictions_file, 
        bin_size=2, 
        min_recovery=50.0,    # At least 50% recovery
        min_fp_reduction=30.0 # At least 30% FP reduction
    )

if __name__ == "__main__":
    main()
