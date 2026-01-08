import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import re
import argparse
import sys

def create_comprehensive_features(interaction_csv="all_sampled_poses_with-pi-cation-interactions.csv",
                                 results_csv="exhaust50_detailed_results.csv", 
                                 output_file="comprehensive_features.csv"):
    """Create comprehensive features from interaction and results CSVs"""
    # Read the interaction data
    interaction_df = pd.read_csv(interaction_csv)
    
    # Convert Ring_Type to numeric: '5-membered' -> 5, '6-membered' -> 6
    def convert_ring_type(ring_type):
        if pd.isna(ring_type):
            return np.nan
        if '5-membered' in str(ring_type):
            return 5
        elif '6-membered' in str(ring_type):
            return 6
        else:
            return np.nan
    
    interaction_df['Ring_Type_Numeric'] = interaction_df['Ring_Type'].apply(convert_ring_type)
    
    # Extract pose number from PDB_File
    def extract_pose_number(pdb_file):
        parts = pdb_file.replace('.pdb', '').split('_')
        if len(parts) >= 2:
            return int(parts[-1])
        else:
            return np.nan
    
    interaction_df['vina_pose_number'] = interaction_df['PDB_File'].apply(extract_pose_number)
    
    # Group interactions by PDB_ID and vina_pose_number
    grouped_interactions = interaction_df.groupby(['PDB_ID', 'vina_pose_number']).agg({
        'Protein_Residue_Type': list,
        'Predicted_Energy': list,
        'Dihedral_Angle': list,
        'Distance': list,
        'Offset': list,
        'Ring_Type_Numeric': list
    }).reset_index()
    
    # Function to process interaction details
    def process_interaction_details(row):
        residue_list = row['Protein_Residue_Type']
        energy_list = row['Predicted_Energy']
        dihedral_list = row['Dihedral_Angle']
        distance_list = row['Distance']
        offset_list = row['Offset']
        ring_type_list = row['Ring_Type_Numeric']
        
        # Count residue types
        lys_count = sum(1 for res in residue_list if res == 'LYS')
        arg_count = sum(1 for res in residue_list if res == 'ARG')
        his_count = sum(1 for res in residue_list if res == 'HIS')
        
        # Calculate residue-specific averages
        arg_dihedral_vals = [d for d, res in zip(dihedral_list, residue_list) if res == 'ARG' and pd.notna(d)]
        arg_distance_vals = [d for d, res in zip(distance_list, residue_list) if res == 'ARG' and pd.notna(d)]
        arg_offset_vals = [o for o, res in zip(offset_list, residue_list) if res == 'ARG' and pd.notna(o)]
        arg_ring_type_vals = [r for r, res in zip(ring_type_list, residue_list) if res == 'ARG' and pd.notna(r)]
        
        lys_distance_vals = [d for d, res in zip(distance_list, residue_list) if res == 'LYS' and pd.notna(d)]
        lys_offset_vals = [o for o, res in zip(offset_list, residue_list) if res == 'LYS' and pd.notna(o)]
        lys_ring_type_vals = [r for r, res in zip(ring_type_list, residue_list) if res == 'LYS' and pd.notna(r)]
        
        his_distance_vals = [d for d, res in zip(distance_list, residue_list) if res == 'HIS' and pd.notna(d)]
        his_offset_vals = [o for o, res in zip(offset_list, residue_list) if res == 'HIS' and pd.notna(o)]
        his_ring_type_vals = [r for r, res in zip(ring_type_list, residue_list) if res == 'HIS' and pd.notna(r)]
        
        # Calculate averages
        avg_dihedral_arg = np.mean(arg_dihedral_vals) if arg_dihedral_vals else np.nan
        avg_distance_arg = np.mean(arg_distance_vals) if arg_distance_vals else np.nan
        avg_offset_arg = np.mean(arg_offset_vals) if arg_offset_vals else np.nan
        avg_ring_type_arg = np.mean(arg_ring_type_vals) if arg_ring_type_vals else np.nan
        
        avg_distance_lys = np.mean(lys_distance_vals) if lys_distance_vals else np.nan
        avg_offset_lys = np.mean(lys_offset_vals) if lys_offset_vals else np.nan
        avg_ring_type_lys = np.mean(lys_ring_type_vals) if lys_ring_type_vals else np.nan
        
        avg_distance_his = np.mean(his_distance_vals) if his_distance_vals else np.nan
        avg_offset_his = np.mean(his_offset_vals) if his_offset_vals else np.nan
        avg_ring_type_his = np.mean(his_ring_type_vals) if his_ring_type_vals else np.nan
        
        # Separate energies by amino acid type
        lys_energies = [e for e, res in zip(energy_list, residue_list) if res == 'LYS']
        arg_energies = [e for e, res in zip(energy_list, residue_list) if res == 'ARG']
        his_energies = [e for e, res in zip(energy_list, residue_list) if res == 'HIS']
        
        return pd.Series([lys_count, arg_count, his_count, 
                         avg_dihedral_arg, avg_distance_arg, avg_offset_arg, avg_ring_type_arg,
                         avg_distance_lys, avg_offset_lys, avg_ring_type_lys,
                         avg_distance_his, avg_offset_his, avg_ring_type_his,
                         lys_energies, arg_energies, his_energies])
    
    # Apply processing function
    grouped_interactions[['LYS_Count', 'ARG_Count', 'HIS_Count', 
                         'Dihedral_Angle_ARG', 'Distance_ARG', 
                         'Offset_ARG', 'Ring_Type_Numeric_ARG',
                         'Distance_LYS', 'Offset_LYS', 'Ring_Type_Numeric_LYS',
                         'Distance_HIS', 'Offset_HIS', 'Ring_Type_Numeric_HIS',
                         'LYS_Energies', 'ARG_Energies', 'HIS_Energies']] = \
        grouped_interactions.apply(process_interaction_details, axis=1)
    
    # Add interaction count
    grouped_interactions['Num_Interactions'] = grouped_interactions['Protein_Residue_Type'].apply(len)
    
    # Function to create energy features
    def create_energy_features(energies, max_interactions=5):
        energy_features = []
        for i in range(max_interactions):
            if i < len(energies):
                energy_features.append(energies[i])
            else:
                energy_features.append(np.nan)
        return energy_features
    
    # Create LYS-specific energy features
    lys_energy_features = grouped_interactions['LYS_Energies'].apply(
        lambda x: create_energy_features(x if isinstance(x, list) else [])
    ).apply(pd.Series)
    lys_energy_features.columns = [f'LYS_Energy_{i+1}' for i in range(5)]
    
    # Create ARG-specific energy features
    arg_energy_features = grouped_interactions['ARG_Energies'].apply(
        lambda x: create_energy_features(x if isinstance(x, list) else [])
    ).apply(pd.Series)
    arg_energy_features.columns = [f'ARG_Energy_{i+1}' for i in range(5)]
    
    # Create HIS-specific energy features
    his_energy_features = grouped_interactions['HIS_Energies'].apply(
        lambda x: create_energy_features(x if isinstance(x, list) else [])
    ).apply(pd.Series)
    his_energy_features.columns = [f'HIS_Energy_{i+1}' for i in range(5)]
    
    # Combine all features
    interaction_summary = pd.concat([
        grouped_interactions[['PDB_ID', 'vina_pose_number', 'Num_Interactions',
                             'LYS_Count', 'ARG_Count', 'HIS_Count',
                             'Dihedral_Angle_ARG', 'Distance_ARG', 
                             'Offset_ARG', 'Ring_Type_Numeric_ARG',
                             'Distance_LYS', 'Offset_LYS', 'Ring_Type_Numeric_LYS',
                             'Distance_HIS', 'Offset_HIS', 'Ring_Type_Numeric_HIS']],
        lys_energy_features,
        arg_energy_features,
        his_energy_features
    ], axis=1)
    
    # Add amino acid interaction counts
    interaction_summary['LYS_Interaction_Count'] = interaction_summary['LYS_Energy_1'].notna().astype(int) + \
                                                   interaction_summary['LYS_Energy_2'].notna().astype(int) + \
                                                   interaction_summary['LYS_Energy_3'].notna().astype(int) + \
                                                   interaction_summary['LYS_Energy_4'].notna().astype(int) + \
                                                   interaction_summary['LYS_Energy_5'].notna().astype(int)
    interaction_summary['ARG_Interaction_Count'] = interaction_summary['ARG_Energy_1'].notna().astype(int) + \
                                                   interaction_summary['ARG_Energy_2'].notna().astype(int) + \
                                                   interaction_summary['ARG_Energy_3'].notna().astype(int) + \
                                                   interaction_summary['ARG_Energy_4'].notna().astype(int) + \
                                                   interaction_summary['ARG_Energy_5'].notna().astype(int)
    interaction_summary['HIS_Interaction_Count'] = interaction_summary['HIS_Energy_1'].notna().astype(int) + \
                                                   interaction_summary['HIS_Energy_2'].notna().astype(int) + \
                                                   interaction_summary['HIS_Energy_3'].notna().astype(int) + \
                                                   interaction_summary['HIS_Energy_4'].notna().astype(int) + \
                                                   interaction_summary['HIS_Energy_5'].notna().astype(int)
    
    # Create binary feature for more than one interaction
    interaction_summary['More_Than_One_Interaction'] = (interaction_summary['Num_Interactions'] > 1).astype(int)
    
    # Read the results data
    results_df = pd.read_csv(results_csv)
    
    # In the new CSV, 'Vina_Rank' is equivalent to the old 'Pose' column
    merged_df = results_df.merge(interaction_summary, 
                                 left_on=['PDB_ID', 'Vina_Rank'], 
                                 right_on=['PDB_ID', 'vina_pose_number'], 
                                 how='left')
    
    # Fill NaN values for poses without interactions
    fill_cols = ['Num_Interactions', 'More_Than_One_Interaction',
                 'LYS_Count', 'ARG_Count', 'HIS_Count',
                 'Dihedral_Angle_ARG', 'Distance_ARG', 'Offset_ARG', 'Ring_Type_Numeric_ARG',
                 'Distance_LYS', 'Offset_LYS', 'Ring_Type_Numeric_LYS',
                 'Distance_HIS', 'Offset_HIS', 'Ring_Type_Numeric_HIS',
                 'LYS_Interaction_Count', 'ARG_Interaction_Count', 'HIS_Interaction_Count']
    
    for col in fill_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)
    
    # Fill energy columns with 0 (no interaction)
    for aa_type in ['LYS', 'ARG', 'HIS']:
        for i in range(1, 6):
            col_name = f'{aa_type}_Energy_{i}'
            if col_name in merged_df.columns:
                merged_df[col_name] = merged_df[col_name].fillna(0)
    
    # Define features that can be used as model inputs
    feature_columns = [
        'Vina_Score', 'Vina_Rank',
        'Num_Interactions', 'More_Than_One_Interaction',
        'LYS_Count', 'ARG_Count', 'HIS_Count',
        'Dihedral_Angle_ARG', 'Distance_ARG', 'Offset_ARG', 'Ring_Type_Numeric_ARG',
        'Distance_LYS', 'Offset_LYS', 'Ring_Type_Numeric_LYS',
        'Distance_HIS', 'Offset_HIS', 'Ring_Type_Numeric_HIS',
        'LYS_Energy_1', 'LYS_Energy_2', 'LYS_Energy_3', 'LYS_Energy_4', 'LYS_Energy_5',
        'ARG_Energy_1', 'ARG_Energy_2', 'ARG_Energy_3', 'ARG_Energy_4', 'ARG_Energy_5',
        'HIS_Energy_1', 'HIS_Energy_2', 'HIS_Energy_3', 'HIS_Energy_4', 'HIS_Energy_5',
        'LYS_Interaction_Count', 'ARG_Interaction_Count', 'HIS_Interaction_Count',
    ]
    
    # Add identifier columns separately
    model_features = merged_df[['PDB_ID', 'Vina_Rank']].copy()
    model_features[feature_columns] = merged_df[feature_columns]
    
    # Reorder columns: identifiers first, then features
    ordered_columns = ['PDB_ID', 'Vina_Rank'] + feature_columns
    result_df = model_features[ordered_columns].copy()
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Comprehensive features saved to {output_file}")
    
    return result_df

def load_model_and_predict(input_csv, model_path='vina_failure_finetuned_best_model.pkl'):
    """Load the model and make predictions"""
    # Load the trained model
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    
    model = model_info['classifier']
    scaler = model_info['scaler']
    selector = model_info['selector']
    feature_columns = model_info['feature_columns']
    
    # Load the new data
    df = pd.read_csv(input_csv)
    
    # Check if RMSD column exists
    if 'RMSD' not in df.columns:
        df['RMSD'] = np.nan
        df['Is_Good_Pose'] = np.nan
    else:
        df['Is_Good_Pose'] = (df['RMSD'] <= 2.0).astype(int)
    
    # Check if all required feature columns exist
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for col in missing_features:
            df[col] = np.nan
    
    # Prepare features
    X = df[feature_columns]
    
    # Handle missing values
    for col in feature_columns:
        if X[col].dtype in ['float64', 'int64']:
            median_val = X[col].median()
            X.loc[:, col] = X[col].fillna(median_val)
    
    # Scale and select features
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_selected)[:, 1]
    y_pred = model.predict(X_selected)
    
    # Add predictions to the dataframe
    df['Model_Probability'] = y_pred_proba
    df['Model_Prediction'] = y_pred
    
    return df

def get_top_predictions(df, top_n=4):
    """Get top N predictions per PDB_ID by model probability"""
    ranked_data = []
    
    for pdb_id in df['PDB_ID'].unique():
        pdb_group = df[df['PDB_ID'] == pdb_id].copy()
        
        # Sort by model probability descending
        pdb_group_sorted = pdb_group.sort_values(by='Model_Probability', ascending=False).reset_index(drop=True)
        
        # Add model rank
        pdb_group_sorted['Model_Rank'] = pdb_group_sorted.index + 1
        
        # Filter for top N predictions per PDB_ID
        top_n_data = pdb_group_sorted.head(top_n)
        
        # Select required columns
        for idx, row in top_n_data.iterrows():
            ranked_data.append({
                'PDB_ID': row['PDB_ID'],
                'Vina_Rank': row['Vina_Rank'],
                'Model_Rank': row['Model_Rank'],
                'Model_Probability': row['Model_Probability']
            })
    
    return pd.DataFrame(ranked_data)

def merge_with_interactions(predictions_df, interactions_csv='interactions_autobox4_ex50.csv', 
                          output_csv='model_interactions.csv', top_n=4):
    """Merge predictions with interaction data"""
    interactions_df = pd.read_csv(interactions_csv)
    
    # Function to extract Vina rank from PDB filename
    def extract_vina_rank(pdb_file):
        match = re.search(r'complex_(\d+)\.pdb', pdb_file)
        if match:
            return int(match.group(1))
        return None
    
    interactions_df['Extracted_Vina_Rank'] = interactions_df['PDB_File'].apply(extract_vina_rank)
    
    # Merge the dataframes
    merged_df = pd.merge(
        predictions_df, 
        interactions_df, 
        left_on=['PDB_ID', 'Vina_Rank'], 
        right_on=['PDB_ID', 'Extracted_Vina_Rank'],
        how='inner'
    )
    
    # Select columns to keep
    final_columns = ['PDB_ID', 'Vina_Rank', 'Model_Rank', 'Model_Probability'] + \
                    [col for col in interactions_df.columns if col not in ['Extracted_Vina_Rank']]
    
    result_df = merged_df[final_columns].copy()
    result_df.to_csv(output_csv, index=False)
    
    print(f"Merged {len(result_df)} rows from both CSVs")
    print(f"Output saved to {output_csv}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Generate features, predict, and output top results')
    parser.add_argument('--interaction_csv', default='interactions_autobox4_ex50.csv', 
                       help='Path to interactions CSV file')
    parser.add_argument('--results_csv', default='no_rmsd.csv', 
                       help='Path to results CSV file (without RMSD)')
    parser.add_argument('--model_path', default='vina_failure_finetuned_best_model.pkl', 
                       help='Path to trained model file')
    parser.add_argument('--top_n', type=int, default=4, 
                       help='Number of top results to output per PDB_ID (default: 4)')
    parser.add_argument('--features_output', default='comprehensive_features.csv', 
                       help='Output path for features CSV')
    parser.add_argument('--predictions_output', default='predictions_with_model_scores.csv', 
                       help='Output path for predictions CSV')
    parser.add_argument('--final_output', default='model_interactions.csv', 
                       help='Output path for final merged CSV')
    
    args = parser.parse_args()
    
    print("Step 1: Generating comprehensive features...")
    features_df = create_comprehensive_features(
        interaction_csv=args.interaction_csv,
        results_csv=args.results_csv,
        output_file=args.features_output
    )
    
    print("Step 2: Loading model and making predictions...")
    predictions_df = load_model_and_predict(
        input_csv=args.features_output,
        model_path=args.model_path
    )
    
    # Save predictions
    predictions_df.to_csv(args.predictions_output, index=False)
    print(f"Predictions saved to {args.predictions_output}")
    
    print(f"Step 3: Getting top {args.top_n} predictions per PDB_ID...")
    top_predictions_df = get_top_predictions(predictions_df, top_n=args.top_n)
    
    print("Step 4: Merging with interaction data...")
    final_df = merge_with_interactions(
        top_predictions_df, 
        interactions_csv=args.interaction_csv,
        output_csv=args.final_output,
        top_n=args.top_n
    )
    
    print("Process completed successfully!")
    print(f"Generated files: {args.features_output}, {args.predictions_output}, {args.final_output}")

if __name__ == "__main__":
    main()
