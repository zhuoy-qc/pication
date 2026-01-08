import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import pickle
import optuna
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_feature_columns_safe(train_df, exclude_cols={'PDB_ID', 'Pose', 'RMSD', 'Is_Good_Pose'}):
    """
    Get feature columns using only training data to avoid data leakage
    """
    # Columns to exclude initially
    candidate_features = [col for col in train_df.columns if col not in exclude_cols]
    
    # Check for columns with all zero values in training data only
    zero_columns = []
    for col in candidate_features:
        if train_df[col].dtype in ['int64', 'float64']:
            if (train_df[col] == 0).all():
                zero_columns.append(col)
    
    print(f"Removing columns with all zero values: {zero_columns}")
    
    # Remove zero-value columns
    candidate_features = [col for col in candidate_features if col not in zero_columns]
    
    # Identify duplicate columns (same values in same order) in training data only
    duplicate_pairs = []
    for i in range(len(candidate_features)):
        for j in range(i + 1, len(candidate_features)):
            col1, col2 = candidate_features[i], candidate_features[j]
            if train_df[col1].equals(train_df[col2]):
                duplicate_pairs.append((col1, col2))
    
    print(f"Found duplicate column pairs: {duplicate_pairs}")
    
    # Remove one column from each duplicate pair
    removed_duplicates = set()
    final_features = []
    
    for col in candidate_features:
        if col in removed_duplicates:
            continue
            
        # Check if this column is part of a duplicate pair and we haven't removed its duplicate yet
        is_duplicate = False
        for dup_col1, dup_col2 in duplicate_pairs:
            if col == dup_col1 and dup_col2 in candidate_features and dup_col2 not in removed_duplicates:
                # Keep the first one alphabetically and mark the second for removal
                if col > dup_col2:
                    final_features.append(dup_col2)
                    removed_duplicates.add(dup_col2)
                    removed_duplicates.add(col)
                else:
                    final_features.append(col)
                    removed_duplicates.add(dup_col1)
                    removed_duplicates.add(col)
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_features.append(col)
    
    print(f"Using {len(final_features)} feature columns: {final_features[:10]}{'...' if len(final_features) > 10 else ''}")
    
    return final_features

def objective_finetune(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    """
    Fine-tuning objective function with expanded parameter search space
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba)
    
    return auc_score

def train_finetuned_best_model(input_csv="combined_data_for_model.csv", n_trials=50, random_state=42):
    """
    Fine-tune the best performing model (Vina Failure Model No Rank Limit, No RFE)
    """
    # Load the data
    df = pd.read_csv(input_csv)
    df['Is_Good_Pose'] = (df['RMSD'] <= 2.0).astype(int)

    print("Dataset shape:", df.shape)

    # Identify proteins where Vina top1 fails (RMSD > 2.0) - this is OK to do on full dataset
    # because we're only identifying which proteins to include, not using their feature values
    vina_top1_per_protein = df.loc[df.groupby('PDB_ID')['Vina_Score'].idxmin()]
    vina_failure_proteins = vina_top1_per_protein[vina_top1_per_protein['RMSD'] > 2.0]['PDB_ID'].unique()

    # Filter dataset to only include poses from Vina failure proteins
    df_vina_failures = df[df['PDB_ID'].isin(vina_failure_proteins)].copy()

    # Split the failure proteins into train/val/test
    unique_failure_pdb_ids = df_vina_failures['PDB_ID'].unique()
    train_pdb_ids, temp_pdb_ids = train_test_split(
        unique_failure_pdb_ids,
        test_size=0.2,
        random_state=random_state
    )
    val_pdb_ids, test_pdb_ids = train_test_split(
        temp_pdb_ids,
        test_size=0.5,
        random_state=random_state
    )

    train_df = df_vina_failures[df_vina_failures['PDB_ID'].isin(train_pdb_ids)].copy()
    val_df = df_vina_failures[df_vina_failures['PDB_ID'].isin(val_pdb_ids)].copy()
    test_df = df_vina_failures[df_vina_failures['PDB_ID'].isin(test_pdb_ids)].copy()

    print(f"\nTraining PDBs: {len(train_pdb_ids)}, Validation PDBs: {len(val_pdb_ids)}, Testing PDBs: {len(test_pdb_ids)}")
    print(f"Training poses: {len(train_df)}, Validation poses: {len(val_df)}, Testing poses: {len(test_df)}")

    # Get feature columns using ONLY training data to prevent leakage
    all_feature_columns = get_feature_columns_safe(train_df)

    # Prepare features for each set
    X_train = train_df[all_feature_columns]
    y_train = train_df['Is_Good_Pose']
    X_val = val_df[all_feature_columns]
    y_val = val_df['Is_Good_Pose']
    X_test = test_df[all_feature_columns]
    y_test = test_df['Is_Good_Pose']

    # Handle missing values using only training data statistics
    fill_values = {}
    for col in all_feature_columns:
        if X_train[col].dtype in ['float64', 'int64']:
            fill_values[col] = X_train[col].median()
            X_train.loc[:, col] = X_train[col].fillna(fill_values[col])
            X_val.loc[:, col] = X_val[col].fillna(fill_values[col])
            X_test.loc[:, col] = X_test[col].fillna(fill_values[col])

    # Scale using only training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection using ONLY training data
    print(f"\nPerforming feature selection on training data only...")
    k_to_select = min(15, len(all_feature_columns))
    selector = SelectKBest(score_func=f_classif, k=k_to_select)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    selected_features = [all_feature_columns[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} features: {selected_features}")

    # Calculate class weight using only training data
    bad_count = (y_train == 0).sum()
    good_count = (y_train == 1).sum()
    scale_pos_weight = bad_count / good_count if good_count > 0 else 1.0

    # Extended hyperparameter optimization using Optuna
    print(f"\nPerforming EXTENDED hyperparameter optimization with Optuna ({n_trials} trials) for the BEST model...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_finetune(trial, X_train_selected, y_train, X_val_selected, y_val, scale_pos_weight),
        n_trials=n_trials
    )

    best_params = study.best_params
    best_params.update({
        'random_state': random_state,
        'scale_pos_weight': scale_pos_weight
    })

    print(f"Best fine-tuned parameters: {best_params}")
    print(f"Best validation AUC: {study.best_value:.4f}")

    # Train final fine-tuned model with best parameters
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train_selected, y_train)

    # Get predictions on test set
    y_test_pred = best_model.predict(X_test_selected)
    y_test_proba = best_model.predict_proba(X_test_selected)[:, 1]

    # Calculate pose-level metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\nPOSE-LEVEL METRICS (on Vina failure proteins TEST SET):")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"AUC-ROC: {test_auc:.4f}")

    # Evaluate at protein level on the Vina failure TEST set
    print(f"\nEVALUATING AT PROTEIN LEVEL (Vina failure proteins TEST SET Only)...")
    eval_df = test_df.copy()
    eval_df['Model_Probability'] = y_test_proba

    # Model top1 success in Vina failure proteins
    model_top1_df = eval_df.loc[eval_df.groupby('PDB_ID')['Model_Probability'].idxmax()]
    model_top1_good = (model_top1_df['RMSD'] <= 2.0).sum()

    # Vina top1 success in these proteins
    vina_top1_df = eval_df.loc[eval_df.groupby('PDB_ID')['Vina_Score'].idxmin()]
    vina_top1_good = (vina_top1_df['RMSD'] <= 2.0).sum()

    total_failure_proteins = eval_df['PDB_ID'].nunique()

    # Combined success (model OR vina)
    combined_good = len(
        set(model_top1_df[model_top1_df['RMSD'] <= 2.0]['PDB_ID']) |
        set(vina_top1_df[vina_top1_df['RMSD'] <= 2.0]['PDB_ID'])
    )

    print(f"\nPROTEIN-LEVEL RESULTS (Vina failure proteins TEST SET only):")
    print(f"Total Vina failure proteins tested: {total_failure_proteins}")
    print(f"Model top1 finds good pose: {model_top1_good}/{total_failure_proteins} ({model_top1_good/total_failure_proteins:.3f})")
    print(f"Vina top1 finds good pose: {vina_top1_good}/{total_failure_proteins} ({vina_top1_good/total_failure_proteins:.3f})")
    print(f"Combined (model OR vina): {combined_good}/{total_failure_proteins} ({combined_good/total_failure_proteins:.3f})")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': [all_feature_columns[i] for i in selector.get_support(indices=True)],
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 features in Fine-tuned Vina Failure Model:")
    print(feature_importance.head(10))

    # Save the fine-tuned model
    model_name = f'vina_failure_finetuned_best_model.pkl'
    model_type = f'vina_failure_finetuned_best'

    # Save the model
    model_info = {
        'classifier': best_model,
        'scaler': scaler,
        'feature_columns': all_feature_columns,
        'selected_features': [all_feature_columns[i] for i in selector.get_support(indices=True)],
        'selector': selector,
        'model_type': model_type,
        'training_stats': {
            'total_poses': len(df),
            'good_poses': df['Is_Good_Pose'].sum(),
            'bad_poses': (df['Is_Good_Pose'] == 0).sum(),
            'vina_failure_proteins_trained': len(vina_failure_proteins),
            'vina_rank_limit': None,
            'use_rfe': False,
            'train_val_test_splits': (train_pdb_ids, val_pdb_ids, test_pdb_ids)
        },
        'hyperparameters': best_params,
        'optuna_study': study
    }

    with open(model_name, 'wb') as f:
        pickle.dump(model_info, f)

    print(f"\nFine-tuned best model saved as '{model_name}'")
    return model_info, study

if __name__ == "__main__":
    print("Fine-tuning the best performing Vina failure model...")
    print("This will run extensive hyperparameter optimization to improve performance.")
    
    finetuned_model, study = train_finetuned_best_model(n_trials=50)
    
    print(f"\nOptimization completed!")
    print(f"Best parameters found: {study.best_params}")
    print(f"Best validation AUC: {study.best_value:.4f}")
    
    # Compare with original best result
    print(f"\nComparing with original best result (+0.073 improvement)")
    print(f"Fine-tuning may achieve even better performance through extended search.")
