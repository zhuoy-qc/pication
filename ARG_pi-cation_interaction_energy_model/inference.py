import joblib
import pandas as pd
import numpy as np

def load_and_predict(model_path, new_data_path):
    """Load the trained model and make predictions on new data"""
    # Load the complete model package
    model_package = joblib.load(model_path)
    
    # Extract components
    model = model_package['model']
    scaler = model_package['scaler']
    selected_features = model_package['selected_features']
    all_feature_cols = model_package['all_feature_cols']
    test_mae = model_package['test_mae']
    best_params = model_package['best_params']
    
    print(f"Model loaded successfully")
    print(f"Test MAE from training: {test_mae:.4f}")
    print(f"Selected features: {selected_features}")
    print(f"Total features in original dataset: {len(all_feature_cols)}")
    print(f"Number of selected features: {len(selected_features)}")
    
    # Load new data
    new_data = pd.read_csv(new_data_path)
    
    # Apply same feature engineering as training
    new_data = new_data.rename(columns={'offset': 'delta_x', 'rz': 'delta_z', 'dihedral': 'dihedral_angle'})
    
    # Apply same feature engineering as in training
    new_data['inv_distance'] = 1 / new_data['distance']
    new_data['distance_sq'] = new_data['distance'] ** 2
    new_data['sin_dihedral'] = np.sin(np.radians(new_data['dihedral_angle']))
    new_data['cos_dihedral'] = np.cos(np.radians(new_data['dihedral_angle']))
    new_data['tan_dihedral'] = np.tan(np.radians(new_data['dihedral_angle']))
    new_data['delta_z_norm'] = new_data['delta_z'] / new_data['distance']
    new_data['delta_x_norm'] = new_data['delta_x'] / new_data['distance']
    new_data['delta_x_delta_z_ratio'] = new_data['delta_x'] / new_data['delta_z']
    new_data['distance_delta_z_ratio'] = new_data['distance'] / new_data['delta_z']
    new_data['distance_delta_x_ratio'] = new_data['distance'] / new_data['delta_x']
    new_data['log_distance'] = np.log(new_data['distance'])
    new_data['log_abs_delta_x'] = np.log(np.abs(new_data['delta_x']) + 1e-8)
    new_data['log_abs_delta_z'] = np.log(np.abs(new_data['delta_z']) + 1e-8)
    new_data['distance_cubed'] = new_data['distance'] ** 3
    new_data['delta_x_squared'] = new_data['delta_x'] ** 2
    new_data['delta_z_squared'] = new_data['delta_z'] ** 2
    new_data['distance_delta_x'] = new_data['distance'] * new_data['delta_x']
    new_data['distance_delta_z'] = new_data['distance'] * new_data['delta_z']
    new_data['delta_x_delta_z'] = new_data['delta_x'] * new_data['delta_z']
    new_data['effective_distance'] = np.sqrt(new_data['distance']**2 + new_data['delta_x']**2 + new_data['delta_z']**2)
    new_data['planar_distance'] = np.sqrt(new_data['delta_x']**2 + new_data['delta_z']**2)
    
    # Handle potential division by zero
    for col in ['delta_x_delta_z_ratio', 'distance_delta_z_ratio', 'distance_delta_x_ratio', 'tan_dihedral']:
        if col in new_data.columns:
            new_data[col] = new_data[col].replace([np.inf, -np.inf], np.nan)
    
    new_data = new_data.fillna(new_data.median())
    
    # Ensure all required columns exist
    for col in all_feature_cols:
        if col not in new_data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Select only the features used during training
    X_new = new_data[all_feature_cols]
    
    # Scale the data
    X_scaled = scaler.transform(X_new)
    X_scaled_df = pd.DataFrame(X_scaled, columns=all_feature_cols)
    
    # Select only the features the model was trained on
    X_selected = X_scaled_df[selected_features]
    
    # Make predictions
    predictions = model.predict(X_selected)
    
    # Add predictions to the original dataframe
    new_data['predicted_energy'] = predictions
    
    return new_data, predictions

# Example usage
if __name__ == "__main__":
    # Load model and make predictions
    results_df, predictions = load_and_predict('xgboost_model_optimized.pkl', 'combined_sorted.csv')
    
    print(f"Predictions made for {len(predictions)} samples")
    print(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
    
    # Save results
    results_df.to_csv('predictions_with_features.csv', index=False)
    print("Predictions saved to 'predictions_with_features.csv'")
