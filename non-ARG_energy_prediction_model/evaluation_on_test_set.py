import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
import os

# Load the saved model
output_dir = 'model_output'
model_files = [f for f in os.listdir(output_dir) if f.startswith('final_model_') and f.endswith('.pkl')]
latest_model_file = sorted(model_files)[-1]  # Get most recent model
model_path = os.path.join(output_dir, latest_model_file)

print(f"Loading model: {model_path}")
model = joblib.load(model_path)

# Load the data
df = pd.read_csv('cleaned.csv')
# Rename 'angle' to 'dihedral_angle' to match the model's expected feature names, actually angle is used 
X = df[['delta_z', 'delta_x', 'angle', 'distance']].copy()
X.rename(columns={'angle': 'dihedral_angle'}, inplace=True)
y = df['final_energy']

# Load saved test indices
test_indices_path = os.path.join(output_dir, 'test_indices.csv')
test_indices_df = pd.read_csv(test_indices_path)
test_indices = test_indices_df['index'].values

# Get test set
X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=== TEST SET PERFORMANCE ===")
print(f"Model: {latest_model_file}")
print(f"Test samples: {len(y_test)}")
print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {np.sqrt(mse):.6f}")
print(f"MAE: {mae:.6f}")
