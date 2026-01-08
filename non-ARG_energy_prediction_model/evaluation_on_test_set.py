import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error

# Define file paths
model_path = 'model_output/final_model_20250919_025353.pkl'
data_path = 'cleaned.csv'
test_indices_path = 'model_output/test_indices.csv'

# Load the trained model
model = joblib.load(model_path)

# Load the dataset
df = pd.read_csv(data_path)
X = df[['delta_z', 'delta_x', 'dihedral_angle', 'distance']]
y = df['final_energy']

# Load test indices
test_indices_df = pd.read_csv(test_indices_path)
test_indices = test_indices_df['index'].values

# Prepare the test set
X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]

# Make predictions
y_pred = model.predict(X_test)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Compute absolute error for each sample
absolute_errors = np.abs(y_test.values - y_pred)

# Print only MAE
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Print first 10 sample predictions with input features
print("\nFirst 20 Sample Predictions with Input Features:")

# Create a DataFrame with renamed columns and desired format
first_10_input = X_test.iloc[:20].reset_index(drop=True)
first_10_input.columns = ['delta_z', 'delta_x', 'angle', 'distance']  # Rename dihedral_angle to angle

first_10_results = pd.DataFrame({
    'ref_energy': y_test.iloc[:20].values,
    'predicted_energy': y_pred[:20],
    'energy_error': absolute_errors[:20]
})

# Combine input features with predictions
final_output = pd.concat([first_10_input, first_10_results], axis=1)

print(final_output)
