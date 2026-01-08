import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define output directory
output_dir = 'results'  # You can change this to your preferred output directory
os.makedirs(output_dir, exist_ok=True)

# Load the model
model_package = joblib.load('xgboost_model_optimized.pkl')
best_model = model_package['model']
scaler = model_package['scaler']
selected_features = model_package['selected_features']
all_feature_cols = model_package['all_feature_cols']

# Load the original data to recreate test set
df = pd.read_csv('combined_sorted.csv')
df = df.rename(columns={'offset': 'delta_x', 'rz': 'delta_z', 'dihedral': 'dihedral_angle'})

# Apply same feature engineering as training
df['inv_distance'] = 1 / df['distance']
df['distance_sq'] = df['distance'] ** 2
df['sin_dihedral'] = np.sin(np.radians(df['dihedral_angle']))
df['cos_dihedral'] = np.cos(np.radians(df['dihedral_angle']))
df['tan_dihedral'] = np.tan(np.radians(df['dihedral_angle']))
df['delta_z_norm'] = df['delta_z'] / df['distance']
df['delta_x_norm'] = df['delta_x'] / df['distance']
df['delta_x_delta_z_ratio'] = df['delta_x'] / df['delta_z']
df['distance_delta_z_ratio'] = df['distance'] / df['delta_z']
df['distance_delta_x_ratio'] = df['distance'] / df['delta_x']
df['log_distance'] = np.log(df['distance'])
df['log_abs_delta_x'] = np.log(np.abs(df['delta_x']) + 1e-8)
df['log_abs_delta_z'] = np.log(np.abs(df['delta_z']) + 1e-8)
df['distance_cubed'] = df['distance'] ** 3
df['delta_x_squared'] = df['delta_x'] ** 2
df['delta_z_squared'] = df['delta_z'] ** 2
df['distance_delta_x'] = df['distance'] * df['delta_x']
df['distance_delta_z'] = df['distance'] * df['delta_z']
df['delta_x_delta_z'] = df['delta_x'] * df['delta_z']
df['effective_distance'] = np.sqrt(df['distance']**2 + df['delta_x']**2 + df['delta_z']**2)
df['planar_distance'] = np.sqrt(df['delta_x']**2 + df['delta_z']**2)

# Handle potential division by zero
for col in ['delta_x_delta_z_ratio', 'distance_delta_z_ratio', 'distance_delta_x_ratio', 'tan_dihedral']:
    if col in df.columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

df = df.fillna(df.median())

# Recreate the 8:1:1 split to get test set
X = df[all_feature_cols]
y = df['energy']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)

# Prepare the test data
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=all_feature_cols)
X_test_selected = X_test_scaled_df[selected_features]

# Make predictions
y_pred = best_model.predict(X_test_selected)

# Calculate metrics
r2_score_test = r2_score(y_test, y_pred)
mse_score_test = mean_squared_error(y_test, y_pred)
mae_score_test = mean_absolute_error(y_test, y_pred)

print("\n=== FINAL RESULTS ===")
print(f"\nTest Scores:")
print(f"R²: {r2_score_test:.4f}")
print(f"MSE: {mse_score_test:.6f}")
print(f"MAE: {mae_score_test:.4f}")

# Feature importance analysis
print("\n=== FEATURE IMPORTANCE ===")
feature_names = selected_features  # Use selected features
importances = best_model.feature_importances_
print("Feature importances:")
for idx in np.argsort(importances)[::-1]:
    print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

# Visualization - Feature Importance
plt.figure(figsize=(8, 6))
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices], align='center', color='steelblue')
plt.xticks(range(len(importances)), [feature_names[idx] for idx in indices], rotation=45, ha='right')
plt.ylabel('Importance Score', fontsize=12)
plt.title('Feature Importance - final_energy', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.show()

# Plot actual vs predicted with R² and MAE (energy range from lowest to 0)
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the scatter points
ax.scatter(y_test, y_pred, alpha=0.6, s=30, edgecolors='none', c='steelblue')

# Determine the range: from the lowest energy value to 0
min_val = min(y_test.min(), y_pred.min())
max_val = 0  # Fixed upper limit at 0

# Customize the plot for scientific publication
ax.set_xlabel('Actual final_energy', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted final_energy', fontsize=14, fontweight='bold')
ax.set_title(f'Actual vs Predicted final_energy\n(R² = {r2_score_test:.4f}, MAE = {mae_score_test:.4f})', 
             fontsize=16, fontweight='bold', pad=20)

# Set axis limits to show range from lowest to 0
ax.set_xlim([min_val, 0])
ax.set_ylim([min_val, 0])

# Add grid for better readability
ax.grid(True, linestyle='--', alpha=0.6, color='gray')

# Add text box with metrics
textstr = f'R² = {r2_score_test:.4f}\nMAE = {mae_score_test:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props, fontweight='bold')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save with high DPI for publication quality
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.pdf'), bbox_inches='tight',
            facecolor='white', edgecolor='none')

plt.show()

print(f"High-quality plot saved as 'actual_vs_predicted.png' and 'actual_vs_predicted.pdf'")
