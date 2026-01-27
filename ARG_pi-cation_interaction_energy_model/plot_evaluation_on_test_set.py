import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

# Define output directory
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Load the model
print("Loading model...")
model_package = joblib.load('xgboost_model_optimized.pkl')
best_model = model_package['model']
scaler = model_package['scaler']
selected_features = model_package['selected_features']
all_feature_cols = model_package['all_feature_cols']

# Load and prepare data
print("Loading data...")
df = pd.read_csv('combined_sorted.csv')
df = df.rename(columns={'offset': 'delta_x', 'rz': 'delta_z', 'dihedral': 'dihedral_angle'})

# Apply feature engineering
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

# Split data
print("Splitting data...")
X = df[all_feature_cols]
y = df['energy']
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)

# Prepare test data and make predictions
print("Making predictions...")
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=all_feature_cols)
X_test_selected = X_test_scaled_df[selected_features]
y_pred = best_model.predict(X_test_selected)

# Calculate metrics
r2_score_test = r2_score(y_test, y_pred)
mae_score_test = mean_absolute_error(y_test, y_pred)

# Create custom colormap: white -> yellow -> orange -> red
colors = ['white', '#FFFF00', '#FFA500', '#FF4500', '#FF0000', '#8B0000']
cmap = LinearSegmentedColormap.from_list('density_cmap', colors, N=256)

# Calculate point density for coloring using KDE
print("Calculating point density...")
xy = np.vstack([y_test, y_pred])
try:
    kde = gaussian_kde(xy)
    density = kde(xy)
    # Normalize density on log scale for better visual contrast
    log_density = np.log1p(density)
    point_colors = cmap(log_density / log_density.max())
except:
    # Fallback to 2D histogram for density
    print("Using histogram fallback for density calculation")
    hist, xedges, yedges = np.histogram2d(y_test, y_pred, bins=50)
    x_indices = np.digitize(y_test, xedges) - 1
    y_indices = np.digitize(y_pred, yedges) - 1
    x_indices = np.clip(x_indices, 0, hist.shape[0]-1)
    y_indices = np.clip(y_indices, 0, hist.shape[1]-1)
    density = hist[x_indices, y_indices]
    log_density = np.log1p(density)
    point_colors = cmap(log_density / log_density.max())

# Create figure
fig, ax = plt.subplots(figsize=(10, 10))

# Create scatter plot with circular markers and density-based colors
scatter = ax.scatter(y_test, y_pred, c=point_colors, s=30, alpha=0.7, 
                     marker='o', edgecolors='none', linewidth=0)

# Add perfect prediction line with high transparency
min_val = min(y_test.min(), y_pred.min())
max_val = 0
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.15)  # Very transparent

# Set axis labels with larger font
ax.set_xlabel('Reference interaction energy (kcal/mol)', fontsize=18, fontweight='bold')
ax.set_ylabel('Predicted interaction energy (kcal/mol)', fontsize=18, fontweight='bold')

# Set axis limits
ax.set_xlim([min_val, 0])
ax.set_ylim([min_val, 0])

# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=16)

# Calculate position for R² and MAE below the dashed line
# Find the midpoint for placement
x_mid = min_val + (max_val - min_val) * 0.6  # 60% from min to max
y_mid = min_val + (max_val - min_val) * 0.3  # 30% from min to max (below line)

# Add R² and MAE text box below the dashed line
textstr = f'R² = {r2_score_test:.3f}\nMAE = {mae_score_test:.3f} kcal/mol'
props = dict(boxstyle='round', facecolor='white', alpha=0.9, 
             edgecolor='black', linewidth=2)
# Position in data coordinates, not axes coordinates
ax.text(x_mid, y_mid, textstr, fontsize=18, fontweight='bold', 
        verticalalignment='top', bbox=props)

# Remove grid
ax.grid(False)

# Add compact colorbar at top-left with "Density" on the right
# Create a separate axes for the colorbar in the top-left corner
# Position: [left, bottom, width, height] in figure coordinates
cbar_ax = fig.add_axes([0.15, 0.85, 0.25, 0.03])  # Top-left position

# Create gradient for colorbar
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack((gradient, gradient))
im = cbar_ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 1])

# Customize colorbar - only show Low and High labels
cbar_ax.set_yticks([])
cbar_ax.set_xticks([0, 1])  # Only show endpoints
cbar_ax.set_xticklabels(['Low', 'High'], fontsize=14, fontweight='bold')

# Remove ticks and spines for cleaner look
cbar_ax.tick_params(axis='x', which='both', length=0)  # Remove tick marks
for spine in cbar_ax.spines.values():
    spine.set_visible(False)

# Add "Density" label on the right side
# Create a separate text label positioned to the right of the colorbar
# Position in figure coordinates: [x, y, text, transform, ...]
fig.text(0.41, 0.865, 'Density', fontsize=14, fontweight='bold', 
         verticalalignment='center', horizontalalignment='left')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
print("Saving plot...")
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), 
            dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"Plot saved to {os.path.join(output_dir, 'actual_vs_predicted.png')}")
plt.show()
