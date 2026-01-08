import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import time
import warnings
import os
from datetime import datetime
import optuna
from scipy.stats import randint, uniform

# Suppress warnings
warnings.filterwarnings('ignore')

# Set universal font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Set random seeds
SEED = 1
np.random.seed(SEED)

# Create output directory
output_dir = 'model_output'
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv('cleaned.csv')

# Define feature matrix and target vector (single output)
X = df[['delta_z', 'delta_x', 'angle', 'distance']]
y = df['final_energy']  # Single target

# Check for existing test indices
test_indices_path = os.path.join(output_dir, 'test_indices.csv')
if os.path.exists(test_indices_path):
    print("\nLoading existing test indices...")
    test_indices_df = pd.read_csv(test_indices_path)
    test_indices = test_indices_df['index'].values
    train_val_indices = np.setdiff1d(np.arange(len(df)), test_indices)
else:
    print("\nCreating new test indices...")
    train_val_indices, test_indices = train_test_split(
        np.arange(len(df)), test_size=0.1, random_state=SEED
    )
    test_indices_df = pd.DataFrame({'index': test_indices, 'dataset': 'test'})
    test_indices_df.to_csv(test_indices_path, index=False)

# Split data using consistent indices
X_train_val, X_test = X.iloc[train_val_indices], X.iloc[test_indices]
y_train_val, y_test = y.iloc[train_val_indices], y.iloc[test_indices]

train_indices, val_indices = train_test_split(
    train_val_indices, test_size=0.111, random_state=SEED
)

X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

# Display data split information
print(f"\nData splits:")
print(f"Training: {X_train.shape[0]} points ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Validation: {X_val.shape[0]} points ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"Test: {X_test.shape[0]} points ({X_test.shape[0]/len(df)*100:.1f}%)")

# Hyperparameter optimization
def optimize_hyperparameters(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }

    model = GradientBoostingRegressor(**params, random_state=SEED)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    return r2_score(y_val, y_val_pred)

print("\nOptimizing hyperparameters with Optuna...")
study = optuna.create_study(direction='maximize')
study.optimize(optimize_hyperparameters, n_trials=50, n_jobs=1)

best_params = study.best_params
print(f"\nBest hyperparameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# Training with monitoring
class TrainingMonitor:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_r2 = []
        self.val_r2 = []
        self.start_time = time.time()
        self.best_model = None
        self.best_score = -np.inf
        self.best_iter = 0

    def record(self, model, X_train, y_train, X_val, y_val, iteration):
        # Training metrics
        y_train_pred = model.predict(X_train)
        train_loss = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # Validation metrics
        y_val_pred = model.predict(X_val)
        val_loss = mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_r2.append(train_r2)
        self.val_r2.append(val_r2)

        if val_r2 > self.best_score:
            self.best_score = val_r2
            self.best_model = joblib.load(joblib.dump(model, 'temp_model.pkl')[0])
            self.best_iter = iteration

    def get_elapsed(self):
        return time.time() - self.start_time

# Initialize and train model
print("\nTraining final model...")
model = GradientBoostingRegressor(**best_params, random_state=SEED)
monitor = TrainingMonitor()

for i in range(best_params['n_estimators']):
    if i == 0:
        model.fit(X_train, y_train)
    else:
        # Incremental training for single estimator
        model.set_params(n_estimators=model.n_estimators + 1)
        model.fit(X_train, y_train)

    monitor.record(model, X_train, y_train, X_val, y_val, i)

    if (i + 1) % 10 == 0 or (i + 1) == best_params['n_estimators']:
        print(f"Iter {i+1:3d}/{best_params['n_estimators']} | "
              f"Train R²: {monitor.train_r2[-1]:.4f} | "
              f"Val R²: {monitor.val_r2[-1]:.4f} | "
              f"Time: {monitor.get_elapsed():.1f}s")

# Clean up
if os.path.exists('temp_model.pkl'):
    os.remove('temp_model.pkl')

# Final evaluation
best_model = monitor.best_model
y_pred = best_model.predict(X_test)
r2_score_test = r2_score(y_test, y_pred)
mse_score_test = mean_squared_error(y_test, y_pred)

print("\n=== FINAL RESULTS ===")
print(f"Best model at iteration {monitor.best_iter + 1}")
print(f"\nTest Scores:")
print(f"R²: {r2_score_test:.4f}")
print(f"MSE: {mse_score_test:.6f}")

# Feature importance analysis
print("\n=== FEATURE IMPORTANCE ===")
feature_names = X.columns
importances = best_model.feature_importances_
print("Feature importances:")
for idx in np.argsort(importances)[::-1]:
    print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

# Visualization - Feature Importance
plt.figure(figsize=(8, 6))
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[idx] for idx in indices], rotation=45)
plt.ylabel('Importance Score')
plt.title('Feature Importance - final_energy')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.show()


# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, s=20)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
plt.xlabel('Actual final_energy')
plt.ylabel('Predicted final_energy')
plt.title(f'Actual vs Predicted\nR²: {r2_score_test:.4f}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save final model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_model_path = os.path.join(output_dir, f'final_model_{timestamp}.pkl')
joblib.dump(best_model, final_model_path)
print(f"\nModel saved to: {final_model_path}")

# Save training history
history = {
    'train_loss': monitor.train_losses,
    'val_loss': monitor.val_losses,
    'train_r2': monitor.train_r2,
    'val_r2': monitor.val_r2,
    'best_val_score': monitor.best_score,
    'best_iteration': monitor.best_iter,
    'test_indices': test_indices,
    'feature_importances': importances,
    'best_hyperparameters': best_params,
    'optuna_study': study.trials_dataframe()
}
history_path = os.path.join(output_dir, f'training_history_{timestamp}.pkl')
joblib.dump(history, history_path)
print(f"Training history saved to: {history_path}")
