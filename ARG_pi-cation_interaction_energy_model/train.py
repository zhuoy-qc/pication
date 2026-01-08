import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
import xgboost as xgb
import joblib
import optuna
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """Engineer additional features from geometric parameters"""
    df = df.rename(columns={'offset': 'delta_x', 'rz': 'delta_z', 'dihedral': 'dihedral_angle'})
    
    # Basic features
    df['inv_distance'] = 1 / df['distance']
    df['distance_sq'] = df['distance'] ** 2
    
    # Trigonometric features
    df['sin_dihedral'] = np.sin(np.radians(df['dihedral_angle']))
    df['cos_dihedral'] = np.cos(np.radians(df['dihedral_angle']))
    df['tan_dihedral'] = np.tan(np.radians(df['dihedral_angle']))
    
    # Normalized features
    df['delta_z_norm'] = df['delta_z'] / df['distance']
    df['delta_x_norm'] = df['delta_x'] / df['distance']
    
    # Additional ratios
    df['delta_x_delta_z_ratio'] = df['delta_x'] / df['delta_z']
    df['distance_delta_z_ratio'] = df['distance'] / df['delta_z']
    df['distance_delta_x_ratio'] = df['distance'] / df['delta_x']
    
    # Log transformations
    df['log_distance'] = np.log(df['distance'])
    df['log_abs_delta_x'] = np.log(np.abs(df['delta_x']) + 1e-8)
    df['log_abs_delta_z'] = np.log(np.abs(df['delta_z']) + 1e-8)
    
    # Polynomial features
    df['distance_cubed'] = df['distance'] ** 3
    df['delta_x_squared'] = df['delta_x'] ** 2
    df['delta_z_squared'] = df['delta_z'] ** 2
    
    # Interaction features
    df['distance_delta_x'] = df['distance'] * df['delta_x']
    df['distance_delta_z'] = df['distance'] * df['delta_z']
    df['delta_x_delta_z'] = df['delta_x'] * df['delta_z']
    
    # Combined geometric features
    df['effective_distance'] = np.sqrt(df['distance']**2 + df['delta_x']**2 + df['delta_z']**2)
    df['planar_distance'] = np.sqrt(df['delta_x']**2 + df['delta_z']**2)
    
    # Handle potential division by zero
    for col in ['delta_x_delta_z_ratio', 'distance_delta_z_ratio', 'distance_delta_x_ratio', 'tan_dihedral']:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    return df.fillna(df.median())

def load_data(file_path):
    """Load and prepare the dataset with extended features"""
    df = pd.read_csv(file_path)
    df = engineer_features(df)
    
    FEATURE_COLS = [
        'delta_z', 'distance', 'delta_x', 'dihedral_angle',
        'inv_distance', 'distance_sq', 'sin_dihedral', 'cos_dihedral',
        'tan_dihedral', 'delta_z_norm', 'delta_x_norm', 'delta_x_delta_z_ratio',
        'distance_delta_z_ratio', 'distance_delta_x_ratio', 'log_distance',
        'log_abs_delta_x', 'log_abs_delta_z', 'distance_cubed', 'delta_x_squared',
        'delta_z_squared', 'distance_delta_x', 'distance_delta_z', 'delta_x_delta_z',
        'effective_distance', 'planar_distance'
    ]
    
    FEATURE_COLS = [col for col in FEATURE_COLS if col in df.columns]
    X = df[FEATURE_COLS]
    y = df['energy']
    
    return X, y, FEATURE_COLS

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)

def train_model():
    """Train XGBoost model with 8:1:1 split and Optuna optimization"""
    # Load data
    X, y, feature_cols = load_data('combined_sorted.csv')
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target range: {y.min():.4f} to {y.max():.4f}")
    
    # 8:1:1 split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)
    
    print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
    
    # Feature selection using train+validation
    print("Feature selection...")
    selector = RFECV(xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, tree_method='hist'),
                     step=1, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    
    X_train_val = pd.concat([X_train_scaled, X_val_scaled], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    selector.fit(X_train_val, y_train_val)
    
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.support_[i]]
    print(f"Features selected: {selector.n_features_}")
    
    X_train_selected = X_train_scaled[selected_features]
    X_val_selected = X_val_scaled[selected_features]
    X_test_selected = X_test_scaled[selected_features]
    
    # Optuna optimization
    print("Hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_selected, y_train, X_val_selected, y_val), n_trials=100)
    
    best_params = study.best_params
    best_params.update({'random_state': 42, 'tree_method': 'hist'})
    print(f"Best params: {best_params}")
    
    # Train final model
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train_selected, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_selected)
    final_mae = mean_absolute_error(y_test, y_pred)
    final_mse = mean_squared_error(y_test, y_pred)
    final_r2 = r2_score(y_test, y_pred)
    
    print(f"Test MAE: {final_mae:.4f}")
    print(f"Test MSE: {final_mse:.4f}")
    print(f"Test R²: {final_r2:.4f}")
    
    # Save model
    model_package = {
        'model': model,
        'scaler': scaler,
        'selected_features': selected_features,
        'all_feature_cols': feature_cols,
        'test_mae': final_mae,
        'best_params': best_params,
        'created_date': pd.Timestamp.now()
    }
    
    joblib.dump(model_package, 'xgboost_model_optimized.pkl')
    print("Model saved as 'xgboost_model_optimized.pkl'")

if __name__ == "__main__":
    train_model()
