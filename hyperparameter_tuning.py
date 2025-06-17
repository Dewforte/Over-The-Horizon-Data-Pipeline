import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("updated_plasma_oth_dataset.csv")

# Features and target
input_features = ["Cloud Altitude", "Release Mass", "Cloud Age", "Diffusion Coefficient", "Transmitter Frequency", "Transmitter Power", 'Transmitter Gain', 'Receiver Gain',
                "Distance", "Windspeed", "Kp Index", "Solar Angle", "Doppler Shift"]

X = df[input_features]
y = df["Signal to Noise Ratio"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 🔍 Random Forest Tuning
# ---------------------------
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_rf = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid_rf,
    cv=5,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

print("\n🔧 Best Random Forest Parameters:")
print(grid_rf.best_params_)

# Evaluate on test set
y_pred_rf = best_rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"\n📊 Random Forest Tuned Performance (SNR):")
print(f"  RMSE: {rmse_rf:.4f}")
print(f"  MAE: {mae_rf:.4f}")
print(f"  R²: {r2_rf:.4f}")

# ---------------------------
# 🔍 XGBoost Tuning
# ---------------------------
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 10],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

grid_xgb = GridSearchCV(
    estimator=XGBRegressor(random_state=42, verbosity=0),
    param_grid=param_grid_xgb,
    cv=5,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_

print("\n🔧 Best XGBoost Parameters:")
print(grid_xgb.best_params_)

# Evaluate on test set
y_pred_xgb = best_xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"\n📊 XGBoost Tuned Performance (SNR):")
print(f"  RMSE: {rmse_xgb:.4f}")
print(f"  MAE: {mae_xgb:.4f}")
print(f"  R²: {r2_xgb:.4f}")

# ---------------------------
# Plot Results
# ---------------------------
def plot_results(y_test, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual SNR (dB)")
    plt.ylabel("Predicted SNR (dB)")
    plt.title(f"{model_name} — Tuned Prediction")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_results(y_test, y_pred_rf, "Random Forest (Tuned)")
plot_results(y_test, y_pred_xgb, "XGBoost (Tuned)")
