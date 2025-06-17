import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("updated_plasma_oth_dataset.csv")

# Define input and output features
input_features = ["Cloud Altitude", "Release Mass", "Cloud Age", "Diffusion Coefficient", "Transmitter Frequency", "Transmitter Power", 'Transmitter Gain', 'Receiver Gain',
                "Distance", "Windspeed", "Kp Index", "Solar Angle", "Doppler Shift"]

output_targets = ["Signal to Noise Ratio"]

X = df[input_features]
y = df[output_targets]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    'Linear Regression': MultiOutputRegressor(LinearRegression()),
    'Random Forest': MultiOutputRegressor(RandomForestRegressor(random_state=42)),
    'XGBoost': MultiOutputRegressor(XGBRegressor(random_state=42, verbosity=0))
}

# Evaluation function
def evaluate_model(name, model, X_test, y_test, y_pred):
    print(f"\n📊 {name} Performance:")
    for i, target in enumerate(output_targets):
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"  {target} → RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Train and evaluate all models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(name, model, X_test, y_test, y_pred)

    # Plot predicted vs. actual
    for i, target in enumerate(output_targets):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.6)
        plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
                 [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
                 'r--')
        plt.title(f"{name} — Predicted vs Actual ({target})")
        plt.xlabel(f"Actual {target}")
        plt.ylabel(f"Predicted {target}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
