import matplotlib.pyplot as plt
import seaborn as sns

# Residuals
residuals_rf = y_test - y_pred_rf
residuals_xgb = y_test - y_pred_xgb

# ---------- Residual Plots ----------
plt.figure(figsize=(6, 5))
plt.scatter(y_test, residuals_rf, alpha=0.6, label="RF Residuals")
plt.axhline(0, color='r', linestyle='--')
plt.title("Residual Plot — Random Forest (SNR)")
plt.xlabel("Actual SNR (dB)")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(y_test, residuals_xgb, alpha=0.6, label="XGB Residuals", color='darkorange')
plt.axhline(0, color='r', linestyle='--')
plt.title("Residual Plot — XGBoost (SNR)")
plt.xlabel("Actual SNR (dB)")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Error Histogram ----------
plt.figure(figsize=(6, 5))
sns.histplot(residuals_rf, kde=True, bins=30, color='steelblue')
plt.title("Error Distribution — Random Forest")
plt.xlabel("Prediction Error (dB)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.histplot(residuals_xgb, kde=True, bins=30, color='darkorange')
plt.title("Error Distribution — XGBoost")
plt.xlabel("Prediction Error (dB)")
plt.tight_layout()
plt.show()
