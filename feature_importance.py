import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming best_rf and best_xgb are already fitted
# and X_train columns are available from earlier

features = X_train.columns

# --- Random Forest Feature Importance ---
rf_importance = best_rf.feature_importances_
rf_sorted_idx = np.argsort(rf_importance)[::-1]

plt.figure(figsize=(8, 6))
plt.barh(range(len(features)), rf_importance[rf_sorted_idx], align='center')
plt.yticks(range(len(features)), [features[i] for i in rf_sorted_idx])
plt.gca().invert_yaxis()
# plt.title("Random Forest Feature Importance (SNR Prediction)")
plt.xlabel("Importance")
# plt.grid(True)
plt.tight_layout()
plt.show()

# --- XGBoost Feature Importance ---
xgb_importance = best_xgb.feature_importances_
xgb_sorted_idx = np.argsort(xgb_importance)[::-1]

plt.figure(figsize=(8, 6))
plt.barh(range(len(features)), xgb_importance[xgb_sorted_idx], align='center', color='darkorange')
plt.yticks(range(len(features)), [features[i] for i in xgb_sorted_idx])
plt.gca().invert_yaxis()
# plt.title("XGBoost Feature Importance (SNR Prediction)")
plt.xlabel("Importance")
# plt.grid(True)
plt.tight_layout()
plt.show()
