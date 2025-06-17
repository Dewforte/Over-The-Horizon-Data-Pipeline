import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("updated_plasma_oth_dataset.csv")

# Optional: Sample subset for cleaner plots (e.g., pairplot)
sample_df = df.sample(n=300, random_state=42)

# 1. Histograms of key features
key_features = ["Cloud Altitude", "Release Mass", "Cloud Age", "Diffusion Coefficient", "Transmitter Frequency", "Transmitter Power",
                "Distance", "Windspeed", "Signal to Noise Ratio", "Delay Spread", "Kp Index", "Solar Angle"]


df[key_features].hist(bins=30, figsize=(15, 10), grid=False)
# plt.suptitle("Feature Distributions (Histograms)", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# 2. Pairplot to inspect trends and noise patterns
sns.pairplot(sample_df[["Release Mass", 'Electron Density', 'Critical Frequency', 'Signal to Noise Ratio', 'Delay Spread']])
# plt.suptitle("Pairwise Relationships (Sampled Subset)", y=1.02)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# 3. Clear-sky check: release_mass ≈ 0
clear_sky_df = df[df['Release Mass'] < 1e-3]
print("✅ Clear-Sky Rows:", len(clear_sky_df))

if not clear_sky_df.empty:
    print("\nClear-Sky Summary:")
    print(clear_sky_df[['Electron Density', 'Critical Frequency', 'Received Power', 'Signal to Noise Ratio', 'Bit Error Rate']].describe())
else:
    print("⚠️ No clear-sky (release_mass ≈ 0) rows found.")

# 4. Correlation heatmap
plt.figure(figsize=(12, 10))
corr = df.corr(numeric_only=True).round(2)
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", square=True, cbar_kws={"shrink": 0.7})
# plt.title("Correlation Heatmap of All Features")
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.show()
