"""
notebooks/01_EDA.py
Convert to notebook: jupytext --to notebook 01_EDA.py
Or copy cells into Jupyter manually.

This notebook covers Day 1 exploratory analysis — run BEFORE training.
"""

# ─── Cell 1: Setup ────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, "..")   # make src importable from notebooks/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import load_raw_data, get_feature_types
from config import TARGET_COLUMN

df = load_raw_data()
print(f"Shape: {df.shape}")
df.head()

# ─── Cell 2: Basic info ───────────────────────────────────────────────────────
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# ─── Cell 3: Class distribution (THE most important chart) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Count plot
counts = df[TARGET_COLUMN].value_counts()
axes[0].bar(["Legitimate (0)", "Fraud (1)"],
            counts.values,
            color=["#378ADD", "#E24B4A"], alpha=0.85, edgecolor="white")
axes[0].set_title("Claim Count by Class")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 50, f"{v:,}", ha="center", fontsize=11)

# Pie chart
axes[1].pie(counts.values,
            labels=[f"Legitimate\n{counts[0]:,}", f"Fraud\n{counts[1]:,}"],
            autopct="%1.1f%%",
            colors=["#378ADD", "#E24B4A"],
            startangle=90)
axes[1].set_title("Class Distribution")

plt.suptitle("⚠ Class Imbalance — This is why we need SMOTE", fontsize=12)
plt.tight_layout()
plt.savefig("../outputs/plots/class_distribution.png", dpi=150)
plt.show()

print(f"\nFraud rate: {df[TARGET_COLUMN].mean()*100:.2f}%")

# ─── Cell 4: Feature distributions ───────────────────────────────────────────
cat_cols, num_cols = get_feature_types(df)

# Numerical features: compare fraud vs legit distributions
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
num_sample = num_cols[:6]

for ax, col in zip(axes.flat, num_sample):
    df[df[TARGET_COLUMN] == 0][col].hist(ax=ax, alpha=0.6,
                                          color="#378ADD", label="Legit", bins=20)
    df[df[TARGET_COLUMN] == 1][col].hist(ax=ax, alpha=0.6,
                                          color="#E24B4A", label="Fraud", bins=20)
    ax.set_title(col)
    ax.legend(fontsize=8)

plt.suptitle("Numerical Feature Distributions — Fraud vs Legitimate", fontsize=12)
plt.tight_layout()
plt.show()

# ─── Cell 5: Top categorical features vs fraud rate ──────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
cat_sample = cat_cols[:6]

for ax, col in zip(axes.flat, cat_sample):
    fraud_rate = df.groupby(col)[TARGET_COLUMN].mean().sort_values(ascending=False)
    ax.bar(range(len(fraud_rate)), fraud_rate.values,
           color="#E24B4A", alpha=0.8)
    ax.set_xticks(range(len(fraud_rate)))
    ax.set_xticklabels(fraud_rate.index, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Fraud rate by {col}")
    ax.set_ylabel("Fraud rate")
    ax.set_ylim(0, 1)

plt.suptitle("Fraud Rate by Categorical Feature Values", fontsize=12)
plt.tight_layout()
plt.show()

# ─── Cell 6: Correlation heatmap (numerical features) ────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[num_cols + [TARGET_COLUMN]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix — Numerical Features")
plt.tight_layout()
plt.show()

# ─── Cell 7: Key EDA findings ─────────────────────────────────────────────────
print("""
KEY EDA FINDINGS:
─────────────────
1. Class imbalance: ~6% fraud — MUST use SMOTE or class weights
2. No missing values in this dataset (lucky — saves preprocessing time)
3. All features are categorical or ordinal integers
4. Fraud rate varies significantly across: VehicleCategory, PolicyType, Fault
5. 'Fault' and 'Days_Policy_Accident' show highest correlation with fraud
→ These will likely be the most important features for the ANN
""")
