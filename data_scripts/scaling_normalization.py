import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle

# Load data
#df = pd.read_csv("outputs/outliers_handled.csv")
df = pd.read_csv("../outputs/outliers_handled.csv")
print("Loaded shape:", df.shape)

# Encoding check 
print("\nENCODING")
print("Data types:")
print(df.dtypes)

categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"\nCategorical columns: {categorical_cols if categorical_cols else 'None'}\n")
#print(f"\nCategorical columns: {categorical_cols if categorical_cols else 'None\n'}")
print(f"Numerical columns  : {numerical_cols}")

print("\nAll features are numerical — no categorical encoding needed.")
print("Target is already binary (0/1).")

# Separate features and target
feature_cols = [col for col in df.columns if col != "Target"]
X = df[feature_cols]
y = df["Target"]

print(f"\nFeatures: {len(feature_cols)}")

# Compare three scalers 
print("\nSCALER COMPARISON (on MonthlyIncome):")

col = "MonthlyIncome"
print(f"\n  Before: Min={X[col].min():.0f}, Max={X[col].max():.0f}, "
      f"Mean={X[col].mean():.0f}, Std={X[col].std():.0f}")

print(f"\n  {'Scaler':<20} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
print("  " + "-" * 57)

for name, scaler in [("StandardScaler", StandardScaler()),
                      ("MinMaxScaler", MinMaxScaler()),
                      ("RobustScaler", RobustScaler())]:
    scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    print(f"  {name:<20} {scaled[col].min():>8.2f} {scaled[col].max():>8.2f} "
          f"{scaled[col].mean():>8.2f} {scaled[col].std():>8.2f}")

# Applying RobustScaler 
print("\nUsing RobustScaler (uses median/IQR, handles remaining skew better)")

scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Before vs after 
print(f"\nBefore vs After:")
print(f"  {'Feature':<28} {'Before Min':>10} {'Before Max':>10} {'After Min':>10} {'After Max':>10}")
print("  " + "-" * 72)
for col in feature_cols:
    print(f"  {col:<28} {X[col].min():>10.2f} {X[col].max():>10.2f} "
          f"{X_scaled[col].min():>10.2f} {X_scaled[col].max():>10.2f}")

# Save 
df_scaled = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
#df_scaled.to_csv("outputs/scaled_dataset.csv", index=False)
df_scaled.to_csv("../outputs/scaled_dataset.csv", index=False)
print(f"\nSaved: outputs/scaled_dataset.csv ({df_scaled.shape[0]:,} rows)")

# Saving scaler for later use on test set (cs-test.csv)
#with open("outputs/robust_scaler.pkl", "wb") as f:
with open("../outputs/robust_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved: outputs/robust_scaler.pkl")