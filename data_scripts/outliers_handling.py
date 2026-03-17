import pandas as pd
import numpy as np

# Load cleaned data from Script 1
df = pd.read_csv("outputs/cleaned.csv")
print("Loaded shape:", df.shape)

features = [
    "RevolvingUtilization", "Age", "DebtRatio", "MonthlyIncome",
    "OpenCreditLines", "RealEstateLines", "Dependents",
    "Times30_59Late", "Times60_89Late", "Times90Late"
]

# IQR method to detect outliers 
print("\nOutlier Detection (IQR Method):")
print(f"  {'Feature':<28} {'Below Q1-1.5IQR':>15} {'Above Q3+1.5IQR':>15} {'Total':>8}")
print("  " + "-" * 70)

for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = (df[col] < Q1 - 1.5 * IQR).sum()
    upper = (df[col] > Q3 + 1.5 * IQR).sum()
    print(f"  {col:<28} {lower:>15,} {upper:>15,} {lower + upper:>8,}")

# Check suspicious values 
print("\nSuspicious values found:")

# Late payment columns have sentinel codes 96 and 98
for col in ["Times30_59Late", "Times60_89Late", "Times90Late"]:
    count = (df[col] >= 96).sum()
    print(f"  {col}: {count} rows with value >= 96 (sentinel codes, not real counts)")

# RevolvingUtilization should be 0-1 normally
extreme = (df["RevolvingUtilization"] > 1).sum()
print(f"  RevolvingUtilization: {extreme:,} rows > 1 (exceeds credit limit or data error)")

# DebtRatio extreme values
extreme_dr = (df["DebtRatio"] > 10).sum()
print(f"  DebtRatio: {extreme_dr:,} rows > 10 (likely data errors)")

# MonthlyIncome = 0
zero_inc = (df["MonthlyIncome"] == 0).sum()
print(f"  MonthlyIncome: {zero_inc:,} rows = 0")

# Min/Max for each feature 
print(f"\nCurrent Min/Max values:")
print(f"  {'Feature':<28} {'Min':>12} {'Max':>12}")
print("  " + "-" * 55)
for col in features:
    print(f"  {col:<28} {df[col].min():>12.2f} {df[col].max():>12.2f}")

# 1. Cap sentinel values in late payment columns 
# Values 96 and 98 are codes, not real counts. Cap at 20.
late_cols = ["Times30_59Late", "Times60_89Late", "Times90Late"]

print("\n1. Late payment sentinel values (96, 98):")
for col in late_cols:
    count = (df[col] >= 96).sum()
    print(f"   {col}: {count} rows >= 96 -> capped at 20")
    df.loc[df[col] >= 96, col] = 20

# 2. Cap RevolvingUtilization at 99th percentile 
cap = df["RevolvingUtilization"].quantile(0.99)
count = (df["RevolvingUtilization"] > cap).sum()
df["RevolvingUtilization"] = df["RevolvingUtilization"].clip(upper=cap)
print(f"\n2. RevolvingUtilization: {count:,} rows capped at p99 ({cap:.4f})")

# 3. Cap DebtRatio at 99th percentile 
cap = df["DebtRatio"].quantile(0.99)
count = (df["DebtRatio"] > cap).sum()
df["DebtRatio"] = df["DebtRatio"].clip(upper=cap)
print(f"3. DebtRatio: {count:,} rows capped at p99 ({cap:.2f})")

# 4. Cap MonthlyIncome at 99th percentile 
cap = df["MonthlyIncome"].quantile(0.99)
count = (df["MonthlyIncome"] > cap).sum()
df["MonthlyIncome"] = df["MonthlyIncome"].clip(upper=cap)
print(f"4. MonthlyIncome: {count:,} rows capped at p99 ({cap:.0f})")

# 5. Cap OpenCreditLines at 99th percentile 
cap = df["OpenCreditLines"].quantile(0.99)
count = (df["OpenCreditLines"] > cap).sum()
df["OpenCreditLines"] = df["OpenCreditLines"].clip(upper=cap)
print(f"5. OpenCreditLines: {count:,} rows capped at p99 ({cap:.0f})")

# 6. Cap RealEstateLines at 99th percentile 
cap = df["RealEstateLines"].quantile(0.99)
count = (df["RealEstateLines"] > cap).sum()
df["RealEstateLines"] = df["RealEstateLines"].clip(upper=cap)
print(f"6. RealEstateLines: {count:,} rows capped at p99 ({cap:.0f})")

# Summary after treatment 
features = [
    "RevolvingUtilization", "Age", "DebtRatio", "MonthlyIncome",
    "OpenCreditLines", "RealEstateLines", "Dependents",
    "Times30_59Late", "Times60_89Late", "Times90Late"
]

print(f"\nAfter treatment:")
print(f"  {'Feature':<28} {'Min':>10} {'Max':>10} {'Median':>10}")
print("  " + "-" * 60)
for col in features:
    print(f"  {col:<28} {df[col].min():>10.2f} {df[col].max():>10.2f} {df[col].median():>10.2f}")

# Save
df.to_csv("outputs/outliers_handled.csv", index=False)
print(f"\nSaved: outputs/outliers_handled.csv ({df.shape[0]:,} rows)")