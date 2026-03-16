import pandas as pd

# Loading the dataset
df = pd.read_csv("data/cs-training.csv")

# Dropping the unnamed index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Original shape:", df.shape)
print("Original columns:", df.columns.tolist())

# Column Mapping

print("\n" + "=" * 60)
print("COLUMN MAPPING")
print("=" * 60)

column_map = {
    "SeriousDlqin2yrs":                     "Target",
    "RevolvingUtilizationOfUnsecuredLines":  "RevolvingUtilization",
    "age":                                   "Age",
    "NumberOfTime30-59DaysPastDueNotWorse":  "Times30_59Late",
    "DebtRatio":                             "DebtRatio",
    "MonthlyIncome":                         "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans":       "OpenCreditLines",
    "NumberOfTimes90DaysLate":               "Times90Late",
    "NumberRealEstateLoansOrLines":          "RealEstateLines",
    "NumberOfTime60-89DaysPastDueNotWorse":  "Times60_89Late",
    "NumberOfDependents":                    "Dependents",
}

df = df.rename(columns=column_map)

print("\nColumn name mapping applied:")
for old, new in column_map.items():
    print(f"  {old:<45} → {new}")

print(f"\nNew columns: {df.columns.tolist()}")

# Missing Values 

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("\nMissing values before handling:")
print(df.isnull().sum())

missing_cols = df.columns[df.isnull().any()]

print("\nMean, Median, and Mode for columns with missing values:")
for col in missing_cols:
    print(f"\n  Column: {col}")
    print(f"  Missing values : {df[col].isnull().sum()} ({df[col].isnull().mean() * 100:.2f}%)")
    print(f"  Mean           : {df[col].mean():.2f}")
    print(f"  Median         : {df[col].median():.2f}")
    print(f"  Mode           : {df[col].mode().tolist()}")
    print(f"  Std Dev        : {df[col].std():.2f}")
    print(f"  Skewness       : {df[col].skew():.2f}")

# Decision: Using median because both columns are right-skewed.
# Mean would be pulled up by high earners (MonthlyIncome) and large
# families (Dependents), giving misleading central values.
# Median is robust to these outliers.

print("\n--- Imputation Decision ---")
print("Using MEDIAN for both columns.")
print("Reason: Both features are right-skewed (skewness > 0),")
print("so the mean is inflated by outliers. Median is more robust.")

# Filling missing values with median
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].median())

# Handling the invalid case age = 0 
invalid_age = (df["Age"] == 0).sum()
if invalid_age > 0:
    print(f"\nFound {invalid_age} row(s) with Age = 0 (invalid)")
    print(f"Replacing with median Age: {df['Age'].median():.0f}")
    df.loc[df["Age"] == 0, "Age"] = df["Age"].median()

print("\nMissing values after handling:")
print(df.isnull().sum())

# Duplicate Rows 

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print("\nDuplicate rows before handling:", df.duplicated().sum())

# Removing duplicate rows
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)
print("Duplicate rows after handling:", df.duplicated().sum())

# Target Distribution 

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = df["Target"].value_counts()
print(f"\n  No Default (0): {target_counts[0]:>7,} ({target_counts[0]/len(df)*100:.1f}%)")
print(f"  Default    (1): {target_counts[1]:>7,} ({target_counts[1]/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: {target_counts[0] / target_counts[1]:.1f} : 1")

# Saving the preprocessed data

df.to_csv("outputs/cleaned.csv", index=False)
print(f"\nSaved: outputs/cleaned.csv ({df.shape[0]:,} rows x {df.shape[1]} cols)")
print("Script 1 complete.")