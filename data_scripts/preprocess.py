import pandas as pd

# Loading the dataset
df = pd.read_csv("data/cs-training.csv")

# Dropping the unnamed index column 
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Original shape:", df.shape)

# Missing values
print("\nMissing values before handling:")
print(df.isnull().sum())

# Checking mean, median, and mode for columns with missing values
missing_cols = df.columns[df.isnull().any()]

print("\nMean, Median, and Mode for columns with missing values:")
for col in missing_cols:
    print(f"\nColumn: {col}")
    print(f"Missing values: {df[col].isnull().sum()}")
    print(f"Mean: {df[col].mean()}")
    print(f"Median: {df[col].median()}")
    print(f"Mode: {df[col].mode().tolist()}")

# Duplicate rows 
print("\nDuplicate rows before handling:", df.duplicated().sum())

# Filling missing values with median
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())

# Removing duplicate rows
df = df.drop_duplicates()

print("\nShape after removing duplicates:", df.shape)

# Missing values after handling
print("\nMissing values after handling:")
print(df.isnull().sum())

# Duplicate rows after handling
print("\nDuplicate rows after handling:", df.duplicated().sum())

# Target distribution
print("\nTarget distribution:")
print(df["SeriousDlqin2yrs"].value_counts())

# Saving cleaned dataset
df.to_csv("data/cleaned_cs_training.csv", index=False)

print("\nCleaned dataset saved as data/cleaned_cs_training.csv")

