"""Data registration: validate the dataset schema and print a summary.

Run in CI as: python tourism_project/model_building/data_register.py
"""
import os
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

# The business columns we expect the dataset to contain.
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

def main():
    if not os.path.exists(DATA_PATH):
        sys.exit(f"ERROR: dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Validate that every expected column is present.
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: dataset is missing expected columns: {missing}")

    print("Dataset validation passed. Expected columns are present.")
    print(f"Shape           : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns         : {list(df.columns)}")
    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nFirst rows:")
    print(df.head())

if __name__ == "__main__":
    main()
