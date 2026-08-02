"""Data preparation: clean, split, and save train/test sets.

Run in CI as: python tourism_project/model_building/prep.py
Produces Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv in the working dir,
which the workflow uploads as an artifact for the training job.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

def main():
    df = pd.read_csv(DATA_PATH)

    # Drop columns that carry no predictive signal.
    #   'Unnamed: 0' -> leftover row index from the CSV export
    #   'CustomerID' -> unique identifier
    df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

    # Fix a known data-entry inconsistency in Gender.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Simple, robust missing-value handling (safe even if there are none).
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratify to preserve the class imbalance in both splits.
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")
    print(f"Train target distribution:\n{ytrain.value_counts()}")

if __name__ == "__main__":
    main()
