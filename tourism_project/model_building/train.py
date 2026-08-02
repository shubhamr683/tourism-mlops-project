"""Model training with hyperparameter tuning and MLflow tracking.

Run in CI as: python tourism_project/model_building/train.py
Loads the train/test splits, tunes an XGBoost classifier inside a
preprocessing pipeline, logs everything to MLflow, and saves the best
model into tourism_project/deployment/ so the workflow can commit it.
"""
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model.joblib")

CATEGORICAL = ["TypeofContact", "Occupation", "Gender", "ProductPitched",
               "MaritalStatus", "Designation"]

def main():
    # Splits produced by prep.py (downloaded from the workflow artifact).
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")

    numeric = [c for c in Xtrain.columns if c not in CATEGORICAL]

    # OneHotEncode categoricals, scale numerics. handle_unknown keeps the app
    # robust to categories it has not seen before.
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        (StandardScaler(), numeric),
    )

    # Weight the positive class to counter the imbalance (~4:1).
    spw = (ytrain == 0).sum() / max((ytrain == 1).sum(), 1)
    model = XGBClassifier(
        random_state=42, eval_metric="logloss", scale_pos_weight=spw
    )

    pipe = make_pipeline(preprocessor, model)

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", n_jobs=-1)

    mlflow.set_experiment("tourism-wellness-package")
    with mlflow.start_run():
        grid.fit(Xtrain, ytrain)
        best = grid.best_estimator_

        preds = best.predict(Xtest)
        proba = best.predict_proba(Xtest)[:, 1]

        metrics = {
            "accuracy": accuracy_score(ytest, preds),
            "f1": f1_score(ytest, preds),
            "roc_auc": roc_auc_score(ytest, proba),
            "cv_best_f1": grid.best_score_,
        }

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)

        print("Best parameters:", grid.best_params_)
        print("Test metrics    :", metrics)
        print("\nClassification report:\n", classification_report(ytest, preds))

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best, MODEL_PATH)
        # Log the model file to MLflow as an artifact. (Kept resilient: newer
        # MLflow versions can reject XGBoost via the sklearn flavor, but the
        # joblib file above is what the pipeline commits and the app loads.)
        try:
            mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        except Exception as e:
            print(f"MLflow artifact logging skipped: {e}")
        print(f"Saved best model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
