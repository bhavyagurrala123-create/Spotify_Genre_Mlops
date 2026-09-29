import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify_Genre_Classification")


def train_with_mlflow():

    print("=" * 60)
    print("LAB 4 - MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)

    # Load training data
    print("\n[INFO] Loading training data...")

    X_train = pd.read_csv(
        f"{PROCESSED_DIR}/X_train.csv"
    )

    y_train = pd.read_csv(
        f"{PROCESSED_DIR}/y_train.csv"
    ).squeeze("columns")

    # Load test data
    X_test = pd.read_csv(
        f"{PROCESSED_DIR}/X_test.csv"
    )

    y_test = pd.read_csv(
        f"{PROCESSED_DIR}/y_test.csv"
    ).squeeze("columns")

    print("[INFO] X_train shape:", X_train.shape)
    print("[INFO] X_test shape :", X_test.shape)

    # Memory-safe Random Forest parameters
    params = {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 2,
        "random_state": 42,
        "n_jobs": -1
    }

    print("\n[INFO] Starting MLflow run...")

    with mlflow.start_run(run_name="RandomForest"):

        # Log parameters
        mlflow.log_params(params)

        print("[INFO] Training Random Forest...")

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        print("[INFO] Training completed.")

        # Predictions
        print("[INFO] Making predictions...")

        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("macro_f1", f1)

        print("\n" + "=" * 60)
        print("MLFLOW MODEL RESULTS")
        print("=" * 60)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"Macro F1  : {f1:.4f}")

        # Log model
        print("\n[INFO] Logging model to MLflow...")

        mlflow.sklearn.log_model(
    sk_model=model,
    name="random_forest_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

        print("[SUCCESS] Model logged to MLflow.")

    print("\n" + "=" * 60)
    print("[SUCCESS] LAB 4 MLFLOW TRACKING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    train_with_mlflow()