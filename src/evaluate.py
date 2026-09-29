import os
import json
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"


def evaluate_model():

    print("=" * 60)
    print("STARTING SPOTIFY GENRE MODEL EVALUATION")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[INFO] Loading test data...")

    X_test = pd.read_csv(
        f"{PROCESSED_DIR}/X_test.csv"
    )

    y_test = pd.read_csv(
        f"{PROCESSED_DIR}/y_test.csv"
    ).squeeze("columns")

    print("[INFO] X_test shape:", X_test.shape)
    print("[INFO] y_test shape:", y_test.shape)

    print("\n[INFO] Loading trained model...")

    model = joblib.load(
        f"{MODEL_DIR}/random_forest.pkl"
    )

    print("[INFO] Making predictions...")

    y_pred = model.predict(X_test)

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

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Macro F1  : {f1:.4f}")

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(f1)
    }

    with open(
        f"{OUTPUT_DIR}/metrics.json",
        "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    predictions = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    predictions.to_csv(
        f"{OUTPUT_DIR}/predictions.csv",
        index=False
    )

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    with open(
        f"{OUTPUT_DIR}/classification_report.txt",
        "w"
    ) as file:
        file.write(report)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    pd.DataFrame(cm).to_csv(
        f"{OUTPUT_DIR}/confusion_matrix.csv",
        index=False
    )

    print("\n[SUCCESS] Evaluation completed.")

    print("\nFiles created:")
    print("outputs/metrics.json")
    print("outputs/predictions.csv")
    print("outputs/classification_report.txt")
    print("outputs/confusion_matrix.csv")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()