import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def train_model():

    print("=" * 60)
    print("STARTING SPOTIFY GENRE MODEL TRAINING")
    print("=" * 60)

    # Create models folder if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --------------------------------------------------
    # 1. Load training data
    # --------------------------------------------------

    print("\n[INFO] Loading training data...")

    X_train = pd.read_csv(
        f"{PROCESSED_DIR}/X_train.csv"
    )

    y_train = pd.read_csv(
        f"{PROCESSED_DIR}/y_train.csv"
    ).squeeze("columns")

    print("[INFO] X_train shape:", X_train.shape)
    print("[INFO] y_train shape:", y_train.shape)

    # --------------------------------------------------
    # 2. Create Random Forest Model
    # --------------------------------------------------

    print("\n[INFO] Creating Random Forest model...")

    model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
    

    # --------------------------------------------------
    # 3. Train model
    # --------------------------------------------------

    print("[INFO] Training model...")

    model.fit(
        X_train,
        y_train
    )

    print("[SUCCESS] Model training completed.")

    # --------------------------------------------------
    # 4. Save trained model
    # --------------------------------------------------

    model_path = f"{MODEL_DIR}/random_forest.pkl"

    joblib.dump(
        model,
        model_path
    )

    print(
        "[SUCCESS] Model saved:",
        model_path
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    train_model()