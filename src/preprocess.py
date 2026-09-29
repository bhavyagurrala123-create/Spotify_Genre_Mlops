import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_PATH = "data/raw/dataset.csv"
PROCESSED_DIR = "data/processed"
ARTIFACT_DIR = "artifacts"


# --------------------------------------------------
# Features used for Spotify Genre Prediction
# --------------------------------------------------

FEATURES = [
    "popularity",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature"
]


def preprocess_data():

    print("=" * 60)
    print("STARTING SPOTIFY DATA PREPROCESSING")
    print("=" * 60)

    # Create folders if they don't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    print("\n[INFO] Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("[INFO] Original dataset shape:", df.shape)


    # --------------------------------------------------
    # 2. Remove unnecessary columns
    # --------------------------------------------------

    columns_to_drop = [
        "Unnamed: 0",
        "track_id",
        "artists",
        "album_name",
        "track_name"
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    print("[INFO] Unnecessary columns removed.")


    # --------------------------------------------------
    # 3. Remove duplicate rows
    # --------------------------------------------------

    duplicates = df.duplicated().sum()

    print("[INFO] Duplicate rows found:", duplicates)

    df = df.drop_duplicates().reset_index(drop=True)


    # --------------------------------------------------
    # 4. Handle missing values
    # --------------------------------------------------

    missing = df.isnull().sum().sum()

    print("[INFO] Missing values found:", missing)

    df = df.dropna().reset_index(drop=True)


    # --------------------------------------------------
    # 5. Convert explicit column
    # --------------------------------------------------

    df["explicit"] = df["explicit"].astype(int)


    # --------------------------------------------------
    # 6. Encode target genre
    # --------------------------------------------------

    label_encoder = LabelEncoder()

    df["genre_encoded"] = label_encoder.fit_transform(
        df["track_genre"]
    )

    print(
        "[INFO] Number of genres:",
        len(label_encoder.classes_)
    )


    # --------------------------------------------------
    # 7. Separate features and target
    # --------------------------------------------------

    X = df[FEATURES]

    y = df["genre_encoded"]


    # --------------------------------------------------
    # 8. Train/Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("[INFO] Training samples:", len(X_train))
    print("[INFO] Testing samples :", len(X_test))


    # --------------------------------------------------
    # 9. Feature Scaling
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=FEATURES
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=FEATURES
    )


    # --------------------------------------------------
    # 10. Save processed datasets
    # --------------------------------------------------

    X_train.to_csv(
        f"{PROCESSED_DIR}/X_train.csv",
        index=False
    )

    X_test.to_csv(
        f"{PROCESSED_DIR}/X_test.csv",
        index=False
    )

    X_train_scaled.to_csv(
        f"{PROCESSED_DIR}/X_train_scaled.csv",
        index=False
    )

    X_test_scaled.to_csv(
        f"{PROCESSED_DIR}/X_test_scaled.csv",
        index=False
    )

    y_train.to_csv(
        f"{PROCESSED_DIR}/y_train.csv",
        index=False
    )

    y_test.to_csv(
        f"{PROCESSED_DIR}/y_test.csv",
        index=False
    )


    # --------------------------------------------------
    # 11. Save preprocessing objects
    # --------------------------------------------------

    joblib.dump(
        scaler,
        f"{ARTIFACT_DIR}/scaler.pkl"
    )

    joblib.dump(
        label_encoder,
        f"{ARTIFACT_DIR}/label_encoder.pkl"
    )


    print("\n" + "=" * 60)
    print("[SUCCESS] PREPROCESSING COMPLETED")
    print("=" * 60)

    print("\nFiles created in data/processed/")
    print("  X_train.csv")
    print("  X_test.csv")
    print("  X_train_scaled.csv")
    print("  X_test_scaled.csv")
    print("  y_train.csv")
    print("  y_test.csv")

    print("\nArtifacts created:")
    print("  artifacts/scaler.pkl")
    print("  artifacts/label_encoder.pkl")


if __name__ == "__main__":
    preprocess_data()