import mlflow
from mlflow.tracking import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Spotify_Genre_Classification"
REGISTERED_MODEL_NAME = "SpotifyGenreClassifier"


def register_best_model():

    print("=" * 60)
    print("LAB 6 - MLFLOW MODEL REGISTRY")
    print("=" * 60)

    # Connect to MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Find experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print("[ERROR] Experiment not found.")
        return

    print("[INFO] Experiment found:", EXPERIMENT_NAME)

    # Find best run using Macro F1
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.macro_f1 DESC"],
        max_results=1
    )

    if not runs:
        print("[ERROR] No MLflow runs found.")
        return

    best_run = runs[0]

    run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics.get("macro_f1")

    print("\n[INFO] Best Run ID:", run_id)
    print("[INFO] Best Macro F1:", best_f1)

    # Model location inside the MLflow run
    model_uri = f"runs:/{run_id}/random_forest_model"

    print("\n[INFO] Registering model...")
    print("[INFO] Model URI:", model_uri)

    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    print("\n" + "=" * 60)
    print("[SUCCESS] MODEL REGISTERED")
    print("=" * 60)

    print("Model Name   :", REGISTERED_MODEL_NAME)
    print("Model Version:", result.version)
    print("Run ID       :", run_id)


if __name__ == "__main__":
    register_best_model()