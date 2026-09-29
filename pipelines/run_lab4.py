import subprocess
import sys


def run_script(script):
    print("\n" + "=" * 60)
    print(f"[INFO] Executing {script}")
    print("=" * 60)

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed.")
        sys.exit(result.returncode)

    print(f"\n[SUCCESS] {script} completed.")


print("\n" + "=" * 60)
print("LAB 4 - MLFLOW EXPERIMENT TRACKING PIPELINE")
print("SPOTIFY GENRE CLASSIFICATION")
print("=" * 60)

# Step 1: Preprocessing
run_script("src/preprocess.py")

# Step 2: Train model and track with MLflow
run_script("src/train_mlflow.py")

print("\n" + "=" * 60)
print("[SUCCESS] LAB 4 PIPELINE FULLY EXECUTED!")
print("=" * 60)