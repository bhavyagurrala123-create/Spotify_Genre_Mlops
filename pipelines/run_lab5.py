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
print("LAB 5 - COMPLETE MLOPS PIPELINE")
print("SPOTIFY GENRE CLASSIFICATION")
print("=" * 60)

# STEP 1 - Data preprocessing
run_script("src/preprocess.py")

# STEP 2 - MLflow model training and experiment tracking
run_script("src/train_mlflow.py")

# STEP 3 - Model evaluation
run_script("src/evaluate.py")

print("\n" + "=" * 60)
print("[SUCCESS] LAB 5 PIPELINE FULLY EXECUTED!")
print("=" * 60)