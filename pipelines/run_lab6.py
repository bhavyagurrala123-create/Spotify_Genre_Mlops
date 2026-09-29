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
print("LAB 6 - MODEL REGISTRY PIPELINE")
print("SPOTIFY GENRE CLASSIFICATION")
print("=" * 60)

# Register the best MLflow model
run_script("src/register_model.py")

print("\n" + "=" * 60)
print("[SUCCESS] LAB 6 PIPELINE FULLY EXECUTED!")
print("=" * 60)