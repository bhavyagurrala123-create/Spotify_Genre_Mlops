import subprocess
import sys


def run_script(script):

    print("\n" + "=" * 60)
    print(f"[INFO] Executing {script}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script]
    )

    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed.")
        sys.exit(result.returncode)

    print(f"[SUCCESS] {script} completed.")


print("\n" + "=" * 60)
print("LAB 3 - BASELINE ML PIPELINE")
print("SPOTIFY GENRE CLASSIFICATION")
print("=" * 60)

run_script("src/preprocess.py")
run_script("src/train.py")
run_script("src/evaluate.py")

print("\n" + "=" * 60)
print("[SUCCESS] LAB 3 PIPELINE FULLY EXECUTED!")
print("=" * 60)