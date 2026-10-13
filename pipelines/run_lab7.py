
import os
import sys
import json
import traceback
from datetime import datetime

# Set project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    log = {
        "lab": "Lab 7 - Spotify Automated Pipeline",
        "started_at": datetime.now().isoformat(),
        "stages": []
    }

    try:
        print("\n========== LAB 7: SPOTIFY PIPELINE ==========")

        print("\nSTEP 1: PREPROCESSING")
        preprocess_data()
        log["stages"].append("Preprocessing: SUCCESS")

        print("\nSTEP 2: MODEL TRAINING")
        train_model()
        log["stages"].append("Training: SUCCESS")

        print("\nSTEP 3: MODEL EVALUATION")
        evaluate_model()
        log["stages"].append("Evaluation: SUCCESS")

        # Copy evaluation metrics into a Lab 7 report
        with open("outputs/metrics.json", "r") as f:
            metrics = json.load(f)

        report = {
            "lab": "Lab 7 - Spotify Automated Pipeline",
            "status": "SUCCESS",
            "completed_at": datetime.now().isoformat(),
            "metrics": metrics,
            "stages": log["stages"]
        }

        with open("reports/lab7_report.json", "w") as f:
            json.dump(report, f, indent=4)

        log["status"] = "SUCCESS"
        log["finished_at"] = datetime.now().isoformat()

        print("\nLAB 7 COMPLETED SUCCESSFULLY!")
        print("Report saved: reports/lab7_report.json")

    except Exception as e:
        log["status"] = "FAILED"
        log["error"] = str(e)
        log["traceback"] = traceback.format_exc()
        log["finished_at"] = datetime.now().isoformat()

        print("\nLAB 7 FAILED:", e)
        raise

    finally:
        with open("logs/lab7_pipeline_log.json", "w") as f:
            json.dump(log, f, indent=4)


if __name__ == "__main__":
    main()
