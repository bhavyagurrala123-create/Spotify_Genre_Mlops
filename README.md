# 🎵 Spotify Genre Prediction - MLOps Project

## Project Overview

This project implements an end-to-end Machine Learning and MLOps pipeline for predicting the genre of Spotify tracks using their audio features.

The project covers data preprocessing, model training, evaluation, MLflow experiment tracking, complete pipeline automation, and MLflow Model Registry.

---

## Dataset

The project uses the Spotify Tracks Dataset.

Target column:

```text
track_genre
```

The dataset contains approximately 114,000 Spotify tracks belonging to 114 different genres.

---

## Machine Learning Model

The main model used in the MLOps pipeline is:

```text
Random Forest Classifier
```

Configuration:

```text
n_estimators = 100
max_depth = 20
min_samples_split = 2
random_state = 42
n_jobs = -1
```

---

## Project Structure

```text
Spotify_Genre_MLOps/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_mlflow.py
│   └── register_model.py
│
├── pipelines/
│   ├── run_lab3.py
│   ├── run_lab4.py
│   ├── run_lab5.py
│   └── run_lab6.py
│
├── artifacts/
├── models/
├── outputs/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# MLOps Labs

## Lab 3 - Baseline ML Pipeline

Lab 3 performs:

1. Data preprocessing
2. Random Forest training
3. Model evaluation

Run:

```bash
python pipelines/run_lab3.py
```

---

## Lab 4 - MLflow Experiment Tracking

Lab 4 introduces MLflow for tracking:

- Model parameters
- Accuracy
- Precision
- Recall
- Macro F1 Score
- Trained model

Run:

```bash
python pipelines/run_lab4.py
```

Start MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

Experiment name:

```text
Spotify_Genre_Classification
```

---

## Lab 5 - Complete MLOps Pipeline

Lab 5 combines:

```text
Data Preprocessing
        ↓
MLflow Model Training
        ↓
Model Evaluation
```

Run:

```bash
python pipelines/run_lab5.py
```

---

## Lab 6 - MLflow Model Registry

Lab 6 identifies the best tracked run using Macro F1 score and registers its model with MLflow Model Registry.

Run:

```bash
python pipelines/run_lab6.py
```

Registered model:

```text
SpotifyGenreClassifier
```

---

## Evaluation Metrics

The project evaluates the model using:

- Accuracy
- Precision
- Recall
- Macro F1 Score
- Classification Report
- Confusion Matrix

Evaluation outputs are generated inside:

```text
outputs/
```

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Place the Spotify dataset at:

```text
data/raw/dataset.csv
```

Then execute the complete MLOps pipeline:

```bash
python pipelines/run_lab5.py
```

To register the tracked model:

```bash
python pipelines/run_lab6.py
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- MLflow
- Matplotlib
- VS Code
- Git
- GitHub

---

## MLOps Workflow

```text
Spotify Dataset
       ↓
Data Preprocessing
       ↓
Train/Test Split
       ↓
Random Forest Training
       ↓
Model Evaluation
       ↓
MLflow Experiment Tracking
       ↓
Model Registry
```

---

## Generated Files

Preprocessing generates processed training and testing datasets.

Training generates the trained Random Forest model.

Evaluation generates:

```text
metrics.json
predictions.csv
classification_report.txt
confusion_matrix.csv
```

MLflow stores experiment information in:

```text
mlflow.db
```

---

## Conclusion

This project demonstrates an end-to-end MLOps workflow for Spotify genre classification. It integrates machine learning model development with reproducible preprocessing, automated pipelines, experiment tracking, evaluation, and model registration using MLflow.