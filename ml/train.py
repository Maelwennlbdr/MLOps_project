import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import mlflow
import mlflow.sklearn
import subprocess

DATA_PATH = "data/raw/diabetes.csv"
MODEL_DIR = "ml/models"
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_experiment("diabetes-mlops-experiment")

# récupérer le commit git 
git_commit = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"]
).decode("utf-8").strip()

with mlflow.start_run():

    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns="Outcome")
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    max_iter = 1000
    model = LogisticRegression(max_iter=max_iter)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")

    # Logs MLflow
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("git_commit", git_commit)

    result = mlflow.sklearn.log_model(model, "model")

    mlflow.register_model(
    result.model_uri,
    "mlops-model"
    )

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
