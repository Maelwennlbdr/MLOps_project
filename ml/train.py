import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlflow.tracking import MlflowClient
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

mlflow.set_tracking_uri("https://dagshub.com/louiseLV/MLOps_project-dagshub.mlflow")
mlflow.set_experiment("diabetes-mlops-experiment")

# récupérer le commit git
git_commit = (
    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    .decode("utf-8")
    .strip()
)

# récupérer la version DVC du dataset
try:
    dvc_version = (
        subprocess.check_output(["dvc", "dag"])
        .decode("utf-8")
        .strip()
    )
except:
    dvc_version = "unknown"

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
    mlflow.log_param("dvc_version", dvc_version)

    result = mlflow.sklearn.log_model(model, "model")

    # enregistrer le modèle
    registered_model = mlflow.register_model(result.model_uri, "mlops-model")

    # créer client MLflow
    client = MlflowClient()

    # récupérer la version créée
    model_version = registered_model.version

    # assigner automatiquement l'alias Staging
    client.set_registered_model_alias(
        name="mlops-model", alias="Staging", version=model_version
    )

    print(f"Model version {model_version} promoted to Staging")
    print("Note: Promotion to Production happens after quality gates validation")
    print("      Run: python ml/quality_gates.py")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
