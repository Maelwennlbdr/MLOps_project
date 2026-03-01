import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    max_iter = 1000
    model = LogisticRegression(max_iter=max_iter)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig("confusion_matrix.png")

    # Logs MLflow
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_param("git_commit", git_commit)
    mlflow.log_artifact("confusion_matrix.png")

    result = mlflow.sklearn.log_model(model, "model")

    mlflow.register_model(
    result.model_uri,
    "mlops-model"
    )

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
