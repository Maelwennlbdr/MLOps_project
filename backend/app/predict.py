import numpy as np
import mlflow.sklearn
import os

MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/MyModel/Production")

print(f"Loading model from MLflow: {MLFLOW_MODEL_URI}")
model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)

def predict_diabetes(data: np.ndarray):
    """
    Prédiction réelle à partir du modèle MLflow
    """
    prob = model.predict_proba(data)[:, 1][0]  # probabilité de 1
    pred = int(prob > 0.5)
    return pred, float(prob)
