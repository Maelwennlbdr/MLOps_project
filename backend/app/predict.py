import numpy as np
import mlflow.sklearn
import os

mlflow.set_tracking_uri("https://dagshub.com/louiseLV/MLOps_project-dagshub.mlflow")

MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/mlops-model@Staging")

print(f"Loading model from MLflow: {MLFLOW_MODEL_URI}")
model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)


def predict_diabetes(data: np.ndarray):
    """
    Prediction using MLflow pyfunc model.
    """
    import pandas as pd

    columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    df = pd.DataFrame(data, columns=columns)

    prediction = model.predict(df)

    pred_value = int(prediction[0])

    probability = 0.0

    return pred_value, probability
