from fastapi import FastAPI
import numpy as np

from app.schemas import DiabetesInput, DiabetesOutput
from app.predict import predict_diabetes

app = FastAPI(
    title="Diabetes Prediction API",
    description="API de prédiction du diabète (Pima Indians Dataset)",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    """
    Endpoint de vérification de l'état de l'API
    """
    return {"status": "OK"}

@app.post("/predict", response_model=DiabetesOutput)
def predict(input_data: DiabetesInput):
    """
    Endpoint de prédiction du diabète
    """
    data = np.array([[
        input_data.Pregnancies,
        input_data.Glucose,
        input_data.BloodPressure,
        input_data.SkinThickness,
        input_data.Insulin,
        input_data.BMI,
        input_data.DiabetesPedigreeFunction,
        input_data.Age
    ]])

    prediction, probability = predict_diabetes(data)

    return DiabetesOutput(
        prediction=prediction,
        probability=probability
    )
