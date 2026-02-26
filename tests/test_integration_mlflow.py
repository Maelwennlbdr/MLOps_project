import numpy as np
import sys
import os

# Ajoute backend au path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(backend_path)

from app.predict import predict_diabetes, model


def test_mlflow_model_loaded():
    """
    Integration test :
    - MLflow model existe
    - prediction fonctionne
    """
    assert model is not None

    sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
    prediction, probability = predict_diabetes(sample)

    assert isinstance(prediction, int)
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0