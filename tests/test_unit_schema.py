from fastapi.testclient import TestClient
import sys
import os

# Ajoute backend au path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(backend_path)

from backend.app.main import app

client = TestClient(app)


def test_invalid_input_schema():
    """
    Unit test : validation Pydantic
    - input manquant ou type incorrect
    -> doit retourner 422
    """
    invalid_sample = {
        "Pregnancies": "six",  # mauvais type
        "Glucose": 148,
    }

    response = client.post("/predict", json=invalid_sample)
    assert response.status_code == 422
