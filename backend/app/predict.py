import numpy as np

def predict_diabetes(data: np.ndarray):
    """
    Fonction temporaire (mock).
    """
    probability = float(np.random.rand())
    prediction = int(probability > 0.5)

    return prediction, probability
