import numpy as np
from app.model import model
from app.schemas import PredictionInput


def predict(data: PredictionInput):

    features = np.array([
        [
            data.edad,
            data.ingresos,
            data.antiguedad
        ]
    ])

    prediction = model.predict(features)

    return float(prediction[0])