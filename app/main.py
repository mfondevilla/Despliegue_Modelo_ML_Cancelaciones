from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from app import model as ml
from app.schemas import PredictionInput, PredictionOutput
from app.prediction import predict

app = FastAPI(
    title= "Hotel Booking Cancellation Predictor API",
    description= (
        "API para predecir la probabilidad de cancelación de reservas de hoteles "
        "basado en un modelo de ML(XGBoost) entrendo con el dataset de Kaggle 'Hotel Booking Demand'."
    ),
    version= "0.0.1"
)

@app.get("/")
def root():
    return {
        "message": "Machine Learning API funcionando"
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict", response_model=PredictionOutput)
def make_prediction(data: PredictionInput):
    return predict(data)