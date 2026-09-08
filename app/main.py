from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from app import model as ml
from app.schemas import PredictionInput, PredictionOutput
from app.prediction import predict

app = FastAPI(
    title="Hotel Booking Cancellation Predictor API",
    description=(
        "API para predecir la probabilidad de cancelación de reservas de hoteles "
        "basado en un modelo de ML (XGBoost) entrenado con el dataset de Kaggle "
        "'Hotel Booking Demand'."
    ),
    version="0.0.1"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>🏨 Hotel Booking Cancellation Predictor</h2>
    <p>Pulsa el botón para abrir el formulario de predicción.</p>
    <a href="/predictor-form">
        <button>Abrir formulario de predicción</button>
    </a>
    """


@app.get("/predictor-form")
def predictor_form(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict", response_model=PredictionOutput)
def make_prediction(data: PredictionInput):
    return predict(data)