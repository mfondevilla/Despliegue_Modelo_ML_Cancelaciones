import datetime

import joblib
import pandas as pd

from app.schemas import PredictionInput

MODEL_PATH = "model/xgboost_final.pkl"
PIPELINE_PATH = "model/full_pipeline.pkl"
MARKET_SEGMENT_DEFAULT = "Online TA" # valores de la moda del dataset de entrenamiento
DISTRIBUTION_CHANNEL_DEFAULT = "TA/TO" # valores de la moda del dataset de entrenamiento

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

def predict(data: PredictionInput):

    # Fecha de la reserva = momento en que se realiza la predicción
    booking_date = datetime.now()

    # Calculamos lead_time a partir de las fechas
    lead_time = (data.arrival_date.date() - booking_date.date()).days

    if lead_time < 0:
        raise ValueError(
            "La fecha de llegada no puede ser anterior a la fecha de reserva."
        )
    
    # Variables derivadas
    is_family = 1 if data.children > 0 else 0
    
    # Valores por defecto
    babies = 0
    previous_cancellations = 0
    previous_bookings_not_canceled = 0
    days_in_waiting_list = 0
    
    # Asumimos que la habitación asignada coincide con la reservada
    assigned_room_type = data.reserved_room_type
    
    # Valores por defecto obtenidos del dataset
    market_segment = MARKET_SEGMENT_DEFAULT
    distribution_channel = DISTRIBUTION_CHANNEL_DEFAULT

    customer_type = "Transient"
    # Construimos un DataFrame con el formato original del dataset
    reservation = pd.DataFrame([{
        "hotel": "Resort Hotel",

        "lead_time": lead_time,

        "arrival_date_year": data.arrival_date.year,
        "arrival_date_month": data.arrival_date.strftime("%B"),
        "arrival_date_week_number": data.arrival_date.isocalendar().week,
        "arrival_date_day_of_month": data.arrival_date.day,

        "stays_in_weekend_nights": data.stays_in_weekend_nights,
        "stays_in_week_nights": data.stays_in_week_nights,

        "adults": data.adults,
        "children": data.children,
        "babies": babies,

        "meal": data.meal,
        "country": data.country,

        "market_segment": market_segment,
        "distribution_channel": distribution_channel,

        "is_repeated_guest": 0,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,

        "reserved_room_type": data.reserved_room_type,
        "assigned_room_type": assigned_room_type,

        "booking_changes": data.booking_changes,
        "deposit_type": data.deposit_type,

        "agent": data.agent,

        "days_in_waiting_list": days_in_waiting_list,

        "customer_type": customer_type,

        "adr": data.adr,

        "required_car_parking_spaces": data.required_car_parking_spaces,
        "total_of_special_requests": data.total_of_special_requests,

        # Columnas eliminadas por el pipeline
        "company": None,
        "reservation_status": None,
        "reservation_status_date": None,
    }])

    # Aplicamos exactamente el mismo preprocessing
    reservation_processed = pipeline.transform(reservation)

    # Predicción
    prediction = int(model.predict(reservation_processed)[0])

    # Probabilidad de cancelación
    probability = float(
        model.predict_proba(reservation_processed)[0, 1]
    )

    return {
        "prediction": prediction,
        "cancellation_probability": probability,
        "cancellation": bool(prediction)
    }
