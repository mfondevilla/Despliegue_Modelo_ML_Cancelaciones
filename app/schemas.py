from pydantic import BaseModel


class PredictionInput(BaseModel):
    lead_time: int
    is_family: int
    total_nights: int
    total_guests: int
    arrival_date: str


class PredictionOutput(BaseModel):
    prediction: float