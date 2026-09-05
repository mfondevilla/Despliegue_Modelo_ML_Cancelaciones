from datetime import date
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
        
    arrival_date: date

    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)

    adults: int = Field(ge=0)
    children: float = Field(default=0, ge=0)

    meal: str
    country: str

    reserved_room_type: str

    booking_changes: int = Field(ge=0)
    deposit_type: str

    agent: float | None = None

    adr: float = Field(ge=0)

    required_car_parking_spaces: int = Field(default=0, ge=0)
    total_of_special_requests: int = Field(default=0, ge=0)


class PredictionOutput(BaseModel):
    prediction: int
    cancellation_probability: float
    cancellation: bool
