from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class Probabilities(BaseModel):
    cat: float
    dog: float


class PredictResponse(BaseModel):
    label: str
    probabilities: Probabilities
