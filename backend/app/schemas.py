from pydantic import BaseModel, Field

class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., example=6)
    Glucose: float = Field(..., example=148)
    BloodPressure: float = Field(..., example=72)
    SkinThickness: float = Field(..., example=35)
    Insulin: float = Field(..., example=0)
    BMI: float = Field(..., example=33.6)
    DiabetesPedigreeFunction: float = Field(..., example=0.627)
    Age: int = Field(..., example=50)

class DiabetesOutput(BaseModel):
    prediction: int = Field(..., example=1)
    probability: float = Field(..., example=0.87)
