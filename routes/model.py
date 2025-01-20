from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field
from typing import Literal
from routes.task import predict_drug
import skops.io as sio

# Create an instance of the FastAPI class
router = APIRouter(
    prefix="/api",
    tags=["api"],
    responses={404: {"description": "Not found"}}
)

class PredictDrugInput(BaseModel):
    age: int = Field(..., ge=15, le=74, description="Age of the patient (15 to 74)")
    sex: Literal["M", "F"] = Field(..., description="Sex of the patient (M or F)")
    blood_pressure: Literal["HIGH", "LOW", "NORMAL"] = Field(..., description="Blood pressure level")
    cholesterol: Literal["HIGH", "NORMAL"] = Field(..., description="Cholesterol level")
    na_to_k_ratio: float = Field(..., ge=6.2, le=38.2, description="Sodium-to-potassium ratio in blood (6.2 to 38.2)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 30,
                    "sex": "M",
                    "blood_pressure": "HIGH",
                    "cholesterol": "HIGH",
                    "na_to_k_ratio": 10
                }
            ]
        }
    }

# Define a GET endpoint
@router.get("/")
def read_root():
    return {"message": "Hello, welcome to the demo"}

@router.get("/get_perams")
def get_perams():
    pipe = sio.load("./Model/drug_pipeline.skops", trusted=True)
    model = pipe.named_steps["model"]
    model_params = model.get_params()
    return model_params

@router.post("/predict")
def predict_ml(input_data: PredictDrugInput):
    label = predict_drug(input_data.age, input_data.sex, input_data.blood_pressure, input_data.cholesterol, input_data.na_to_k_ratio)
    return label