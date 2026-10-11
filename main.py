import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline = joblib.load(os.path.join(BASE_DIR, "titanic_pipeline.pkl"))

app = FastAPI(title="Titanic Survival API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Lovable site URL once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)


class Passenger(BaseModel):
    Pclass: Literal[1, 2, 3]
    Sex: Literal["female", "male"]
    Age: float = Field(ge=0, le=100)
    SibSp: int = Field(ge=0, le=10)
    Parch: int = Field(ge=0, le=10)
    Fare: float = Field(ge=0)
    Embarked: Literal["C", "Q", "S"]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Passenger):
    try:
        df = pd.DataFrame([p.model_dump()])
        probability = float(pipeline.predict_proba(df)[0][1])
        return {"survived": probability >= 0.5, "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))