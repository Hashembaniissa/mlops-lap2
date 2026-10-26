import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris ML Model API", version="1.0")

# Load the trained model
model = joblib.load("model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the MLOps Lab 2 API!"}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
