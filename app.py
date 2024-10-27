from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from joblib import load

from models.iris import Iris


class Model(BaseModel):
    id: int
    name: str
    param_count: Optional[int] = None
    _model : Optional[Any] = None


models = {
    "0" : Model(id=0, name="CNN"), 
    "1" : Model(id=1, name="Transformer"),
    "2" : Model(id=2, name="Iris"),
}
id_2_hosted_models = {
    model.id : model for model in models.values()
    }
model_names_2_id = { 
    model.name.lower() : model.id for model in models.values()
}

#TODO: fix this mess ^^
ml_models = {
    model.name : model for model in models.values()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["Iris"]._model = load('models/iris_v1.joblib')
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


################################################################


app = FastAPI(
    title="ML Repository API", 
    description="API for getting predictions from hosted ML models.", 
    version="0.0.1",
    lifespan=lifespan)


@app.get("/")
def greet_json():
    return {"Hello World": "Welcome to my ML Repository API!"}


@app.get("/hosted")
def list_models():
    "List all the hosted models."
    return models


@app.get("/hosted/id/{model_id}")
def get_by_id(model_id: int):
    "Get a specific model by its ID."
    if model_id not in id_2_hosted_models:
        raise HTTPException(status_code=404, detail=f"Model with 'id={model_id}' not found")

    return id_2_hosted_models[model_id]


@app.get("/hosted/name/{model_name}")
def get_by_name(model_name: str):
    "Get a specific model by its name."

    model_name = model_name.lower()
    if model_name not in model_names_2_id:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return id_2_hosted_models[model_names_2_id[model_name]]


@app.post("/hosted/name/{model_name}/predict", tags=["Predictions"])
async def get_prediction(model_name: str, iris: Iris):
    
    if model_name.lower() != "iris":
        raise HTTPException(status_code=501, detail="Not implemented yet.")
    
    data = dict(iris)['data']
    prediction = ml_models["Iris"]._model.predict(data).tolist()
    log_probs = ml_models["Iris"]._model.predict_proba(data).tolist()
    return {"predictions": prediction,
            "log_probs": log_probs}
