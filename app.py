from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class Model(BaseModel):
    id: int
    name: str
    param_count: Optional[int] = None


app = FastAPI(
    title="ML Repository API", 
    description="API for getting predictions from hosted ML models.", 
    version="0.0.1")


models = [Model(id=0, name="CNN"), Model(id=1, name="Transformer")]
id_2_hosted_models = {
    model.id : model for model in models
}
model_names_2_id = { 
    model.name.lower() : model.id for model in models
}


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
        raise HTTPException(status_code=404, detail=f"Model with id={model_id} not found")

    return id_2_hosted_models[model_id]

@app.get("/hosted/name/{model_name}")
def get_by_name(model_name: str):
    "Get a specific model by its name."

    model_name = model_name.lower()
    if model_name not in model_names_2_id:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return id_2_hosted_models[model_names_2_id[model_name]]