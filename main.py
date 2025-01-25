from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()


class Entry(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    redisual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def read_root():
    return {"Project": "Wine Prediction Project."}


@app.post("/prediction")
def make_prediction(entry: Entry):
    model = pickle.load(open('../model/wine_rmodel.pkl', 'rb'))
    test = [[entry.fixed_acidity, entry.volatile_acidity, entry.citric_acid,
             entry.redisual_sugar, entry.chlorides, entry.free_sulfur_dioxide,
             entry.total_sulfur_dioxide, entry.density, entry.pH, entry.sulphates, entry.alcohol]]
    return model.predict([test])[0]
