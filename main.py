from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Entry(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Project": "Wine Prediction Project."}


@app.post("/prediction")
def make_prediction(entry: Entry):
    return
    # return {"item_id": item_id, "q": q}