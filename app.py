from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel


model = joblib.load('recommended_product.pkl')
encoder = joblib.load('encoder.pkl')


app = FastAPI()


class Customer(BaseModel):
    credit_score: float
    existing_loans: int
    fraud_risk_score: float
    monthly_expenses: float


@app.post("/recommend/")
def recommend(customer: Customer):
    #
    data = pd.DataFrame([customer.dict()])
    
    pred = model.predict(data)

    product_name = encoder.inverse_transform(pred)[0]
    return {"recommended_product": product_name}
