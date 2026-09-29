from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import pandas as pd
from ml.fraud_model import predict_fraud
from ml.credit_model import predict_credit_risk
from ml.identity_model import analyze_identity_risk
from rag.rag_pipeline import get_relevant_context
from rag.llm import generate_answer

app = FastAPI(title="AI Fraud Intelligence API", description="Enterprise API for Fraud, Credit Risk and RAG Intelligence")

# --- Pydantic Models for Request Validation ---
class FraudInput(BaseModel):
    amount: float
    transaction_type: str
    country: str
    merchant: str
    device: str
    previous_fraud_count: int

class CreditInput(BaseModel):
    income: float
    loan_amount: float
    credit_score: int
    employment_years: int
    debt_ratio: float

class QueryInput(BaseModel):
    question: str

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Fraud Intelligence API. Visit /docs for API documentation."}

@app.post("/predict/fraud")
def predict_fraud_api(data: FraudInput):
    try:
        prob = predict_fraud(data.dict())
        return {"fraud_probability": prob, "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/credit")
def predict_credit_api(data: CreditInput):
    try:
        risk = predict_credit_risk(data.dict())
        return {"credit_risk": risk}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_rag_api(data: QueryInput):
    try:
        context, docs = get_relevant_context(data.question)
        answer = generate_answer(data.question, context)
        sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
        return {
            "answer": answer,
            "context": context,
            "sources": list(set(sources))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
