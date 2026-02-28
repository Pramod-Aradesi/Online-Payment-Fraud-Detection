import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load saved model
model = joblib.load('artifacts/fraud_model.pkl')

# Create app
app = FastAPI(title='Fraud Detection API')

# Define transaction structure
class Transaction(BaseModel):
    step           : int
    type           : int
    amount         : float
    oldbalanceOrg  : float
    newbalanceOrig : float
    oldbalanceDest : float
    newbalanceDest : float

# Home route
@app.get('/')
def home():
    return {'message': 'Fraud Detection API is running!'}

# Prediction route
@app.post('/predict')
def predict(transaction: Transaction):
    data = {
        'step'            : transaction.step,
        'type'            : transaction.type,
        'amount'          : transaction.amount,
        'oldbalanceOrg'   : transaction.oldbalanceOrg,
        'newbalanceOrig'  : transaction.newbalanceOrig,
        'oldbalanceDest'  : transaction.oldbalanceDest,
        'newbalanceDest'  : transaction.newbalanceDest,
        'balanceErrorOrig': transaction.oldbalanceOrg - transaction.amount - transaction.newbalanceOrig,
        'balanceErrorDest': transaction.oldbalanceDest + transaction.amount - transaction.newbalanceDest
    }
    df_input    = pd.DataFrame([data])
    prediction  = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1] * 100
    return {
        'is_fraud'          : bool(prediction),
        'fraud_probability' : round(float(probability), 2),
        'result'            : 'FRAUD DETECTED' if prediction == 1 else 'LEGITIMATE'
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)