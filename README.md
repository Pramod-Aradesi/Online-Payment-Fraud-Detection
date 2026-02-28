# 💳 Online Payment Fraud Detection

An end-to-end machine learning project that detects fraudulent online payment transactions in real time using the PaySim dataset.

---

## 🎯 Project Highlights

- Analyzed **6.3 million** synthetic mobile payment transactions
- Achieved **99.75% fraud recall** using Random Forest
- Deployed as a **production REST API** using FastAPI
- Built an **interactive web dashboard** using Streamlit

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.14 |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest) |
| Model Saving | Joblib |
| API Framework | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Dataset | Online Payment Fraud Detection (Kaggle) |
| Development | Jupyter Notebook |

---

## 📁 Project Structure

```
Online Payment Fraud Detection/
  |
  |-- payment.csv              # PaySim dataset
  |-- fraud_api.py             # FastAPI backend
  |-- dashboard.py             # Streamlit frontend
  |-- Untitled.ipynb           # Jupyter exploration notebook
  |
  |-- artifacts/
       |-- fraud_model.pkl     # Trained Random Forest model
       |-- feature_names.pkl   # Feature names used by model
```

---

## 📊 Dataset

**PaySim** — Synthetic mobile money transactions from Kaggle.

| Property | Value |
|---|---|
| Total Transactions | 6,362,620 |
| Fraud Transactions | 8,213 (0.13%) |
| Legitimate Transactions | 6,354,407 (99.87%) |
| Time Period | 744 steps = 31 days |
| Transaction Types | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |

### Key Fraud Patterns Discovered
- Fraud **only** occurs in `CASH_OUT` and `TRANSFER` transactions
- **98%** of fraud completely drains the sender account to zero
- Fraud amounts are on average **8x higher** than legitimate transactions

---

## ⚙️ How It Works

### 1. Feature Engineering
Two custom features were created to capture fraud fingerprints:

```python
# Captures account draining pattern
df['balanceErrorOrig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']

# Captures sudden large deposits into recipient account
df['balanceErrorDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
```

### 2. Model Training

| Model | Fraud Recall | Result |
|---|---|---|
| Logistic Regression | 44% | Too weak |
| Random Forest | 99.75% | Selected |

### 3. API Prediction Flow
```
User sends transaction → FastAPI → Random Forest → Returns fraud verdict
```

---

## 🚀 How to Run

### Step 1 — Install Dependencies
```bash
pip install pandas scikit-learn fastapi uvicorn streamlit joblib requests
```

### Step 2 — Download Dataset
Download `payment.csv` from [Kaggle](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection/data) and place it in the project folder.

### Step 3 — Train the Model
Run this in terminal from the project folder:
```bash
python -c "
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib, os

df = pd.read_csv('payment.csv')
df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
df['type'] = df['type'].map({'CASH_IN':0,'CASH_OUT':1,'DEBIT':2,'PAYMENT':3,'TRANSFER':4})
df['balanceErrorOrig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
df['balanceErrorDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

X = df.drop(columns=['isFraud'])
y = df['isFraud']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

os.makedirs('artifacts', exist_ok=True)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
joblib.dump(model, 'artifacts/fraud_model.pkl')
joblib.dump(list(X.columns), 'artifacts/feature_names.pkl')
print('Model saved!')
"
```

### Step 4 — Start the API
Open **Terminal 1** and run:
```bash
python fraud_api.py
```
API will be live at `http://localhost:8000`

### Step 5 — Launch the Dashboard
Open **Terminal 2** and run:
```bash
python -m streamlit run dashboard.py
```
Dashboard will open at `http://localhost:8501`

---

## 🔌 API Usage

### Endpoint
```
POST http://localhost:8000/predict
```

### Transaction Type Encoding
| Type | Code |
|---|---|
| CASH_IN | 0 |
| CASH_OUT | 1 |
| DEBIT | 2 |
| PAYMENT | 3 |
| TRANSFER | 4 |

### Example 1 — Fraudulent Transaction

**Input:**
```json
{
  "step": 1,
  "type": 4,
  "amount": 500000,
  "oldbalanceOrg": 500000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 0,
  "newbalanceDest": 500000
}
```

**Output:**
```json
{
  "is_fraud": true,
  "fraud_probability": 74.0,
  "result": "FRAUD DETECTED"
}
```

### Example 2 — Legitimate Transaction

**Input:**
```json
{
  "step": 1,
  "type": 3,
  "amount": 1000,
  "oldbalanceOrg": 50000,
  "newbalanceOrig": 49000,
  "oldbalanceDest": 10000,
  "newbalanceDest": 11000
}
```

**Output:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0,
  "result": "LEGITIMATE"
}
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| True Negatives (Legit → Legit) | 1,270,881 |
| False Positives (Legit → Fraud) | 0 |
| False Negatives (Fraud missed) | 4 |
| True Positives (Fraud caught) | 1,639 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 Score | 1.00 |

---

## 📝 License

This project is for educational purposes using the publicly available PaySim dataset from Kaggle.

---

## 🙌 Acknowledgements

- Dataset: [PaySim — E. Lopez-Rojas, Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- ML Framework: [Scikit-learn](https://scikit-learn.org/)
- API Framework: [FastAPI](https://fastapi.tiangolo.com/)
- Dashboard: [Streamlit](https://streamlit.io/)
