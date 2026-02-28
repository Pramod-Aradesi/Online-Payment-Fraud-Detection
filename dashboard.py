import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("Online Payment Fraud Detection")
st.write("Fill in the transaction details below.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    step     = st.number_input("Step (Hour)", min_value=1, value=1)
    txtype   = st.selectbox("Transaction Type", ["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"])
    amount   = st.number_input("Amount ($)", min_value=0.0, value=1000.0)
    old_orig = st.number_input("Sender Old Balance ($)", min_value=0.0, value=50000.0)

with col2:
    new_orig = st.number_input("Sender New Balance ($)", min_value=0.0, value=49000.0)
    old_dest = st.number_input("Recipient Old Balance ($)", min_value=0.0, value=10000.0)
    new_dest = st.number_input("Recipient New Balance ($)", min_value=0.0, value=11000.0)

st.divider()

if st.button("Check Transaction", use_container_width=True):
    type_map = {"CASH_IN":0,"CASH_OUT":1,"DEBIT":2,"PAYMENT":3,"TRANSFER":4}
    payload  = {
        "step"           : step,
        "type"           : type_map[txtype],
        "amount"         : amount,
        "oldbalanceOrg"  : old_orig,
        "newbalanceOrig" : new_orig,
        "oldbalanceDest" : old_dest,
        "newbalanceDest" : new_dest
    }
    response = requests.post("http://localhost:8000/predict", json=payload)
    result   = response.json()
    st.divider()
    if result["is_fraud"]:
        st.error("FRAUD DETECTED!")
        st.metric("Fraud Probability", str(result["fraud_probability"]) + "%")
        st.write("This transaction is suspicious. Please review immediately.")
    else:
        st.success("LEGITIMATE TRANSACTION")
        st.metric("Fraud Probability", str(result["fraud_probability"]) + "%")
        st.write("This transaction appears safe.")
