import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fraud Detection Decision Dashboard", layout="centered"
)

st.title("🛡️ Mobile Money Fraud Detection System")


# Load model for direct UI scoring
model = joblib.load("fraud_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.sidebar.header("Transaction Parameters")
step = st.sidebar.number_input("Time Step", value=1)
amount = st.sidebar.number_input("Transaction Amount", value=181.0)
oldbalanceOrg = st.sidebar.number_input("Sender Old Balance", value=181.0)
newbalanceOrig = st.sidebar.number_input("Sender New Balance", value=0.0)
oldbalanceDest = st.sidebar.number_input("Receiver Old Balance", value=0.0)
newbalanceDest = st.sidebar.number_input("Receiver New Balance", value=0.0)
tx_type = st.sidebar.selectbox("Transaction Type", ["TRANSFER", "CASH-OUT", "PAYMENT"])

if st.button("Evaluate Transaction Risk"):
    input_data = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "errorBalanceOrg": oldbalanceOrg - amount - newbalanceOrig,
        "errorBalanceDest": newbalanceDest + amount - oldbalanceDest,
    }

    df_input = pd.DataFrame([input_data])
    for col in model_columns:
        if col.startswith("type_"):
            df_input[col] = 1 if col == f"type_{tx_type}" else 0

    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    if pred == 1:
        st.error(
            f"🚨 **HIGH RISK FRAUD DETECTED!** (Probability: {prob:.2f})"
        )
        st.markdown(
            "**Recommended Action:** Automatically freeze account and trigger OTP verification."
        )
    else:
        st.success(
            f"✅ **Legitimate Transaction.** (Fraud Probability: {prob:.4f})"
        )
        st.markdown(
            "**Recommended Action:** Allow transaction to clear normally."
        )