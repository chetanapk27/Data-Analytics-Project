import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load trained model and columns
model = joblib.load("fraud_model.pkl")
model_columns = joblib.load("model_columns.pkl")


@app.route("/")
def home():
    return jsonify(
        {
            "status": "Online",
            "message": "Fraud Detection API is live!",
            "model_accuracy_auc": 0.936,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json()
        input_df = pd.DataFrame([json_data])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify(
            {
                "fraud_prediction": int(prediction),
                "fraud_probability": float(probability),
                "risk_status": (
                    "HIGH RISK - Flagged for Fraud"
                    if prediction == 1
                    else "Low Risk - Legitimate"
                ),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)