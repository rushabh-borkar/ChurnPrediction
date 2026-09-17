import os
import requests
import streamlit as st

# FastAPI URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📉"
)

st.title("📉 Telco Customer Churn Predictor")
st.write("Fill in customer details to predict churn risk.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )


if st.button("Predict Churn"):

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        prob = result["churn_probability"]
        prediction = result["churn_prediction"]
        threshold = result["threshold_used"]

        st.divider()

        if prediction == 1:
            st.error(
                f"⚠️ High Churn Risk — Probability: {prob:.1%}"
            )
        else:
            st.success(
                f"✅ Low Churn Risk — Probability: {prob:.1%}"
            )

        st.caption(
            f"Decision threshold used: {threshold:.2f}"
        )

    except requests.exceptions.RequestException as e:
        st.error(
            f"Could not connect to the prediction API: {e}"
        )