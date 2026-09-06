# Telco Customer Churn Prediction

A machine learning project that predicts customer churn risk for a telecom company, comparing classical ML and deep learning approaches, and deployed as both a REST API and an interactive web app.

## Overview

This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) to predict whether a customer is likely to churn, based on their account details, services subscribed, and billing information.

## Models Compared

Three models were trained and evaluated, with thresholds tuned to optimize F1 score:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Best Threshold |
|---|---|---|---|---|---|---|
| Tuned Logistic Regression | 0.7608 | 0.5359 | 0.7380 | 0.6209 | 0.8402 | 0.5684 |
| **Tuned XGBoost (deployed)** | **0.7828** | **0.5711** | 0.7299 | **0.6408** | **0.8456** | 0.3400 |
| Tuned MLP (PyTorch, Deep Learning) | 0.7679 | 0.5475 | 0.7246 | 0.6237 | 0.8331 | 0.2777 |

**XGBoost was selected as the final deployed model**, outperforming both Logistic Regression and a PyTorch feedforward neural network across every metric. This is consistent with well-documented findings that gradient boosting methods tend to outperform deep learning on small-to-medium tabular datasets — deep learning's advantages typically emerge with much larger datasets or unstructured data (images, text).

## Tech Stack

- **Modeling**: scikit-learn, XGBoost, PyTorch
- **API**: FastAPI
- **UI**: Streamlit
- **Deployment**: Docker, Docker Compose

## Project Structure

```
├── app.py                        # Streamlit UI
├── main.py                       # FastAPI backend
├── code.ipynb                    # Model training, evaluation, and comparison
├── churn_model_pipeline.pkl      # Final trained XGBoost pipeline
├── churn_model_threshold.pkl     # Optimal decision threshold (F1-tuned)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Running Locally with Docker

```bash
docker-compose up --build
```

- FastAPI: [http://localhost:8000/docs](http://localhost:8000/docs) — interactive Swagger UI for the `/predict` endpoint
- Streamlit: [http://localhost:8501](http://localhost:8501) — interactive form-based UI

## API Example

**POST** `/predict`

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0,
  "TotalCharges": 170.0
}
```

**Response:**
```json
{
  "churn_probability": 0.6899,
  "churn_prediction": 1,
  "threshold_used": 0.34
}
```

## Key Learnings

- Tuning the decision threshold (rather than defaulting to 0.5) meaningfully improves F1 score on imbalanced classification tasks like churn prediction.
- Gradient boosting (XGBoost) outperformed a PyTorch neural network on this tabular dataset, reinforcing that deep learning isn't always the right tool — model choice should be driven by data characteristics, not by trend.
- Deployed the same model behind two interfaces (REST API + interactive UI) using Docker Compose, reflecting a realistic multi-service architecture.

## Future Improvements

- Refactor Streamlit to call the FastAPI backend instead of loading the model independently (cleaner single-source-of-truth architecture)
- Add experiment tracking (MLflow / Weights & Biases)
- Add CI/CD via GitHub Actions
