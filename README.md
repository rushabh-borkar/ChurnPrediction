# Telco Customer Churn Prediction

A machine learning project that predicts customer churn risk for a telecom company by comparing classical machine learning and deep learning approaches, with the final model deployed through a REST API and interactive web application.

## Overview

This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) to predict whether a customer is likely to churn based on their account information, subscribed services, contract details, and billing information.

The project covers the complete machine learning workflow:

- Data cleaning and preprocessing
- Exploratory data analysis
- Model training
- Hyperparameter tuning
- Deep learning model comparison
- Model evaluation
- Classification threshold optimization
- Model serialization
- REST API deployment
- Interactive web application
- Containerized deployment with Docker

## Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning & Preprocessing
   ↓
Train/Test Split
   ↓
Feature Transformation
   ↓
Baseline Models
   ├── Logistic Regression
   └── Decision Tree
   ↓
XGBoost
   ↓
Hyperparameter Tuning
   ↓
PyTorch MLP Comparison
   ↓
Model Evaluation
   ↓
F1-Based Threshold Optimization
   ↓
Final XGBoost Pipeline
   ↓
FastAPI + Streamlit
   ↓
Docker Compose

##Models Compared

Three tuned models were evaluated using the same test set. Classification thresholds were optimized using the precision-recall curve to maximize F1 score.

Model	Accuracy	Precision	Recall	F1 Score	ROC AUC	Best Threshold
Tuned Logistic Regression	0.7608	0.5359	0.7380	0.6209	0.8402	0.5684
Tuned XGBoost (Deployed)	0.7828	0.5711	0.7299	0.6408	0.8456	0.3400
Tuned MLP      (PyTorch)	0.7679	0.5475	0.7246	0.6237	0.8331	0.2777

XGBoost was selected for deployment based on the evaluation results, including the highest F1 score and ROC AUC among the evaluated models. Logistic Regression achieved slightly higher recall, while the MLP provided an additional deep learning comparison on the tabular dataset.

##Threshold Optimization

Instead of using the default classification threshold of 0.5, the decision threshold was tuned using the precision-recall curve.
For the deployed XGBoost model:
Optimized F1 threshold: 0.34
This threshold is stored separately and used by the API during prediction.
The purpose of threshold optimization is to control the precision-recall trade-off according to the project's evaluation objective rather than relying on the default probability cutoff.

##Model Evaluation

The models were evaluated using:
Accuracy
Precision
Recall
F1 Score
ROC AUC
Confusion Matrix
ROC Curve
Precision-Recall Curve
Threshold vs F1 Curve

#Deployment Architecture
                 ┌──────────────────┐
                 │    Streamlit     │
                 │   Web Interface  │
                 └────────┬─────────┘
                          │
                          │ HTTP POST
                          ▼
                 ┌──────────────────┐
                 │     FastAPI      │
                 │    REST API      │
                 └────────┬─────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    XGBoost Pipeline   │
              │   + F1 Threshold 0.34 │
              └───────────────────────┘
The Streamlit application communicates with the FastAPI backend rather than loading the model independently. This keeps model inference centralized in the API service.

Streamlit Application
<img width="1917" height="987" alt="image" src="https://github.com/user-attachments/assets/8e842752-d403-4fd3-b3b6-9ea771a2caeb" />

FastAPI Swagger UI
<img width="1907" height="988" alt="image" src="https://github.com/user-attachments/assets/4b35b36b-c089-4388-ab06-4dd6705578f7" />

#Tech Stack
##Machine Learning
Python
Pandas
NumPy
Scikit-learn
XGBoost
PyTorch
##Backend
FastAPI
Pydantic
Uvicorn
##Frontend
Streamlit
##Deployment
Docker
Docker Compose

#Project Structure
├── app.py                        # Streamlit UI
├── main.py                       # FastAPI backend
├── code.ipynb                    # Model training and evaluation
├── churn_model_pipeline.pkl     # Trained XGBoost pipeline
├── churn_model_threshold.pkl    # F1-optimized classification threshold
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md

#Running with Docker
Make sure Docker Desktop is running, then execute:
docker compose up --build
##FastAPI
Open:
http://localhost:8000/docs
This provides the interactive Swagger UI for the API.
##Streamlit
Open:
http://localhost:8501
This provides the interactive customer churn prediction interface.

##API
POST /predict
The API accepts customer information and returns the predicted churn probability, binary prediction, and threshold used for classification.

##Example Request:
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
##Example Response:
{
  "churn_probability": 0.6899,
  "churn_prediction": 1,
  "threshold_used": 0.34
}
#Key Learnings
1.Classification threshold selection can significantly affect precision, recall, and F1 score.
2.Model evaluation should consider multiple metrics rather than accuracy alone, particularly for churn classification.
3.XGBoost, Logistic Regression, and a PyTorch MLP were evaluated on the same problem to compare different modeling approaches.
4.A trained ML model can be integrated into a production-style inference service using FastAPI.
5.Separating the frontend from the inference API provides a cleaner application architecture.
6.Docker Compose can be used to run the frontend and backend as separate services.

#Future Improvements
Add experiment tracking using MLflow or Weights & Biases
Add automated testing for the API and prediction pipeline
Add CI/CD using GitHub Actions
Add model monitoring and data-drift detection
Deploy the application to a cloud platform
Add authentication and API rate limiting

#License
**One important thing:** don't paste the `screenshots/...` section unless you're actually going to add those three image files. Otherwise GitHub will show broken images.
