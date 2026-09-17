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
