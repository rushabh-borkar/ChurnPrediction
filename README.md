Customer Churn Prediction Pipeline
An end-to-end Machine Learning pipeline designed to predict customer churn and extract actionable business insights. This project processes raw customer data, trains multiple classification models, handles severe class imbalance, and optimizes decision thresholds to maximize business retention efforts.

📌 Business Problem
Customer churn is a critical metric for subscription-based businesses. The goal of this project is not only to predict who is likely to cancel their service, but to understand why they are leaving so the business can take proactive steps to retain them.

🛠️ Tech Stack
Language: Python

Data Manipulation: pandas, numpy

Machine Learning: scikit-learn, xgboost

Visualizations: matplotlib, seaborn

Deployment/Export: joblib

🧠 Methodology
Data Preprocessing: Built a robust ColumnTransformer inside a Scikit-Learn Pipeline to handle missing values, scale numerical features, and One-Hot Encode categorical variables.

Model Evaluation: Trained and evaluated 5 different models:

Logistic Regression

Random Forest

Gradient Boosting

XGBoost

Decision Tree

Handling Imbalance: Addressed the heavily imbalanced dataset (73% retained / 27% churned) by applying class_weight='balanced', shifting the model's focus to catching the minority class.

Threshold Tuning: Iterated through probability thresholds to find the exact cutoff (0.54) that perfectly balanced Precision and Recall for maximum business value.

Cross-Validation: Verified model stability using 5-Fold Stratified Cross-Validation.

📊 Key Results
Logistic Regression outperformed tree-based models on Recall and F1-Score after applying class weights and custom thresholding.

Performance Metrics (Threshold = 0.54):

ROC AUC Score: 0.841 (Excellent class separation)

Recall: 76.47% (Successfully catching over 3/4 of at-risk customers)

F1 Score: 62.17%

Accuracy: 75.30%

By adjusting the threshold from the default 0.50 to 0.54, the model successfully minimized false alarms while drastically reducing the number of missed churners (False Negatives).
