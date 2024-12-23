# Telecommunication_Customer_Churning_Analysis

Objective : 
To predict whether a customer will churn (leave the service) based on their usage patterns and demographic information.

Dataset
File Name: train.csv

Columns:
-customerID: Unique identifier for each customer.
-gender: Gender of the customer.
-SeniorCitizen: Whether the customer is a senior citizen.
-tenure: Number of months the customer has stayed with the company.
-MonthlyCharges: Monthly billing amount.
-Churn: Target variable (Yes/No).
-Additional columns containing contract details, internet services, and payment methods.


Steps Involved

EDA:
-Analyze class distribution (churn vs. no churn).
-Explore correlations and visualize trends.

Data Preprocessing:
-Handle missing values.
-Convert categorical variables into numerical using one-hot encoding.
-Standardize numerical features.

Model Building:
-Train models including Logistic Regression, Random Forest, and XGBoost.
-Evaluate model performance using accuracy, precision, recall, and F1-score.

Insights:
-Key factors influencing churn (e.g., tenure, contract type).
-Recommendations for retention strategies.

Tools and Technologies
-Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn).
-XGBoost for advanced modeling.
-Streamlit for model deployment.

Results and Insights

Churn Prediction:
-Achieved an accuracy of more than 85% using XGBoost.
-Identified key drivers of churn such as contract type and monthly charges.
