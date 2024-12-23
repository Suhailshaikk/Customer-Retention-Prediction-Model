# Telecommunication Customer Churning Analysis

## Objective
To predict whether a customer will churn (leave the service) based on their usage patterns and demographic information.

## Dataset Overview

### Dataset File Name
`train.csv`

### Dataset Size
- **Number of Rows**: 5000
- **Number of Columns**: 21

### Columns
- **Unnamed: 0**: Index column.
- **state**: State of the customer.
- **area.code**: Area code of the customer.
- **account.length**: Duration of the account in months.
- **voice.plan**: Whether the customer has a voice plan (Yes/No).
- **voice.messages**: Number of voice messages.
- **intl.plan**: Whether the customer has an international plan (Yes/No).
- **intl.mins**: International call minutes used.
- **intl.calls**: Number of international calls made.
- **intl.charge**: Charge for international calls.
- **day.mins**: Minutes spent on calls during the day.
- **day.calls**: Number of calls made during the day.
- **day.charge**: Charge for daytime calls.
- **eve.mins**: Minutes spent on calls during the evening.
- **eve.calls**: Number of evening calls made.
- **eve.charge**: Charge for evening calls.
- **night.mins**: Minutes spent on calls during the night.
- **night.calls**: Number of night calls made.
- **night.charge**: Charge for nighttime calls.
- **customer.calls**: Number of calls made to customer support.
- **churn**: Target variable indicating churn (Yes/No).

## Steps Involved

### 1. Exploratory Data Analysis (EDA)
- Analyze class distribution (churn vs. no churn).
- Explore correlations and visualize trends.
- Identify outliers and anomalies in key features like `intl.mins` and `day.mins`.

### 2. Data Preprocessing
- Handle missing values in columns like `day.charge` and `eve.mins`.
- Convert categorical variables (`voice.plan`, `intl.plan`, `state`) into numerical values using one-hot encoding.
- Standardize numerical features (`intl.mins`, `day.mins`, `night.mins`, etc.) to improve model performance.

### 3. Model Building
- Train models such as Logistic Regression, Random Forest, and XGBoost.
- Optimize hyperparameters for the best performance.
- Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.

### 4. Insights
- Key factors influencing churn include:
  - High monthly charges.
  - Short tenure.
  - High number of calls to customer support.
- Recommendations for retention strategies:
  - Offer discounts or flexible contracts for customers with shorter tenures.
  - Improve customer support quality for frequent callers.

### 5. Deployment
- Build a Streamlit web application to:
  - Upload new customer data.
  - Predict churn probability.
  - Display insights and recommendations interactively.

## Tools and Technologies
- **Python**: For data processing and analysis.
  - Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
- **XGBoost**: For advanced modeling.
- **Streamlit**: For deployment.

## Results and Insights

### Churn Prediction
- Achieved an accuracy of over **85%** using XGBoost.
- Identified top features contributing to churn, including:
  - `monthly.charges`
  - `customer.calls`
  - `account.length`

### Key Recommendations
- Focus on reducing churn among customers with high monthly charges by offering loyalty benefits.
- Address frequent complaints by improving customer service.
- Use predictive analytics to identify high-risk customers early and intervene proactively.

## Future Enhancements
- Incorporate additional customer behavior data (e.g., web/app usage).
- Use deep learning models to capture complex patterns.
- Perform segmentation analysis to develop tailored retention strategies.


