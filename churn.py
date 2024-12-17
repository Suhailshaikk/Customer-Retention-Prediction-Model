import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Function to train the model
def train_model():
    # Load and clean the dataset
    df = pd.read_csv(r"C:\Users\DELL\Desktop\DS_Project\Churn.csv")
    df.replace('nan', 0, inplace=True)

    # Drop unnecessary columns
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Encode the target variable
    label_encoder = LabelEncoder()
    df['churn'] = label_encoder.fit_transform(df['churn'])

    # Convert specific columns to numeric and handle missing values
    numeric_cols_to_convert = ["day.charge", "eve.mins"]
    for col in numeric_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    # Standardize categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.lower()

    # Encode all object-type columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Add outlier detection as features
    def detect_outliers_iqr(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[f"{col}_outlier"] = detect_outliers_iqr(df[col]).astype(int)

    # Separate features and target
    X = df.drop(columns=['churn'])
    y = df['churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    return xgb_model, X

# Check if the model is already trained and stored
if "model" not in st.session_state:
    st.session_state.model, st.session_state.X = train_model()

# Streamlit app
st.title("Customer Churn Prediction")
st.subheader("Enter Customer Information")

# Input fields
with st.form("churn_form"):
    account_length = st.number_input("Account Length (months)", min_value=1, format="%.0f")
    voice_plan = st.radio("Do you have a Voice Plan?", options=["Yes", "No"])
    intl_plan = st.radio("Do you have an International Plan?", options=["Yes", "No"])
    day_charge = st.number_input("Day Charge", min_value=0.0, format="%.2f")
    eve_charge = st.number_input("Evening charge", min_value=0.0, format="%.2f")
    night_charge = st.number_input("Night Charge", min_value=0.0, format="%.2f")
    submit_button = st.form_submit_button("Predict Churn")

if submit_button:
    # Preprocessing the input
    voice_plan_encoded = 1 if voice_plan.lower() == "yes" else 0
    intl_plan_encoded = 1 if intl_plan.lower() == "yes" else 0
    total_monthly_charge = day_charge + night_charge + eve_charge  # Example approximation for eve charge

    # Collect the data
    input_data = pd.DataFrame({
        "account.length": [account_length],
        "voice.plan": [voice_plan_encoded],
        "intl.plan": [intl_plan_encoded],
        "total.monthly.charge": [total_monthly_charge],
        "eve.charge": [eve_charge],
        "night.charge": [night_charge],
    })

    # Add missing columns to match the model's expected input
    for col in st.session_state.X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[st.session_state.X.columns]

    # Make a prediction using the stored model
    prediction = st.session_state.model.predict(input_data)
    churn_result = "YES" if prediction[0] == 1 else "NO"

    st.write(f"### Churn Prediction: {churn_result}")
