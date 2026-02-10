import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from data_loader import load_datasets

st.set_page_config(page_title="Disease Prediction System", layout="wide")

st.title("🩺 Disease Prediction System")
st.markdown("""
Predict the possibility of diseases using machine learning. 
This dashboard uses datasets for **Breast Cancer**, **Heart Disease**, and **Diabetes**.
""")

@st.cache_data
def get_data():
    return load_datasets()

datasets = get_data()

# Sidebar for selection
st.sidebar.header("Settings")
disease_choice = st.sidebar.selectbox("Select Disease", list(datasets.keys()))
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "SVM", "Random Forest", "XGBoost"])

df = datasets[disease_choice]
X = df.drop('target', axis=1) if 'target' in df.columns else df.drop('Outcome', axis=1)
y = df['target'] if 'target' in df.columns else df['Outcome']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model Training (Cache for performance)
@st.cache_resource
def train_model(X_train, y_train, model_name):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "SVM":
        model = SVC(probability=True)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100)
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    model.fit(X_train, y_train)
    return model

model = train_model(X_scaled, y, model_choice)

# Input Section
st.header(f"Predict {disease_choice.replace('_', ' ').title()}")
cols = st.columns(3)
user_input = {}

for i, col_name in enumerate(X.columns):
    with cols[i % 3]:
        val = st.number_input(f"{col_name}", value=float(X[col_name].mean()))
        user_input[col_name] = val

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None
    
    st.subheader("Result:")
    if prediction == 1:
        st.error(f"⚠️ High Possibility of Disease detected.")
    else:
        st.success(f"✅ Low Possibility of Disease detected.")
    
    if prob is not None:
        st.info(f"Confidence Level: {prob*100:.2f}%")

# Data Preview
if st.checkbox("Show Data Preview"):
    st.write(df.head())
