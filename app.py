import streamlit as st
st.markdown(
    """
    <style>
    .stApp {
        background-color: rgb(194, 127, 102, 1);
    }
    </style>
    """,
    unsafe_allow_html=True)

st.title('Wildfire Risk Prediction')
st.write("Enter the weather and environmental data to predict wildfire risk.")

import pandas as pd
import joblib
import numpy as np

# Load model and preprocessors
model = joblib.load('wildfire_model.joblib')
scaler = joblib.load('scaler.joblib')
season_encoder = joblib.load('season_encoder.joblib')

features = [
    "PRECIPITATION", "MAX_TEMP", "MIN_TEMP", "AVG_WIND_SPEED",
    "TEMP_RANGE", "WIND_TEMP_RATIO", "MONTH", "SEASON",
    "LAGGED_PRECIPITATION", "LAGGED_AVG_WIND_SPEED", "DAY_OF_YEAR"
]

# User input widgets
user_input = {}
for feat in features:
    if feat == "SEASON":
        season_options = list(season_encoder.classes_)
        user_input[feat] = st.selectbox("Season", season_options)
    else:
        user_input[feat] = st.number_input(feat, value=0.0)

if st.button("Predict Wildfire Risk"):
    vals = []
    for feat in features:
        if feat == "SEASON":
            vals.append(season_encoder.transform([user_input[feat]])[0])
        else:
            vals.append(user_input[feat])

    # Identify if all numeric inputs are zero (ignores SEASON)
    check_vals = [user_input[feat] for feat in features if feat != "SEASON"]
    all_zero = all(val == 0 or val == 0.0 for val in check_vals)

    # If all numeric inputs are zero, output 0%
    if all_zero:
        proba = 0.0
    else:
        input_df = pd.DataFrame([vals], columns=features)
        input_scaled = scaler.transform(input_df)
        proba = model.predict_proba(input_scaled)[0, 1]

    st.write(f"**Predicted wildfire risk:** {proba:.2%}")
    risk_level = "Low" if proba < 0.3 else "Moderate" if proba < 0.7 else "High"
    st.write(f"**Risk Level:** {risk_level}")

import matplotlib.pyplot as plt
import seaborn as sns

if st.checkbox("Show correlation heatmap"):
    df = pd.read_csv('wildfire_dataset.csv')
    corr = df.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
    st.pyplot(fig)
