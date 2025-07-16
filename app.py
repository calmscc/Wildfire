import streamlit as st
import base64

file_ = open("wild1.jpg", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{data_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

st.title('Wildfire Risk Prediction')
st.write("Enter weather/environmental data to predict wildfire risk.")

# Load model and preprocessors
model = joblib.load('wildfire_model.joblib')
scaler = joblib.load('scaler.joblib')
season_encoder = joblib.load('season_encoder.joblib')

# Condensed input UI
PRECIPITATION = st.number_input("Precipitation", value=0.0)
AVG_WIND_SPEED = st.number_input("Average Wind Speed", value=0.0)
WIND_TEMP_RATIO = st.number_input("Wind/Temp Ratio", value=0.0)
LAGGED_PRECIPITATION = st.number_input("Lagged Precipitation", value=0.0)
LAGGED_AVG_WIND_SPEED = st.number_input("Lagged Average Wind Speed", value=0.0)

# Combined temperature inputs
MIN_TEMP = st.number_input("Minimum Temperature (°C)", value=0.0)
MAX_TEMP = st.number_input("Maximum Temperature (°C)", value=0.0)
TEMP_RANGE = MAX_TEMP - MIN_TEMP if MAX_TEMP >= MIN_TEMP else 0.0

# Combined date+season input
selected_date = st.date_input("Date", value=datetime.today())
MONTH = selected_date.month
DAY_OF_YEAR = selected_date.timetuple().tm_yday

# Choose season by mapping date to season or let user override
season_dict = {
    (12,1,2): "Winter", (3,4,5): "Spring",
    (6,7,8): "Summer", (9,10,11): "Autumn"
}
# Simple mapping: customize as needed
SEASON = next(
    (name for months, name in zip(season_dict.keys(), season_dict.values()) if MONTH in months),
    "Unknown"
)

SEASON = st.selectbox("Season (auto-set by date, you can override)", list(season_encoder.classes_), index=list(season_encoder.classes_).index(SEASON) if SEASON in season_encoder.classes_ else 0)

features = [
    "PRECIPITATION", "MAX_TEMP", "MIN_TEMP", "AVG_WIND_SPEED",
    "TEMP_RANGE", "WIND_TEMP_RATIO", "MONTH", "SEASON",
    "LAGGED_PRECIPITATION", "LAGGED_AVG_WIND_SPEED", "DAY_OF_YEAR"
]

user_input = {
    "PRECIPITATION": PRECIPITATION,
    "MAX_TEMP": MAX_TEMP,
    "MIN_TEMP": MIN_TEMP,
    "AVG_WIND_SPEED": AVG_WIND_SPEED,
    "TEMP_RANGE": TEMP_RANGE,
    "WIND_TEMP_RATIO": WIND_TEMP_RATIO,
    "MONTH": MONTH,
    "SEASON": SEASON,
    "LAGGED_PRECIPITATION": LAGGED_PRECIPITATION,
    "LAGGED_AVG_WIND_SPEED": LAGGED_AVG_WIND_SPEED,
    "DAY_OF_YEAR": DAY_OF_YEAR
}

if st.button("Predict Wildfire Risk"):
    vals = [
        user_input['PRECIPITATION'],
        user_input['MAX_TEMP'],
        user_input['MIN_TEMP'],
        user_input['AVG_WIND_SPEED'],
        user_input['TEMP_RANGE'],
        user_input['WIND_TEMP_RATIO'],
        user_input['MONTH'],
        season_encoder.transform([user_input["SEASON"]])[0],
        user_input['LAGGED_PRECIPITATION'],
        user_input['LAGGED_AVG_WIND_SPEED'],
        user_input['DAY_OF_YEAR']
    ]
    # Zero-check for all numerics except season
    check_vals = [
        user_input[feat] for feat in features if feat != "SEASON"
    ]
    all_zero = all(val == 0 or val == 0.0 for val in check_vals)
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
