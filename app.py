import streamlit as st
import base64

file_ = open("wild.jpg", "rb")
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



file_ = open("wild.jpg", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode()

st.markdown(
    """<a href="https://github.com/calmscc/Wildfire">
    <div style="text-align: right;">
    <img src="data:image/jpg;base64,{}" width="25">
    </a>""".format(
        base64.b64encode(open("github.jpg", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime


# Set Streamlit layout to wide for better horizontal space
st.set_page_config(layout="wide")

st.markdown(
    "<h2 style='text-align: center; font-size: 22px;'>Wildfire Risk Prediction</h2>",
    unsafe_allow_html=True

)
st.markdown(
    "<h2 style='text-align: center; font-size: 14px;'>Enter the weather and environmental data to predict wildfire risk.</h2>",
    unsafe_allow_html=True
)


# --- LOAD MODEL AND ENCODERS ---
model = joblib.load('wildfire_model.joblib')
scaler = joblib.load('scaler.joblib')
season_encoder = joblib.load('season_encoder.joblib')

# --- HORIZONTAL LAYOUT: INPUT COLUMNS ---
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    PRECIPITATION = st.number_input("Rain Precipitation (inches)", value=0.0)
with col2:
    AVG_WIND_SPEED = st.number_input("Average Wind Speed (mph)", value=0.0)
with col3:
    WIND_TEMP_RATIO = st.number_input("Wind/Temp Ratio", value=0.0)
with col4:
    LAGGED_PRECIPITATION = st.number_input("Past Precipitation (inches)", value=0.0)
with col5:
    LAGGED_AVG_WIND_SPEED = st.number_input("Past Avg Wind Speed (mph)", value=0.0)
with col6:
    MIN_TEMP_F = st.number_input("Minimum Temperature (°F)", value=0.0)
with col7:
    MAX_TEMP_F = st.number_input("Maximum Temperature (°F)", value=0.0)

# Additional features, vertical below the main row for clarity:
st.divider()
col_date, col_season = st.columns(2)
with col_date:
    selected_date = st.date_input("Date", value=datetime.today())
    MONTH = selected_date.month
    DAY_OF_YEAR = selected_date.timetuple().tm_yday
with col_season:
    # Estimate season and allow override
    season_dict = {
        (12, 1, 2): "Winter",
        (3, 4, 5): "Spring",
        (6, 7, 8): "Summer",
        (9, 10, 11): "Autumn"
    }
    detected_season = next(
        (name for months, name in zip(season_dict.keys(), season_dict.values()) if MONTH in months),
        "Unknown"
    )
    SEASON = st.selectbox(
        "Season (auto-set by date, override if desired)",
        list(season_encoder.classes_),
        index=list(season_encoder.classes_).index(detected_season) if detected_season in season_encoder.classes_ else 0
    )

# Compute derived features
TEMP_RANGE = MAX_TEMP_F - MIN_TEMP_F if MAX_TEMP_F >= MIN_TEMP_F else 0.0

features = [
    "PRECIPITATION", "MAX_TEMP", "MIN_TEMP", "AVG_WIND_SPEED",
    "TEMP_RANGE", "WIND_TEMP_RATIO", "MONTH", "SEASON",
    "LAGGED_PRECIPITATION", "LAGGED_AVG_WIND_SPEED", "DAY_OF_YEAR"
]

user_input = {
    "PRECIPITATION": PRECIPITATION,
    "MAX_TEMP": MAX_TEMP_F,
    "MIN_TEMP": MIN_TEMP_F,
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
    input_vals = [
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
    zero_fields = [
        PRECIPITATION, AVG_WIND_SPEED, WIND_TEMP_RATIO, LAGGED_PRECIPITATION,
        LAGGED_AVG_WIND_SPEED, MIN_TEMP_F, MAX_TEMP_F
    ]
    if all(val == 0 or val == 0.0 for val in zero_fields):
        proba = 0.0
    else:
        input_df = pd.DataFrame([input_vals], columns=features)
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
