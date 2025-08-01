import streamlit as st
import base64
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# === Cache helpers ===
@st.cache_data
def get_base64_image(img_path):
    with open(img_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

@st.cache_resource
def load_model():
    return joblib.load('wildfire_model.joblib')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.joblib')

@st.cache_resource
def load_encoder():
    return joblib.load('season_encoder.joblib')

@st.cache_data
def load_features():
    return [
        "PRECIPITATION", "MAX_TEMP", "MIN_TEMP", "AVG_WIND_SPEED",
        "TEMP_RANGE", "WIND_TEMP_RATIO", "MONTH", "SEASON",
        "LAGGED_PRECIPITATION", "LAGGED_AVG_WIND_SPEED", "DAY_OF_YEAR", "TEMP_DIFF"
    ]

# === Load resources ===
bg_data_url = get_base64_image("wild.jpg")
github_data_url = get_base64_image("github.jpg")
model = load_model()
scaler = load_scaler()
season_encoder = load_encoder()
season_encoder.classes_ = np.array(["Winter", "Spring", "Summer", "Fall"])

features = load_features()

# === Styling ===
st.set_page_config(layout="wide")
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_data_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
.stNumberInput input,
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {{
    background-color: #fff !important;
    color: #000 !important;
}}
.stColumns > div {{ padding: 0.25em; }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<a href="https://github.com/calmscc/Wildfire">
<div style="text-align: right;">
<img src="data:image/jpg;base64,{}" width="25">
</div>
</a>
""".format(github_data_url), unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; font-size: 40px;'> California Wildfire Risk Prediction</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 17.5px;'>Enter weather and environmental data to predict wildfire risk.</h2>", unsafe_allow_html=True)

with st.form(key="fire_form"):
    cols = st.columns(4)
    PRECIPITATION = cols[0].number_input("Rain Precipitation (inches)", value=0.0)
    AVG_WIND_SPEED = cols[1].number_input("Avg Wind Speed (mph)", value=0.0)
    WIND_TEMP_RATIO = cols[2].number_input("Wind/Max Temp Ratio", value=0.0)
    LAGGED_PRECIPITATION = cols[3].number_input("Past Week Precipitation (inches)", value=0.0)

    cols2 = st.columns(4)
    LAGGED_AVG_WIND_SPEED = cols2[0].number_input("Past 7d Avg Wind Speed (mph)", value=0.0)
    MIN_TEMP_F = cols2[1].number_input("Min Temperature (°F)", value=0.0)
    MAX_TEMP_F = cols2[2].number_input("Max Temperature (°F)", value=0.0)

    selected_date = cols2[3].date_input("Select Date", value=datetime.today())
    MONTH = selected_date.month
    DAY_OF_YEAR = selected_date.timetuple().tm_yday

    # Automatically determine season
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    SEASON = get_season(MONTH)
    st.markdown(f"**Season (auto-detected):** {SEASON}")

    TEMP_RANGE = MAX_TEMP_F - MIN_TEMP_F if MAX_TEMP_F >= MIN_TEMP_F else 0.0
    TEMP_DIFF = TEMP_RANGE  # Keeping consistent with model input naming

    submitted = st.form_submit_button("Predict Wildfire Risk")

if submitted:
    input_data = {
        "PRECIPITATION": PRECIPITATION,
        "MAX_TEMP": MAX_TEMP_F,
        "MIN_TEMP": MIN_TEMP_F,
        "AVG_WIND_SPEED": AVG_WIND_SPEED,
        "TEMP_RANGE": TEMP_RANGE,
        "WIND_TEMP_RATIO": WIND_TEMP_RATIO,
        "MONTH": MONTH,
        "SEASON": season_encoder.transform([SEASON])[0],
        "LAGGED_PRECIPITATION": LAGGED_PRECIPITATION,
        "LAGGED_AVG_WIND_SPEED": LAGGED_AVG_WIND_SPEED,
        "DAY_OF_YEAR": DAY_OF_YEAR,
        "TEMP_DIFF": TEMP_DIFF
    }

    model_input = [input_data[feat] for feat in features]

    zero_fields = [
        PRECIPITATION, AVG_WIND_SPEED, WIND_TEMP_RATIO, LAGGED_PRECIPITATION,
        LAGGED_AVG_WIND_SPEED, MIN_TEMP_F, MAX_TEMP_F
    ]
    if all(val == 0 or val == 0.0 for val in zero_fields):
        proba = 0.0
    else:
        input_df = pd.DataFrame([model_input], columns=features)
        input_scaled = scaler.transform(input_df)
        proba = model.predict_proba(input_scaled)[0, 1]

    st.write(f"**Predicted wildfire risk:** {proba:.2%}")
    risk_level = "Low" if proba < 0.3 else "Moderate" if proba < 0.7 else "High"
    st.write(f"**Risk Level:** {risk_level}")

if st.checkbox("Show correlation heatmap"):
    df_for_heatmap = pd.read_csv('wildfire_updated.csv')

    if 'TEMP_DIFF' not in df_for_heatmap.columns:
        df_for_heatmap['TEMP_DIFF'] = df_for_heatmap['MAX_TEMP'] - df_for_heatmap['MIN_TEMP']

    # Select only features that exist
    heatmap_features = [col for col in features if col in df_for_heatmap.columns]
    df_for_heatmap = df_for_heatmap[heatmap_features].copy()

    # Convert to numeric, coerce errors
    df_for_heatmap = df_for_heatmap.apply(pd.to_numeric, errors='coerce')

    # Optionally fill NaN to avoid errors
    df_for_heatmap = df_for_heatmap.fillna(0)

    corr = df_for_heatmap.corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
    st.pyplot(fig)

