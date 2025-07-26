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
    # Hard-code or load from training if saved separately
    return [
        "PRECIPITATION","MAX_TEMP","MIN_TEMP","AVG_WIND_SPEED",
        "TEMP_RANGE","WIND_TEMP_RATIO","MONTH","SEASON",
        "LAGGED_PRECIPITATION","LAGGED_AVG_WIND_SPEED","DAY_OF_YEAR","TEMP_DIFF",
        "LATITUDE", "LONGITUDE", "DIST_TO_HOTSPOT_KM", "SPATIAL_CLUSTER"
    ]
    # Remove or add features as per your actual training columns

# === Load resources ===
bg_data_url = get_base64_image("wild.jpg")
github_data_url = get_base64_image("github.jpg")
model = load_model()
scaler = load_scaler()
season_encoder = load_encoder()
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

st.markdown("<h2 style='text-align: center; font-size: 35px;'>Wildfire Risk Prediction</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 17.5px;'>Enter weather, environmental, and spatial data to predict wildfire risk.</h2>", unsafe_allow_html=True)

with st.form(key="fire_form"):
    # Use a flexible arrangement for many features
    cols = st.columns(4)
    # Core weather and environmental inputs
    PRECIPITATION = cols[0].number_input("Rain Precipitation (inches)", value=0.0)
    AVG_WIND_SPEED = cols[1].number_input("Avg Wind Speed (mph)", value=0.0)
    WIND_TEMP_RATIO = cols[2].number_input("Wind/Temp Ratio", value=0.0)
    LAGGED_PRECIPITATION = cols[3].number_input("Past 7d Precip (in)", value=0.0)

    cols2 = st.columns(4)
    LAGGED_AVG_WIND_SPEED = cols2[0].number_input("Past 7d Avg Wind (mph)", value=0.0)
    MIN_TEMP_F = cols2[1].number_input("Min Temp (°F)", value=0.0)
    MAX_TEMP_F = cols2[2].number_input("Max Temp (°F)", value=0.0)
    TEMP_RANGE = MAX_TEMP_F - MIN_TEMP_F if MAX_TEMP_F >= MIN_TEMP_F else 0.0
    MONTH = cols2[3].number_input("Month (1-12)", min_value=1, max_value=12, value=datetime.today().month)

    cols3 = st.columns(4)
    DAY_OF_YEAR = cols3[0].number_input("Day of Year (1-366)", min_value=1, max_value=366, value=datetime.today().timetuple().tm_yday)
    # SEASON handling
    season_dict = {
        (12, 1, 2): "Winter",
        (3, 4, 5): "Spring",
        (6, 7, 8): "Summer",
        (9, 10, 11): "Autumn"
    }
    guessed_season = next(
        (name for months, name in zip(season_dict.keys(), season_dict.values()) if MONTH in months),
        "Unknown"
    )
    SEASON = cols3[1].selectbox(
        "Season",
        list(season_encoder.classes_),
        index=list(season_encoder.classes_).index(guessed_season) if guessed_season in season_encoder.classes_ else 0
    )
    # Geospatial features
    LATITUDE = cols3[2].number_input("Latitude", value=36.5, format="%.6f")
    LONGITUDE = cols3[3].number_input("Longitude", value=-119.5, format="%.6f")

    # Extra engineered spatial features, if your model expects them!
    cols4 = st.columns(2)
    DIST_TO_HOTSPOT_KM = cols4[0].number_input("Dist to Hotspot (km)", value=0.0)
    SPATIAL_CLUSTER = cols4[1].number_input("Spatial Cluster (int)", value=0, min_value=0)

    # TEMP_DIFF (duplicate of TEMP_RANGE, but included if your model uses both!)
    TEMP_DIFF = TEMP_RANGE

    submitted = st.form_submit_button("Predict Wildfire Risk")

# -- Prediction logic --
if submitted:
    # Compose input in the precise order expected by your model
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
        "TEMP_DIFF": TEMP_DIFF,
        "LATITUDE": LATITUDE,
        "LONGITUDE": LONGITUDE,
        "DIST_TO_HOTSPOT_KM": DIST_TO_HOTSPOT_KM,
        "SPATIAL_CLUSTER": SPATIAL_CLUSTER
    }
    # Retain only features expected by the model (in order)
    model_input = [input_data[feat] for feat in features]

    # Check for all-zeroes input
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

# -- Optional: show correlation heatmap --
if st.checkbox("Show correlation heatmap"):
    df_for_heatmap = pd.read_csv('wildfire_updated.csv')  # Use updated dataset
    df_for_heatmap = df_for_heatmap[features].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df_for_heatmap.corr()
    sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
    st.pyplot(fig)
