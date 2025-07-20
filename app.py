import streamlit as st
import base64
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(layout="wide")

# Load image and convert to base64 once
with open("wild.jpg", "rb") as f:
    contents = f.read()
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

# GitHub icon
st.markdown(
    """<a href="https://github.com/calmscc/Wildfire">
    <div style="text-align: right;">
    <img src="data:image/jpg;base64,{}" width="25">
    </a>""".format(
        base64.b64encode(open("github.jpg", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)

# Custom widget styles
st.markdown("""
<style>
.stNumberInput input,
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    background-color: #fff !important;
    color: #222 !important;
}
.stColumns {
    background: rgba(8, 5, 5, 0.52); 
    padding: 1.2em;
    border-radius: 10px;
    margin-bottom: 1em;
}
.stColumns > div {
    padding: 0.25em;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('wildfire_model.joblib')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.joblib')

@st.cache_resource
def load_encoder():
    return joblib.load('season_encoder.joblib')

model = load_model()
scaler = load_scaler()
season_encoder = load_encoder()

st.markdown("<h2 style='text-align: center; font-size: 35px;'>Wildfire Risk Prediction</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 17.5px;'>Enter the weather and environmental data to predict wildfire risk.</h2>", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    PRECIPITATION = st.number_input("Rain Precipitation (inches)", value=st.session_state.get("PRECIPITATION", 0.0), key="PRECIPITATION")
with col2:
    AVG_WIND_SPEED = st.number_input("Average Wind Speed (mph)", value=st.session_state.get("AVG_WIND_SPEED", 0.0), key="AVG_WIND_SPEED")
with col3:
    WIND_TEMP_RATIO = st.number_input("Wind over Temp Ratio", value=st.session_state.get("WIND_TEMP_RATIO", 0.0), key="WIND_TEMP_RATIO")
with col4:
    LAGGED_PRECIPITATION = st.number_input("Past Precipitation (inches)", value=st.session_state.get("LAGGED_PRECIPITATION", 0.0), key="LAGGED_PRECIPITATION")
with col5:
    LAGGED_AVG_WIND_SPEED = st.number_input("Past Avg Wind Speed (mph)", value=st.session_state.get("LAGGED_AVG_WIND_SPEED", 0.0), key="LAGGED_AVG_WIND_SPEED")
with col6:
    MIN_TEMP_F = st.number_input("Minimum Temperature (°F)", value=st.session_state.get("MIN_TEMP_F", 0.0), key="MIN_TEMP_F")
with col7:
    MAX_TEMP_F = st.number_input("Maximum Temperature (°F)", value=st.session_state.get("MAX_TEMP_F", 0.0), key="MAX_TEMP_F")

st.divider()
col_date, col_season = st.columns(2)
with col_date:
    selected_date = st.date_input("Date", value=datetime.today())
    MONTH = selected_date.month
    DAY_OF_YEAR = selected_date.timetuple().tm_yday
with col_season:
    season_dict = {
        (12, 1, 2): "Winter",
        (3, 4, 5): "Spring",
        (6, 7, 8): "Summer",
        (9, 10, 11): "Autumn"
    }
    detected_season = next((name for months, name in zip(season_dict.keys(), season_dict.values()) if MONTH in months), "Unknown")
    SEASON = st.selectbox("Season", list(season_encoder.classes_), index=list(season_encoder.classes_).index(detected_season) if detected_season in season_encoder.classes_ else 0)

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

if st.checkbox("Show correlation heatmap"):
    df = pd.read_csv('wildfire_dataset.csv')
    corr = df.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
    st.pyplot(fig)
