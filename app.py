import streamlit as st
import base64
import joblib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cache helpers ---
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

# Load resources
bg_data_url = get_base64_image("wild.jpg")
github_data_url = get_base64_image("github.jpg")
model = load_model()
scaler = load_scaler()
season_encoder = load_encoder()

# --- Page Styling ---
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
    color: #222 !important;
}}

.input-container {{
    background: rgba(255, 255, 255, 0.55); /* Lighter and more clear */
    border-radius: 18px;
    padding: 30px 25px 25px 25px;
    margin-top: 16px;
    margin-bottom: 32px;
    box-shadow: 0 4px 32px rgba(30,30,30,0.05);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    transition: background 0.4s;
}}

.stColumns > div {{
    padding: 0.25em;
}}
</style>
""", unsafe_allow_html=True)

# --- Header and GitHub ---
st.markdown("""
<a href="https://github.com/calmscc/Wildfire">
<div style="text-align: right;">
<img src="data:image/jpg;base64,{}" width="25">
</div>
</a>
""".format(github_data_url), unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; font-size: 35px;'>Wildfire Risk Prediction</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 17.5px;'>Enter the weather and environmental data to predict wildfire risk.</h2>", unsafe_allow_html=True)

st.markdown('<div class="input-container">', unsafe_allow_html=True)  # Open the box

with st.form("fire_form"):
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        PRECIPITATION = st.number_input("Rain Precipitation (inches)", value=0.0)
    with col2:
        AVG_WIND_SPEED = st.number_input("Average Wind Speed (mph)", value=0.0)
    with col3:
        WIND_TEMP_RATIO = st.number_input("Wind over Temp Ratio", value=0.0)
    with col4:
        LAGGED_PRECIPITATION = st.number_input("Past Precipitation (inches)", value=0.0)
    with col5:
        LAGGED_AVG_WIND_SPEED = st.number_input("Past Avg Wind Speed (mph)", value=0.0)
    with col6:
        MIN_TEMP_F = st.number_input("Minimum Temperature (°F)", value=0.0)
    with col7:
        MAX_TEMP_F = st.number_input("Maximum Temperature (°F)", value=0.0)

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
        detected_season = next(
            (name for months, name in zip(season_dict.keys(), season_dict.values()) if MONTH in months),
            "Unknown"
        )
        SEASON = st.selectbox(
            "Season",
            list(season_encoder.classes_),
            index=list(season_encoder.classes_).index(detected_season) if detected_season in season_encoder.classes_ else 0
        )

    submitted = st.form_submit_button("Predict Wildfire Risk")
    # ... your prediction logic here
st.markdown('</div>', unsafe_allow_html=True)  # Close the box
if submitted:
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

# --- Heatmap toggle ---
if st.checkbox("Show correlation heatmap"):
    df = pd.read_csv('wildfire_dataset.csv')
    corr = df.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots(figsize=(7, 3)) 
    sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
    st.pyplot(fig)
