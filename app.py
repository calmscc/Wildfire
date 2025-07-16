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
from datetime import datetime

st.set_page_config(layout="wide")
st.title('Wildfire Risk Prediction')

model = joblib.load('wildfire_model.joblib')
scaler = joblib.load('scaler.joblib')
season_encoder = joblib.load('season_encoder.joblib')

# Horizontal input with Streamlit columns
fields = [
    ("Precipitation (inches)", "PRECIPITATION"),
    ("Average Wind Speed (mph)", "AVG_WIND_SPEED"),
    ("Wind/Temp Ratio", "WIND_TEMP_RATIO"),
    ("Lagged Precipitation (inches)", "LAGGED_PRECIPITATION"),
    ("Lagged Avg Wind Speed (mph)", "LAGGED_AVG_WIND_SPEED"),
    ("Minimum Temperature (°F)", "MIN_TEMP"),
    ("Maximum Temperature (°F)", "MAX_TEMP"),
]
inputs = st.columns(len(fields))
user_values = {}
for i, (label, key) in enumerate(fields):
    user_values[key] = inputs[i].number_input(label, value=0.0)

# Additional fields
st.divider()
col_date, col_season = st.columns(2)
with col_date:
    selected_date = st.date_input("Date", value=datetime.today())
    user_values["MONTH"] = selected_date.month
    user_values["DAY_OF_YEAR"] = selected_date.timetuple().tm_yday
with col_season:
    season_guess = (
        "Winter" if user_values["MONTH"] in [12,1,2]
        else "Spring" if user_values["MONTH"] in [3,4,5]
        else "Summer" if user_values["MONTH"] in [6,7,8]
        else "Autumn"
    )
    user_values["SEASON"] = st.selectbox(
        "Season (auto-set, override if needed)",
        list(season_encoder.classes_),
        index=list(season_encoder.classes_).index(season_guess)
    )

user_values["TEMP_RANGE"] = max(user_values["MAX_TEMP"] - user_values["MIN_TEMP"], 0)

if st.button("Predict Wildfire Risk"):
    zero_keys = [
        "PRECIPITATION", "AVG_WIND_SPEED", "WIND_TEMP_RATIO", 
        "LAGGED_PRECIPITATION", "LAGGED_AVG_WIND_SPEED", 
        "MIN_TEMP", "MAX_TEMP"
    ]
    # All critical numeric fields are zero
    if all(user_values[k] == 0 or user_values[k] == 0.0 for k in zero_keys):
        proba = 0.0
    else:
        feature_order = [
            "PRECIPITATION", "MAX_TEMP", "MIN_TEMP", "AVG_WIND_SPEED",
            "TEMP_RANGE", "WIND_TEMP_RATIO", "MONTH", "SEASON",
            "LAGGED_PRECIPITATION", "LAGGED_AVG_WIND_SPEED", "DAY_OF_YEAR"
        ]
        input_list = [
            user_values["PRECIPITATION"],
            user_values["MAX_TEMP"],
            user_values["MIN_TEMP"],
            user_values["AVG_WIND_SPEED"],
            user_values["TEMP_RANGE"],
            user_values["WIND_TEMP_RATIO"],
            user_values["MONTH"],
            season_encoder.transform([user_values["SEASON"]])[0],
            user_values["LAGGED_PRECIPITATION"],
            user_values["LAGGED_AVG_WIND_SPEED"],
            user_values["DAY_OF_YEAR"]
        ]
        inp_df = pd.DataFrame([input_list], columns=feature_order)
        inp_scaled = scaler.transform(inp_df)
        proba = model.predict_proba(inp_scaled)[0, 1]
    st.write(f"**Predicted wildfire risk:** {proba:.2%}")
    risk_level = "Low" if proba < 0.3 else "Moderate" if proba < 0.7 else "High"
    st.write(f"**Risk Level:** {risk_level}")

# Add heatmap, background, or any other feature as in your code.

