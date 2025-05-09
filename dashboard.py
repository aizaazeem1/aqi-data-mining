import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="ShineAQI Dashboard", layout="wide")
st.title("🌤️ ShineAQI: Lahore Air Quality Dashboard")

# Load data
data = pd.read_csv("data/lahore_features_with_aqi.csv")

# Sidebar: Select a row or enter features manually
st.sidebar.header("Select Data Point or Enter Features")
row_idx = st.sidebar.slider("Select row from dataset", 0, len(data)-1, len(data)-1)
row = data.iloc[row_idx]

# Show selected features
st.write("### Selected Features", row)

# Prepare API data
def get_api_data(row):
    return {
        "co": float(row["co"]),
        "no": float(row["no"]),
        "no2": float(row["no2"]),
        "o3": float(row["o3"]),
        "so2": float(row["so2"]),
        "pm2_5": float(row["pm2_5"]),
        "pm10": float(row["pm10"]),
        "temp": float(row["temp"]),
        "rhum": float(row["rhum"]),
        "pres": float(row["pres"]),
        "wspd": float(row["wspd"]),
        "hour": int(row["hour"]),
        "day": int(row["day"]),
        "weekday": int(row["weekday"]),
        "month": int(row["month"]),
        "AQI_change_1h": float(row["AQI_change_1h"]) if pd.notna(row["AQI_change_1h"]) else 0.0,
        "AQI_rolling_mean_6h": float(row["AQI_rolling_mean_6h"]) if pd.notna(row["AQI_rolling_mean_6h"]) else 0.0
    }

api_data = get_api_data(row)

# Call API for prediction
if st.button("Predict AQI for Selected Row"):
    try:
        response = requests.post("http://localhost:8000/predict", json=api_data)
        if response.status_code == 200:
            pred = response.json()["predicted_aqi"]
            st.success(f"**Predicted AQI:** {pred:.2f}")
            st.info(f"**Actual AQI:** {row['AQI']:.2f}")
        else:
            st.error(f"API Error: {response.status_code}\n{response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")

# Plot actual AQI trend
st.write("### AQI Trend (Actual)")
st.line_chart(data["AQI"])

# --- Future AQI Forecast Section ---
st.write("## 🌟 3-Day AQI Forecast")
try:
    forecast = pd.read_csv("data/lahore_aqi_forecast_3days.csv")
    forecast["dt"] = pd.to_datetime(forecast["dt"])
    st.line_chart(
        data=forecast.set_index("dt")["AQI_predicted"],
        use_container_width=True,
    )
    # Highlight hazardous AQI
    hazardous = forecast[forecast["AQI_predicted"] >= 150]
    if not hazardous.empty:
        st.error(f"⚠️ Hazardous AQI expected at {hazardous['dt'].iloc[0]}: {hazardous['AQI_predicted'].iloc[0]:.1f}")
    st.dataframe(forecast[["dt", "AQI_predicted"]], hide_index=True)
except Exception as e:
    st.warning(f"Could not load forecast: {e}")

# Optionally, show more data or features
with st.expander("Show raw data"):
    st.dataframe(data) 