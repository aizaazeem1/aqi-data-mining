import requests
import pandas as pd
import json

# Read the latest data point from our dataset
df = pd.read_csv('data/lahore_features_with_aqi.csv')
latest_data = df.iloc[-1].to_dict()

# Prepare the data for API request
api_data = {
    "co": float(latest_data['co']),
    "no": float(latest_data['no']),
    "no2": float(latest_data['no2']),
    "o3": float(latest_data['o3']),
    "so2": float(latest_data['so2']),
    "pm2_5": float(latest_data['pm2_5']),
    "pm10": float(latest_data['pm10']),
    "temp": float(latest_data['temp']),
    "rhum": float(latest_data['rhum']),
    "pres": float(latest_data['pres']),
    "wspd": float(latest_data['wspd']),
    "hour": int(latest_data['hour']),
    "day": int(latest_data['day']),
    "weekday": int(latest_data['weekday']),
    "month": int(latest_data['month']),
    "AQI_change_1h": float(latest_data['AQI_change_1h']) if pd.notna(latest_data['AQI_change_1h']) else 0.0,
    "AQI_rolling_mean_6h": float(latest_data['AQI_rolling_mean_6h']) if pd.notna(latest_data['AQI_rolling_mean_6h']) else 0.0
}

# Print the actual AQI for comparison
print(f"Actual AQI: {latest_data['AQI']}")

# Make the API request
response = requests.post('http://localhost:8000/predict', json=api_data)

# Print the prediction
if response.status_code == 200:
    prediction = response.json()
    print(f"Predicted AQI: {prediction['predicted_aqi']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text) 