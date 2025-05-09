import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from meteostat import Hourly, Point
# from hopsworks import login  # Uncomment if using Hopsworks

def compute_aqi(conc, breakpoints):
    for (Clow, Chigh, Ilow, Ihigh) in breakpoints:
        if Clow <= conc <= Chigh:
            return round(((Ihigh - Ilow)/(Chigh - Clow)) * (conc - Clow) + Ilow)
    return None

def main():
    model = joblib.load('data/lr_model.pkl')
    feature_columns = joblib.load('data/feature_columns.pkl')
    now = datetime.utcnow()
    future_times = pd.date_range(start=now, periods=72, freq='h')
    lahore = Point(31.5497, 74.3436)
    weather = Hourly(lahore, now, now + timedelta(hours=72))
    forecast_weather = weather.fetch().reset_index()
    forecast_weather.rename(columns={'time': 'dt'}, inplace=True)
    future_df = pd.DataFrame({'dt': future_times})
    future_df['dt'] = future_df['dt'].dt.floor('h')
    forecast_weather['dt'] = pd.to_datetime(forecast_weather['dt']).dt.floor('h')
    future_df = pd.merge(future_df, forecast_weather, on='dt', how='left')
    future_df['hour'] = future_df['dt'].dt.hour
    future_df['day'] = future_df['dt'].dt.day
    future_df['weekday'] = future_df['dt'].dt.weekday
    future_df['month'] = future_df['dt'].dt.month
    hist = pd.read_csv('data/lahore_features_with_aqi.csv')
    pollutant_cols = ['pm2_5', 'pm10', 'no', 'no2', 'o3', 'so2', 'co']
    recent_avg = hist[pollutant_cols].tail(24).mean()
    for col in pollutant_cols:
        noise = np.random.normal(0, recent_avg[col] * 0.05, size=len(future_df))
        future_df[col] = recent_avg[col] + noise
        future_df[col] = future_df[col].clip(lower=0)
    weather_cols = ['temp', 'rhum', 'wspd', 'pres']
    for col in weather_cols:
        if col in future_df.columns:
            if future_df[col].isnull().any():
                future_df[col] = future_df[col].fillna(hist[col].tail(24).mean())
    pm25_bp = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)
    ]
    pm10_bp = [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500)
    ]
    future_df['AQI_PM25'] = future_df['pm2_5'].apply(lambda x: compute_aqi(x, pm25_bp))
    future_df['AQI_PM10'] = future_df['pm10'].apply(lambda x: compute_aqi(x, pm10_bp))
    future_df['AQI'] = future_df[['AQI_PM25', 'AQI_PM10']].max(axis=1)
    future_df['AQI_change_1h'] = 0
    future_df['AQI_rolling_mean_6h'] = hist['AQI'].rolling(6).mean().iloc[-1]
    for col in feature_columns:
        if col not in future_df.columns:
            future_df[col] = 0
    X_future = future_df[feature_columns].fillna(0)
    future_df['AQI_predicted'] = model.predict(X_future)
    future_df.to_csv('data/lahore_aqi_forecast_3days.csv', index=False)
    print('✅ Forecast saved to data/lahore_aqi_forecast_3days.csv')

    # --- Hopsworks Upload ---
    import hopsworks
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if api_key:
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        feature_group = fs.get_or_create_feature_group(
            name="lahore_aqi_features",
            version=1,
            description="Lahore AQI features with weather and pollution data",
            primary_key=["dt"],
            online_enabled=True
        )
        feature_group.insert(future_df, write_options={"wait_for_job": True})
        print("✅ Features uploaded to Hopsworks!")
    else:
        print("⚠️ HOPSWORKS_API_KEY not found in .env. Skipping Hopsworks upload.")

if __name__ == '__main__':
    main()