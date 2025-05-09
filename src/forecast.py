import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from meteostat import Hourly, Point

def main():
    # Load model and feature columns
    model = joblib.load('data/rf_model.pkl')
    feature_columns = joblib.load('data/feature_columns.pkl')

    # Generate next 72 hourly timestamps
    now = datetime.utcnow()
    future_times = pd.date_range(start=now, periods=72, freq='h')

    # Get weather forecast for Lahore
    lahore = Point(31.5497, 74.3436)
    weather = Hourly(lahore, now, now + timedelta(hours=72))
    forecast_weather = weather.fetch().reset_index()
    forecast_weather.rename(columns={'time': 'dt'}, inplace=True)
    print('Fetched weather columns:', forecast_weather.columns.tolist())

    # Create DataFrame with timestamps (rounded to hour, no microseconds)
    future_df = pd.DataFrame({'dt': future_times})
    future_df['dt'] = future_df['dt'].dt.floor('h')

    # Also round Meteostat times to hour
    forecast_weather['dt'] = pd.to_datetime(forecast_weather['dt']).dt.floor('h')

    # Merge on the rounded hour
    future_df = pd.merge(future_df, forecast_weather, on='dt', how='left')
    print('After merge, sample rows:')
    print(future_df[['dt', 'temp', 'rhum', 'wspd', 'pres']].head(10))

    # Add time features
    future_df['hour'] = future_df['dt'].dt.hour
    future_df['day'] = future_df['dt'].dt.day
    future_df['weekday'] = future_df['dt'].dt.weekday
    future_df['month'] = future_df['dt'].dt.month

    # Approximate pollutants with recent averages + random noise for realism
    hist = pd.read_csv('data/lahore_features_with_aqi.csv')
    pollutant_cols = ['pm2_5', 'pm10', 'no', 'no2', 'o3', 'so2', 'co']
    recent_avg = hist[pollutant_cols].tail(24).mean()
    for col in pollutant_cols:
        noise = np.random.normal(0, recent_avg[col] * 0.05, size=len(future_df))
        future_df[col] = recent_avg[col] + noise
        future_df[col] = future_df[col].clip(lower=0)

    # Fill missing weather columns with recent historical averages
    weather_cols = ['temp', 'rhum', 'wspd', 'pres']
    for col in weather_cols:
        if col in future_df.columns:
            if future_df[col].isnull().any():
                future_df[col] = future_df[col].fillna(hist[col].tail(24).mean())

    # Derived features
    future_df['AQI_change_1h'] = 0
    future_df['AQI_rolling_mean_6h'] = hist['AQI'].rolling(6).mean().iloc[-1]

    # Recompute AQI features for forecast
    def compute_aqi(conc, breakpoints):
        for (Clow, Chigh, Ilow, Ihigh) in breakpoints:
            if Clow <= conc <= Chigh:
                return round(((Ihigh - Ilow)/(Chigh - Clow)) * (conc - Clow) + Ilow)
        return None

    pm25_bp = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    pm10_bp = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ]
    future_df['AQI_PM25'] = future_df['pm2_5'].apply(lambda x: compute_aqi(x, pm25_bp))
    future_df['AQI_PM10'] = future_df['pm10'].apply(lambda x: compute_aqi(x, pm10_bp))
    future_df['AQI'] = future_df[['AQI_PM25', 'AQI_PM10']].max(axis=1)

    # Fill missing columns
    for col in feature_columns:
        if col not in future_df.columns:
            future_df[col] = 0
    print('Final features sample:')
    print(future_df[feature_columns].head())
    X_future = future_df[feature_columns].fillna(0)
    future_df['AQI_predicted'] = model.predict(X_future)
    future_df.to_csv('data/lahore_aqi_forecast_3days.csv', index=False)
    print('✅ Forecast saved to data/lahore_aqi_forecast_3days.csv')

if __name__ == '__main__':
    main() 