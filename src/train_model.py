import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import hopsworks
import os
from dotenv import load_dotenv
# from hopsworks import login  # Uncomment if using Hopsworks

def main():
    # --- Hopsworks Read ---
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if api_key:
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group("lahore_aqi_features", version=1)
        df = feature_group.read()
        print("✅ Features loaded from Hopsworks!")
    else:
        print("⚠️ HOPSWORKS_API_KEY not found in .env. Loading from CSV.")
        df = pd.read_csv('ShineAQI/data/lahore_features_with_aqi.csv')
    df = df[df['AQI'].notna()].ffill()
    features = [col for col in df.columns if col not in ['dt', 'AQI']]
    X = df[features].fillna(df[features].mean()).fillna(0)
    y = df['AQI']

    # Train-test split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    ridge = Ridge(alpha=1.0)
    rf.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    # Evaluate
    def evaluate(model, X_test, y_test):
        preds = model.predict(X_test)
        print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
        print("MAE:", mean_absolute_error(y_test, preds))
        print("R²:", r2_score(y_test, preds))
    print("Random Forest:")
    evaluate(rf, X_test, y_test)
    print("\nRidge Regression:")
    evaluate(ridge, X_test, y_test)

    # Save best model
    joblib.dump(rf, 'ShineAQI/data/rf_model.pkl')
    joblib.dump(list(X.columns), 'ShineAQI/data/feature_columns.pkl')
    print('✅ Model and feature columns saved.')

    # --- Upload to Hopsworks Model Registry ---
    if api_key:
        # Evaluate metrics for registry
        preds = rf.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        mr = project.get_model_registry()
        model_dir = "ShineAQI/data"  # Directory where your model is saved
        model = mr.python.create_model(
            name="rf_model",
            metrics={"rmse": rmse, "mae": mae, "r2": r2},
            description="Random Forest model for Lahore AQI prediction"
        )
        model.save(model_dir)
        print("✅ Model uploaded to Hopsworks Model Registry!")
    else:
        print("⚠️ HOPSWORKS_API_KEY not found in .env. Skipping model registry upload.")

if __name__ == '__main__':
    main() 