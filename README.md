# AQI Data Mining Dashboard

This Streamlit dashboard provides insights into the relationships between weather conditions and air quality in Lahore. It includes various data mining analyses and visualizations to understand patterns and correlations.

## Features

- **Overview**: Basic statistics and correlations between weather features and AQI
- **Classification**: Binary classification of AQI levels with feature importance
- **Clustering**: Weather-AQI pattern clustering with silhouette analysis
- **Association Rules**: Mining of weather-AQI relationships
- **Temporal Patterns**: Analysis of AQI patterns across time and weather categories

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aqi-data-mining.git
cd aqi-data-mining
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run data_mining_dashboard.py
```

## Data

The dashboard uses weather and AQI data from Lahore, Pakistan. The data includes:
- Temperature
- Relative humidity
- Wind speed
- Pressure
- AQI (PM2.5)

## Technologies Used

- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- MLxtend

## License

MIT License 