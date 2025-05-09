import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_mining import (
    prepare_data_for_mining,
    perform_association_rule_mining,
    perform_classification,
    perform_clustering
)

# Set page config
st.set_page_config(
    page_title="AQI Data Mining Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("AQI Data Mining Dashboard")
st.markdown("""
This dashboard provides insights into the relationships between weather conditions and air quality in Lahore.
Explore the patterns, correlations, and predictions through various visualizations and analysis.
""")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('data/lahore_features_with_aqi.csv')
    df = prepare_data_for_mining(df)
    return df

try:
    df = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis",
        ["Overview", "Classification", "Clustering", "Association Rules", "Temporal Patterns"]
    )
    
    if page == "Overview":
        st.header("Data Overview")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("AQI Statistics")
            st.write(df['AQI_PM25'].describe())
        
        with col2:
            st.subheader("Weather Statistics")
            weather_stats = df[['temp', 'rhum', 'wspd', 'pres']].describe()
            st.write(weather_stats)
        
        # AQI Distribution
        st.subheader("AQI Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='AQI_PM25', bins=50)
        plt.title('Distribution of AQI PM2.5 Values')
        st.pyplot(fig)
        
        # Weather Correlations
        st.subheader("Weather Feature Correlations")
        weather_features = ['temp', 'rhum', 'wspd', 'pres', 'AQI_PM25']
        correlation = df[weather_features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        st.pyplot(fig)
    
    elif page == "Classification":
        st.header("Classification Analysis")
        
        # Run classification
        classification_results = perform_classification(df)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Performance")
            st.write(f"Accuracy: {classification_results['accuracy']:.3f}")
            st.write("Cross-validation scores:")
            st.write(f"Mean: {classification_results['cv_scores'].mean():.3f} (+/- {classification_results['cv_scores'].std() * 2:.3f})")
        
        with col2:
            st.subheader("Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            importance_df = classification_results['feature_importance'].head(10)
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Top 10 Important Features')
            st.pyplot(fig)
    
    elif page == "Clustering":
        st.header("Clustering Analysis")
        
        # Run clustering
        cluster_results = perform_clustering(df)
        
        # Display results
        st.write(f"Clustering Method: {cluster_results['method']}")
        st.write(f"Silhouette Score: {cluster_results['silhouette_score']:.3f}")
        
        # Cluster profiles
        st.subheader("Cluster Profiles")
        st.write(cluster_results['cluster_profiles'])
        
        # Cluster visualization
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = plt.scatter(df['temp'], df['AQI_PM25'], 
                            c=cluster_results['cluster_assignments'], 
                            cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('AQI (PM2.5)')
        plt.title('Weather-AQI Clusters')
        st.pyplot(fig)
    
    elif page == "Association Rules":
        st.header("Association Rules Analysis")
        
        # Run association rule mining
        rules = perform_association_rule_mining(df)
        
        # Display results
        st.write(f"Number of rules found: {len(rules)}")
        
        # Top rules
        st.subheader("Top Rules by Lift")
        top_rules = rules.nlargest(10, 'lift')
        st.write(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        
        # Rule visualization
        st.subheader("Rule Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = plt.scatter(rules['support'], rules['confidence'], 
                            alpha=0.6, c=rules['lift'], cmap='viridis')
        plt.colorbar(scatter, label='Lift')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Association Rules: Support vs Confidence')
        st.pyplot(fig)
    
    elif page == "Temporal Patterns":
        st.header("Temporal Patterns Analysis")
        
        # Create temporal visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Hourly Patterns")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x='hour', y='AQI_PM25')
            plt.title('AQI Distribution by Hour')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Monthly Patterns")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x='month', y='AQI_PM25')
            plt.title('AQI Distribution by Month')
            st.pyplot(fig)
        
        # Weather categories
        st.subheader("Weather Categories")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        sns.boxplot(data=df, x='Temp_Category', y='AQI_PM25', ax=axes[0])
        axes[0].set_title('AQI by Temperature Category')
        axes[0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=df, x='Humidity_Category', y='AQI_PM25', ax=axes[1])
        axes[1].set_title('AQI by Humidity Category')
        axes[1].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=df, x='Wind_Category', y='AQI_PM25', ax=axes[2])
        axes[2].set_title('AQI by Wind Category')
        axes[2].tick_params(axis='x', rotation=45)
        
        st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please check if the data file exists and has the correct format.") 