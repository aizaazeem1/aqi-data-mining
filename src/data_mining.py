import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.impute import SimpleImputer
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def prepare_data_for_mining(df):
    """Prepare data for mining tasks with enhanced features"""
    # Convert timestamp to datetime if it's not already
    if 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'])
        df['hour'] = df['dt'].dt.hour
        df['day_of_week'] = df['dt'].dt.dayofweek
        df['month'] = df['dt'].dt.month
        df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)
    
    # Create binary AQI classification (Good vs Not Good)
    df['AQI_Binary'] = (df['AQI_PM25'] <= 50).astype(int)
    
    # Create more granular weather categories
    df['Temp_Category'] = pd.cut(df['temp'], 
                                bins=[-float('inf'), 10, 20, 25, 30, 35, float('inf')],
                                labels=['Very Cold', 'Cold', 'Cool', 'Warm', 'Hot', 'Very Hot'])
    
    df['Humidity_Category'] = pd.cut(df['rhum'],
                                    bins=[-float('inf'), 30, 50, 70, 90, float('inf')],
                                    labels=['Very Dry', 'Dry', 'Moderate', 'Humid', 'Very Humid'])
    
    df['Wind_Category'] = pd.cut(df['wspd'],
                                bins=[-float('inf'), 2, 5, 10, 15, float('inf')],
                                labels=['Calm', 'Light', 'Moderate', 'Strong', 'Very Strong'])
    
    # Create binary features for weather conditions
    df['High_Temp'] = (df['temp'] > df['temp'].mean()).astype(int)
    df['High_Humidity'] = (df['rhum'] > df['rhum'].mean()).astype(int)
    df['High_Wind'] = (df['wspd'] > df['wspd'].mean()).astype(int)
    
    # Add rolling averages
    df['AQI_rolling_24h'] = df['AQI_PM25'].rolling(window=24).mean()
    df['Temp_rolling_24h'] = df['temp'].rolling(window=24).mean()
    
    # Add weather change features
    df['Temp_change_1h'] = df['temp'].diff()
    df['Humidity_change_1h'] = df['rhum'].diff()
    df['Wind_change_1h'] = df['wspd'].diff()
    
    return df

def perform_association_rule_mining(df):
    """Perform association rule mining with lower support threshold"""
    # Create binary features for mining
    mining_df = df[['High_Temp', 'High_Humidity', 'High_Wind', 
                    'Temp_Category', 'Humidity_Category', 'Wind_Category',
                    'AQI_Binary']].copy()
    
    # Convert categorical to binary
    mining_df = pd.get_dummies(mining_df)
    
    # Generate frequent itemsets with lower support
    frequent_itemsets = apriori(mining_df, min_support=0.05, use_colnames=True)
    
    # Generate rules with higher lift threshold
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    
    return rules

def perform_classification(df):
    """Perform binary classification using Random Forest"""
    # Prepare features
    feature_columns = ['temp', 'rhum', 'wspd', 'pres', 'hour', 'day_of_week', 
                      'month', 'season', 'AQI_rolling_24h', 'Temp_rolling_24h',
                      'Temp_change_1h', 'Humidity_change_1h', 'Wind_change_1h']
    
    X = df[feature_columns]
    y = df['AQI_Binary']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features using RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred),
        'feature_importance': feature_importance,
        'cv_scores': cv_scores
    }

def perform_clustering(df):
    """Perform clustering using DBSCAN and Hierarchical clustering"""
    # Prepare features for clustering
    features = ['temp', 'rhum', 'wspd', 'pres', 'AQI_PM25', 
                'hour', 'day_of_week', 'month']
    X = df[features].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale features using RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Try Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=4)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    # Calculate silhouette scores
    dbscan_score = silhouette_score(X_scaled, dbscan_labels) if len(np.unique(dbscan_labels)) > 1 else 0
    hierarchical_score = silhouette_score(X_scaled, hierarchical_labels)
    
    # Use the better performing clustering
    if dbscan_score > hierarchical_score:
        df['Cluster'] = dbscan_labels
        best_score = dbscan_score
        best_method = 'DBSCAN'
    else:
        df['Cluster'] = hierarchical_labels
        best_score = hierarchical_score
        best_method = 'Hierarchical'
    
    # Create cluster profiles
    cluster_profiles = df.groupby('Cluster')[features].mean()
    
    return {
        'silhouette_score': best_score,
        'cluster_profiles': cluster_profiles,
        'cluster_assignments': df['Cluster'],
        'method': best_method
    }

def generate_visualizations(df, rules, cluster_results):
    """Generate enhanced visualizations for the data mining results"""
    # Create directory for plots if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Set style for better visualizations
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # 1. Association Rules Visualization
    # 1.1 Support vs Confidence with Lift
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(rules['support'], rules['confidence'], 
                alpha=0.6, c=rules['lift'], cmap='viridis', s=100)
    plt.colorbar(scatter, label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Association Rules: Support vs Confidence (colored by Lift)')
    plt.savefig('plots/association_rules_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1.2 Top 10 Rules by Lift
    plt.figure(figsize=(15, 8))
    top_rules = rules.nlargest(10, 'lift')
    sns.barplot(data=top_rules, x='lift', y=top_rules.index)
    plt.title('Top 10 Association Rules by Lift')
    plt.xlabel('Lift')
    plt.ylabel('Rule Index')
    plt.savefig('plots/top_rules_lift.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Clustering Visualizations
    # 2.1 Temperature vs AQI by Cluster
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['temp'], df['AQI_PM25'], 
                         c=cluster_results['cluster_assignments'], 
                         cmap='viridis', alpha=0.6, s=100)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('AQI (PM2.5)')
    plt.title(f'Weather-AQI Clusters ({cluster_results["method"]})')
    plt.colorbar(scatter, label='Cluster')
    
    # Add cluster centroids
    centroids = cluster_results['cluster_profiles']
    plt.scatter(centroids['temp'], centroids['AQI_PM25'], 
                c='red', marker='x', s=200, linewidths=3, 
                label='Cluster Centroids')
    plt.legend()
    plt.savefig('plots/clusters_temp_aqi.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2.2 Cluster Profiles Heatmap
    plt.figure(figsize=(15, 8))
    # Normalize the cluster profiles for better visualization
    profiles_norm = (cluster_results['cluster_profiles'] - 
                    cluster_results['cluster_profiles'].mean()) / \
                    cluster_results['cluster_profiles'].std()
    sns.heatmap(profiles_norm, annot=True, cmap='RdYlBu_r', 
                fmt='.2f', center=0)
    plt.title('Normalized Cluster Profiles Heatmap')
    plt.savefig('plots/cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. AQI and Weather Patterns
    # 3.1 AQI Distribution
    plt.figure(figsize=(12, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Binary AQI Distribution
    sns.countplot(data=df, x='AQI_Binary', ax=ax1)
    ax1.set_title('Distribution of AQI Categories')
    ax1.set_xticklabels(['Not Good', 'Good'])
    
    # AQI PM2.5 Distribution
    sns.histplot(data=df, x='AQI_PM25', bins=50, ax=ax2)
    ax2.set_title('Distribution of AQI PM2.5 Values')
    ax2.set_xlabel('AQI PM2.5')
    
    plt.tight_layout()
    plt.savefig('plots/aqi_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.2 Weather Features Correlation
    plt.figure(figsize=(12, 10))
    weather_features = ['temp', 'rhum', 'wspd', 'pres', 'AQI_PM25', 
                       'AQI_rolling_24h', 'Temp_rolling_24h']
    correlation = df[weather_features].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation between Weather Features and AQI')
    plt.savefig('plots/weather_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.3 Time Patterns
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Hour of day vs AQI
    sns.boxplot(data=df, x='hour', y='AQI_PM25', ax=axes[0,0])
    axes[0,0].set_title('AQI Distribution by Hour of Day')
    axes[0,0].set_xlabel('Hour')
    axes[0,0].set_ylabel('AQI PM2.5')
    
    # Day of week vs AQI
    sns.boxplot(data=df, x='day_of_week', y='AQI_PM25', ax=axes[0,1])
    axes[0,1].set_title('AQI Distribution by Day of Week')
    axes[0,1].set_xlabel('Day of Week')
    axes[0,1].set_ylabel('AQI PM2.5')
    
    # Month vs AQI
    sns.boxplot(data=df, x='month', y='AQI_PM25', ax=axes[1,0])
    axes[1,0].set_title('AQI Distribution by Month')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('AQI PM2.5')
    
    # Season vs AQI
    sns.boxplot(data=df, x='season', y='AQI_PM25', ax=axes[1,1])
    axes[1,1].set_title('AQI Distribution by Season')
    axes[1,1].set_xlabel('Season')
    axes[1,1].set_ylabel('AQI PM2.5')
    
    plt.tight_layout()
    plt.savefig('plots/temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.4 Weather Categories vs AQI
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    sns.boxplot(data=df, x='Temp_Category', y='AQI_PM25', ax=axes[0])
    axes[0].set_title('AQI by Temperature Category')
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='Humidity_Category', y='AQI_PM25', ax=axes[1])
    axes[1].set_title('AQI by Humidity Category')
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='Wind_Category', y='AQI_PM25', ax=axes[2])
    axes[2].set_title('AQI by Wind Category')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/weather_categories_aqi.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run all data mining tasks"""
    # Load the data
    df = pd.read_csv('data/lahore_features_with_aqi.csv')
    
    # Prepare data
    df = prepare_data_for_mining(df)
    
    # Perform data mining tasks
    rules = perform_association_rule_mining(df)
    classification_results = perform_classification(df)
    cluster_results = perform_clustering(df)
    
    # Generate visualizations
    generate_visualizations(df, rules, cluster_results)
    
    # Save results
    rules.to_csv('data/association_rules.csv', index=False)
    cluster_results['cluster_profiles'].to_csv('data/cluster_profiles.csv')
    
    # Print summary
    print("\nData Mining Results Summary:")
    print("\n1. Association Rules:")
    print(f"Number of rules found: {len(rules)}")
    print("\nTop 3 rules by lift:")
    print(rules.sort_values('lift', ascending=False).head(3))
    
    print("\n2. Classification Results:")
    print(f"Accuracy: {classification_results['accuracy']:.3f}")
    print("\nCross-validation scores:")
    print(f"Mean: {classification_results['cv_scores'].mean():.3f} (+/- {classification_results['cv_scores'].std() * 2:.3f})")
    print("\nTop 5 important features:")
    print(classification_results['feature_importance'].head())
    
    print("\n3. Clustering Results:")
    print(f"Method: {cluster_results['method']}")
    print(f"Silhouette Score: {cluster_results['silhouette_score']:.3f}")
    print("\nCluster Profiles:")
    print(cluster_results['cluster_profiles'])

if __name__ == "__main__":
    main() 