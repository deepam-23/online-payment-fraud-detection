import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    if df is None:
        return None
    
    # Check if 'transaction_hour' column exists
    if 'transaction_hour' not in df.columns:
        df['transaction_hour'] = 0  # Default value if column is missing
    
    # Extract time features
    df['hour'] = df['transaction_hour'].astype(float)
    
    # Calculate transaction frequency by user
    user_transaction_count = df.groupby('user_id').size().reset_index(name='user_transaction_count')
    df = df.merge(user_transaction_count, on='user_id', how='left')
    
    # Check if 'transaction_amount' column exists
    if 'transaction_amount' not in df.columns:
        df['transaction_amount'] = 0  # Default value if column is missing
    
    # Calculate transaction amount statistics by user
    user_amount_stats = df.groupby('user_id')['transaction_amount'].agg(['mean', 'std']).reset_index()
    user_amount_stats.columns = ['user_id', 'user_mean_amount', 'user_std_amount']
    df = df.merge(user_amount_stats, on='user_id', how='left')
    
    # Fill NaN values in std with 0 (for users with only one transaction)
    df['user_std_amount'] = df['user_std_amount'].fillna(0)
    
    # Calculate amount deviation from user mean
    df['amount_deviation'] = abs(df['transaction_amount'] - df['user_mean_amount'])
    
    # One-hot encode app categories
    if 'app' in df.columns:
        df['app_category'] = df['app'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'Unknown')
        app_dummies = pd.get_dummies(df['app_category'], prefix='app')
        df = pd.concat([df, app_dummies], axis=1)
    else:
        df['app_category'] = 'Unknown'
        app_dummies = pd.get_dummies(df['app_category'], prefix='app')
        df = pd.concat([df, app_dummies], axis=1)
    
    # Check if 'location_count' column exists
    if 'location_count' not in df.columns:
        df['location_count'] = 0  # Default value if column is missing
    
    # Location features
    df['is_new_location'] = (df['location_count'] <= 1).astype(int)
    
    # Time pattern features
    df['unusual_hour'] = ((df['hour'] < 6) | (df['hour'] > 23)).astype(int)
    
    # Check and add missing columns with default values
    default_columns = [
        'time_spent_on_page', 'past_day_frequency', 
        'past_week_frequency', 'day_of_week'
    ]
    for col in default_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Prepare features
    feature_cols = [
        'transaction_amount', 'hour', 'day_of_week', 'location_count',
        'time_spent_on_page', 'past_day_frequency', 'past_week_frequency',
        'user_transaction_count', 'amount_deviation', 'is_new_location', 'unusual_hour'
    ]
    
    # Add app dummies to features
    app_dummy_cols = [col for col in df.columns if col.startswith('app_')]
    feature_cols.extend(app_dummy_cols)
    
    # Return processed dataframe and feature columns
    return df, feature_cols

def train_model(df, feature_cols):
    if df is None:
        return None
    
    # Select and prepare features
    X = df[feature_cols]
    y = df['is_fraud']
    
    # Convert all feature columns to numeric, replacing non-numeric values with 0
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Impute any remaining missing values
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Return model and scaler
    return model, scaler, feature_cols
