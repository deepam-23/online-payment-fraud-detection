import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    if df is None:
        return None
    
    # Label encoding for categorical variables
    categorical_columns = [
        'user_id', 'upi_id', 'device_id', 'device_type', 'os_version', 
        'app', 'transaction_type', 'merchant', 'receiver_upi', 'location'
    ]
    
    label_encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
            label_encoders[col] = le
    
    # Parse datetime
    if 'transaction_datetime' in df.columns:
        df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
        df['hour'] = df['transaction_datetime'].dt.hour
        df['day_of_week'] = df['transaction_datetime'].dt.dayofweek
    
    # Feature engineering
    df['login_ratio'] = df['login_attempts'] / (df['past_day_transactions'] + 1)
    df['transaction_frequency_ratio'] = df['past_week_transactions'] / (df['past_day_transactions'] + 1)
    
    # Calculate transaction amount statistics
    user_amount_stats = df.groupby('user_id')['amount'].agg(['mean', 'std']).reset_index()
    user_amount_stats.columns = ['user_id', 'user_mean_amount', 'user_std_amount']
    df = df.merge(user_amount_stats, on='user_id', how='left')
    
    # Deviation from user's mean transaction amount
    df['amount_deviation'] = abs(df['amount'] - df['user_mean_amount'])
    
    # One-hot encode app and transaction type
    app_dummies = pd.get_dummies(df['app'], prefix='app')
    transaction_type_dummies = pd.get_dummies(df['transaction_type'], prefix='type')
    df = pd.concat([df, app_dummies, transaction_type_dummies], axis=1)
    
    # Prepare feature columns
    feature_cols = [
        'amount', 'hour', 'day_of_week', 'time_spent_on_app', 
        'past_day_transactions', 'past_week_transactions', 
        'login_attempts', 'login_ratio', 'transaction_frequency_ratio', 
        'amount_deviation'
    ]
    
    # Add encoded categorical features
    encoded_categorical_cols = [f'{col}_encoded' for col in categorical_columns if f'{col}_encoded' in df.columns]
    feature_cols.extend(encoded_categorical_cols)
    
    # Add dummy columns
    feature_cols.extend(list(app_dummies.columns))
    feature_cols.extend(list(transaction_type_dummies.columns))
    
    return df, feature_cols, label_encoders

def train_model(df, feature_cols):
    if df is None:
        return None
    
    # Ensure 'is_fraud' column exists
    if 'is_fraud' not in df.columns:
        print("Warning: 'is_fraud' column not found. Using random fraud labels.")
        df['is_fraud'] = np.random.randint(0, 2, size=len(df))
    
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
    
    # Calculate model performance
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
    return model, scaler, feature_cols
