import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Initialize Flask app
app = Flask(__name__)

def create_sample_dataset():
    """Create a comprehensive sample dataset for UPI fraud detection"""
    np.random.seed(42)
    
    # Generate sample transaction data
    data = {
        'transaction_id': [f'TXN{i:05d}' for i in range(1000)],
        'user_id': [f'user{i:03d}' for i in range(1000)],
        'upi_id': [f'user{i:03d}@upi' for i in range(1000)],
        'device_type': np.random.choice(['mobile', 'tablet', 'desktop'], 1000),
        'app': np.random.choice(['Google Pay', 'PhonePe', 'Paytm', 'Amazon Pay'], 1000),
        'amount': np.random.uniform(100, 10000, 1000),
        'transaction_datetime': pd.date_range(start='2025-01-01', periods=1000),
        'location': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'], 1000),
        'login_attempts': np.random.randint(0, 6, 1000),
        'past_day_transactions': np.random.randint(0, 10, 1000),
        'past_week_transactions': np.random.randint(0, 20, 1000)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add fraud labels based on some synthetic rules
    df['is_fraud'] = (
        (df['amount'] > 5000) |  # High amount transactions
        (df['login_attempts'] > 4) |  # Multiple login attempts
        (df['past_day_transactions'] > 8) |  # Unusual transaction frequency
        (df['device_type'] == 'mobile') & (df['amount'] > 7000)
    ).astype(int)
    
    return df

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Separate features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Identify column types
    numeric_features = ['amount', 'login_attempts', 'past_day_transactions', 'past_week_transactions']
    categorical_features = ['device_type', 'app', 'location']
    
    # Custom function to extract datetime features
    def extract_datetime_features(X):
        X_ = X.copy()
        datetime_col = pd.to_datetime(X_['transaction_datetime'])
        X_['hour'] = datetime_col.dt.hour
        X_['day_of_week'] = datetime_col.dt.dayofweek
        X_['month'] = datetime_col.dt.month
        return X_.drop('transaction_datetime', axis=1)
    
    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create full preprocessing pipeline
    full_pipeline = Pipeline(steps=[
        ('datetime_extractor', FunctionTransformer(func=extract_datetime_features)),
        ('preprocessor', preprocessor)
    ])
    
    # Fit and transform
    X_processed = full_pipeline.fit_transform(X)
    
    return X_processed, y, full_pipeline

def train_fraud_model(X, y):
    """Train a Random Forest Classifier for fraud detection"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'
    )
    rf_classifier.fit(X_train, y_train)
    
    return rf_classifier

# Global model and preprocessing pipeline
sample_df = create_sample_dataset()
X_processed, y, preprocessing_pipeline = preprocess_data(sample_df)
fraud_model = train_fraud_model(X_processed, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-fraud', methods=['POST'])
def predict_fraud():
    try:
        # Get input data
        data = request.json
        
        # Convert to DataFrame
        input_df = pd.DataFrame([{
            'transaction_id': data.get('transaction_id', 'TXN00000'),
            'user_id': data.get('user_id', 'user000'),
            'upi_id': data.get('upi_id', 'user000@upi'),
            'device_type': data.get('device_type', 'mobile'),
            'app': data.get('app', 'Google Pay'),
            'amount': float(data.get('amount', 0)),
            'transaction_datetime': pd.to_datetime(data.get('transaction_datetime', pd.Timestamp.now())),
            'location': data.get('location', 'Mumbai'),
            'login_attempts': int(data.get('login_attempts', 0)),
            'past_day_transactions': int(data.get('past_day_transactions', 0)),
            'past_week_transactions': int(data.get('past_week_transactions', 0))
        }])
        
        # Preprocess input data
        X_input_processed = preprocessing_pipeline.transform(input_df)
        
        # Predict probability
        fraud_probability = fraud_model.predict_proba(X_input_processed)[0][1]
        is_fraud = fraud_probability > 0.5
        
        return jsonify({
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(is_fraud)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
