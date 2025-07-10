import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import hashlib

def generate_upi_dataset(num_records=10000, fraud_ratio=0.2):
    """
    Generate a synthetic UPI transaction dataset with both legitimate and fraudulent transactions.
    
    Parameters:
    -----------
    num_records : int
        Number of records to generate
    fraud_ratio : float
        Ratio of fraudulent transactions (between 0 and 1)
    
    Returns:
    --------
    pandas.DataFrame
        Generated dataset
    """
    # Calculate number of fraudulent transactions
    num_fraud = int(num_records * fraud_ratio)
    num_legitimate = num_records - num_fraud
    
    # Lists to store data
    data = []
    
    # Common UPI apps
    upi_apps = ['Google Pay', 'PhonePe', 'Paytm', 'BHIM', 'Amazon Pay', 'WhatsApp Pay']
    
    # Common transaction types
    transaction_types = ['P2P', 'P2M', 'Bill Payment', 'Recharge', 'E-commerce']
    
    # Common merchants
    merchants = [None, 'Amazon', 'Flipkart', 'Swiggy', 'Zomato', 'BigBasket', 'Myntra', 
                'Reliance Digital', 'Airtel', 'Jio', 'TATA Power', 'BSES', 'Movie Tickets']
    
    # Device types
    device_types = ['Android', 'iOS']
    
    # Operating system versions
    android_versions = ['10', '11', '12', '13', '14']
    ios_versions = ['15', '16', '17']
    
    # Start date for transaction timestamps
    start_date = datetime.now() - timedelta(days=180)
    
    # Generate legitimate transactions
    for i in range(num_legitimate):
        # More consistent user behavior for legitimate transactions
        user_id = f"user_{random.randint(1, int(num_legitimate/10))}"
        device_id = f"device_{random.randint(1, int(num_legitimate/8))}"
        
        # Use hash to create consistent UPI ID for same user
        hash_obj = hashlib.md5(user_id.encode())
        upi_base = hash_obj.hexdigest()[:8]
        upi_id = f"{upi_base}@{random.choice(['oksbi', 'okicici', 'okhdfc', 'okaxis', 'yesbank'])}"
        
        # Device info
        device_type = random.choice(device_types)
        if device_type == 'Android':
            os_version = random.choice(android_versions)
        else:
            os_version = random.choice(ios_versions)
            
        # Transaction details
        app = random.choice(upi_apps)
        transaction_type = random.choice(transaction_types)
        
        # Merchant is None for P2P transactions
        merchant = None if transaction_type == 'P2P' else random.choice(merchants[1:])
        
        # Amount - legitimate transactions usually have more reasonable amounts
        if transaction_type == 'P2P':
            amount = round(random.uniform(100, 5000), 2)
        elif transaction_type == 'Recharge':
            amount = round(random.choice([199, 249, 299, 349, 399, 499, 699, 999]), 2)
        elif transaction_type == 'Bill Payment':
            amount = round(random.uniform(500, 3000), 2)
        else:
            amount = round(random.uniform(200, 10000), 2)
            
        # Time features
        transaction_date = start_date + timedelta(days=random.randint(0, 180), 
                                                hours=random.randint(0, 23),
                                                minutes=random.randint(0, 59))
        hour = transaction_date.hour
        day_of_week = transaction_date.weekday()
        
        # Location - most legitimate transactions come from a few familiar locations
        location = f"Location_{random.randint(1, 5)}" if random.random() < 0.9 else f"Location_{random.randint(6, 20)}"
        
        # Transaction success rate high for legitimate
        status = 'SUCCESS' if random.random() < 0.95 else 'FAILURE'
        
        # User behavioral patterns
        login_attempts = 1 if random.random() < 0.95 else random.randint(2, 3)
        time_spent_on_app = round(random.uniform(30, 300), 2)  # seconds
        
        # Receiver for P2P
        receiver_upi = None
        if transaction_type == 'P2P':
            receiver_hash = hashlib.md5(f"receiver_{random.randint(1, 1000)}".encode())
            receiver_upi = f"{receiver_hash.hexdigest()[:8]}@{random.choice(['oksbi', 'okicici', 'okhdfc', 'okaxis', 'yesbank'])}"
        
        # Frequency features
        past_day_transactions = random.randint(0, 3)
        past_week_transactions = past_day_transactions + random.randint(0, 10)
        
        data.append({
            'transaction_id': f"TR{i+1:08d}",
            'user_id': user_id,
            'upi_id': upi_id,
            'device_id': device_id,
            'device_type': device_type,
            'os_version': os_version,
            'app': app,
            'transaction_type': transaction_type,
            'merchant': merchant,
            'receiver_upi': receiver_upi,
            'amount': amount,
            'transaction_datetime': transaction_date,
            'hour': hour,
            'day_of_week': day_of_week,
            'location': location,
            'login_attempts': login_attempts,
            'time_spent_on_app': time_spent_on_app,
            'past_day_transactions': past_day_transactions,
            'past_week_transactions': past_week_transactions,
            'status': status,
            'is_fraud': 0
        })
    
    # Generate fraudulent transactions
    for i in range(num_fraud):
        # More random user behavior for fraudulent transactions
        user_id = f"user_{random.randint(1, num_records)}"  # More spread out
        device_id = f"device_{random.randint(1, num_records)}"  # More random devices
        
        # UPI ID generation - less consistent patterns
        upi_id = f"{random.choice('abcdefghijklmnopqrstuvwxyz')}{random.randint(100, 999)}@{random.choice(['oksbi', 'okicici', 'okhdfc', 'okaxis', 'yesbank'])}"
        
        # Device info - more varied
        device_type = random.choice(device_types)
        if device_type == 'Android':
            os_version = random.choice(android_versions)
        else:
            os_version = random.choice(ios_versions)
            
        # Transaction details
        app = random.choice(upi_apps)
        transaction_type = random.choice(transaction_types)
        
        # Merchant is None for P2P transactions
        merchant = None if transaction_type == 'P2P' else random.choice(merchants[1:])
        
        # Amount - fraudulent transactions often have unusual amounts
        # Either very small test transactions or large transfers
        if random.random() < 0.3:
            amount = round(random.uniform(1, 10), 2)  # Small test amounts
        else:
            amount = round(random.uniform(10000, 50000), 2)  # Large amounts
            
        # Time features - more likely during night hours
        transaction_date = start_date + timedelta(days=random.randint(0, 180), 
                                               hours=random.randint(0, 23),
                                               minutes=random.randint(0, 59))
        
        # More likely to be at odd hours (night)
        if random.random() < 0.7:
            hour = random.randint(22, 23) if random.random() < 0.5 else random.randint(0, 4)
        else:
            hour = random.randint(5, 21)
            
        transaction_date = transaction_date.replace(hour=hour)
        day_of_week = transaction_date.weekday()
        
        # Location - fraudulent transactions often from unusual locations
        location = f"Location_{random.randint(21, 100)}"
        
        # Status - fraudulent transactions may fail more often
        status = 'SUCCESS' if random.random() < 0.7 else 'FAILURE'
        
        # User behavioral patterns - suspicious patterns
        login_attempts = random.randint(3, 10) if random.random() < 0.6 else 1
        time_spent_on_app = round(random.uniform(5, 30), 2) if random.random() < 0.7 else round(random.uniform(30, 300), 2)
        
        # Receiver for P2P - often new receivers
        receiver_upi = None
        if transaction_type == 'P2P':
            receiver_upi = f"{random.choice('abcdefghijklmnopqrstuvwxyz')}{random.randint(100, 999)}@{random.choice(['oksbi', 'okicici', 'okhdfc', 'okaxis', 'yesbank'])}"
        
        # Frequency features - fraud often has burst patterns
        past_day_transactions = random.randint(5, 20) if random.random() < 0.6 else random.randint(0, 3)
        past_week_transactions = past_day_transactions + random.randint(0, 10)
        
        data.append({
            'transaction_id': f"TR{num_legitimate+i+1:08d}",
            'user_id': user_id,
            'upi_id': upi_id,
            'device_id': device_id,
            'device_type': device_type,
            'os_version': os_version,
            'app': app,
            'transaction_type': transaction_type,
            'merchant': merchant,
            'receiver_upi': receiver_upi,
            'amount': amount,
            'transaction_datetime': transaction_date,
            'hour': hour,
            'day_of_week': day_of_week,
            'location': location,
            'login_attempts': login_attempts,
            'time_spent_on_app': time_spent_on_app,
            'past_day_transactions': past_day_transactions,
            'past_week_transactions': past_week_transactions,
            'status': status,
            'is_fraud': 1
        })
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Generate the dataset
df = generate_upi_dataset(num_records=10000, fraud_ratio=0.2)

# Save to CSV
df.to_csv('upi_transactions.csv', index=False)

print(f"Dataset generated with {len(df)} records ({df['is_fraud'].sum()} fraudulent transactions)")
print(f"Dataset saved to 'upi_transactions.csv'")
