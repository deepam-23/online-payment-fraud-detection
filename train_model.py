from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

# Generate dummy data
X = np.random.rand(200, 6)
y = np.random.randint(0, 2, 200)

# Train simple model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'model/fraud_model.pkl')

print("✅ Model trained and saved successfully.")
