from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import os
import re
import csv
import numpy as np

app = Flask(__name__)
app.secret_key = "secret_key_123"

# --- Load Model ---
MODEL_PATH = "model/fraud_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print("❌ Failed to load model:", e)
else:
    print("⚠️ No model found — rule-based logic will be used.")

# --- Utility Functions ---
def valid_login(email: str, password: str) -> bool:
    """Check for valid email and password length."""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email)) and len(password) >= 4

def save_user(email: str):
    """Save user email to CSV (for record)."""
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "users.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["email"])
        writer.writerow([email])

def rule_based_fraud_check(tx: dict) -> (bool, str):
    """Fallback rule-based logic for fraud detection."""
    amt = float(tx.get("amount", 0))
    login_attempts = int(tx.get("login_attempts", 0))
    past_day = int(tx.get("past_day_transactions", 0))
    past_week = int(tx.get("past_week_transactions", 0))
    device = tx.get("device_type", "").lower()

    if amt > 50000 and login_attempts > 3:
        return True, "High amount with multiple login attempts"
    if past_day > 10 or past_week > 30:
        return True, "Unusually high transaction frequency"
    if device == "mobile" and amt > 30000:
        return True, "Large mobile transaction"
    if amt < 5:
        return True, "Suspiciously small test transaction"
    return False, "No suspicious behavior detected"

def model_predict(tx: dict):
    """Run prediction using ML model."""
    try:
        features = np.array([[ 
            float(tx.get("amount", 0)),
            float(tx.get("login_attempts", 0)),
            float(tx.get("past_day_transactions", 0)),
            float(tx.get("past_week_transactions", 0))
        ]])
        probs = model.predict_proba(features)
        fraud_prob = float(probs[0][1])
        return fraud_prob, None
    except Exception as e:
        return None, str(e)

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        if not valid_login(email, password):
            error = "Invalid email or password format."
        else:
            save_user(email)
            session["logged_in"] = True
            session["email"] = email
            return redirect(url_for("predict"))
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    result = None
    details = None

    if request.method == "POST":
        tx = {
            "device_type": request.form.get("device_type", ""),
            "app": request.form.get("app", ""),
            "amount": request.form.get("amount", 0),
            "login_attempts": request.form.get("login_attempts", 0),
            "past_day_transactions": request.form.get("past_day_transactions", 0),
            "past_week_transactions": request.form.get("past_week_transactions", 0),
            "location": request.form.get("location", "")
        }

        if model is not None:
            fraud_prob, err = model_predict(tx)
            if err:
                is_fraud, reason = rule_based_fraud_check(tx)
                result = "Fraud" if is_fraud else "Not Fraud"
                details = {
                    "source": "Rule-based (Model error)",
                    "reason": reason,
                    "fraud_probability": None,
                    "error": err
                }
            else:
                is_fraud = fraud_prob > 0.5
                result = "Fraud" if is_fraud else "Not Fraud"
                details = {
                    "source": "Model",
                    "reason": None,
                    "fraud_probability": round(fraud_prob * 100, 2)
                }
        else:
            is_fraud, reason = rule_based_fraud_check(tx)
            result = "Fraud" if is_fraud else "Not Fraud"
            details = {
                "source": "Rule-based",
                "reason": reason,
                "fraud_probability": None
            }

    return render_template("predict.html", result=result, details=details)

if __name__ == "__main__":
    app.run(debug=True)
