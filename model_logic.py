import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
time_slots = [(6, 10), (11, 15), (19, 23)]  # Morning, Afternoon, Night slots

# Data Processing
def load_data():
    """Simulate loading or generating a dataset."""
    # Example dataset
    data = pd.DataFrame({
        "day_of_week": np.random.choice(days_of_week, size=100),
        "weather_condition": np.random.choice(["Sunny", "Cloudy", "Rainy"], size=100),
        "customers": np.random.randint(20, 50, size=100),
    })
    return data

def preprocess_data(data):
    """Preprocess the data for model training."""
    le = LabelEncoder()
    data["day_of_week"] = le.fit_transform(data["day_of_week"])
    data["weather_condition"] = le.fit_transform(data["weather_condition"])
    
    X = data[["day_of_week", "weather_condition"]]
    y = data["customers"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Modeling
def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Prediction
def predict(input_data):
    """Predict using the trained model."""
    # Load data and train model (for example purposes, we retrain each time)
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    
    # Assume input_data is a dictionary
    input_df = pd.DataFrame([input_data])
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df)
    
    prediction = model.predict(input_scaled)
    return prediction
