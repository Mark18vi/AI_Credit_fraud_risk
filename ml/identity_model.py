import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_identity_model():
    # Load data
    df = pd.read_csv('data/identity.csv')
    
    # Use only numeric features for Isolation Forest
    # For a fresher project, we keep it simple: login_time as feature
    X = df[['login_time']]
    
    # Model: Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/identity_model.pkl')
    
    print("Identity anomaly model trained.")

def analyze_identity_risk(df):
    model = joblib.load('models/identity_model.pkl')
    
    X = df[['login_time']]
    # IsolationForest returns -1 for outliers (suspicious) and 1 for inliers (normal)
    preds = model.predict(X)
    
    df['is_suspicious'] = [1 if p == -1 else 0 for p in preds]
    return df

if __name__ == "__main__":
    train_identity_model()
