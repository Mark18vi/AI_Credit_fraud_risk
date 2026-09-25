import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_fraud_model():
    # Load data
    df = pd.read_csv('data/fraud.csv')
    
    # Preprocessing
    # Convert categorical variables to dummy variables
    df_encoded = pd.get_dummies(df, columns=['transaction_type', 'country', 'merchant', 'device'])
    
    X = df_encoded.drop(['transaction_id', 'is_fraud'], axis=1)
    y = df_encoded['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model: Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and columns for consistent prediction
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fraud_model.pkl')
    joblib.dump(X.columns.tolist(), 'models/fraud_columns.pkl')
    
    print(f"Fraud model trained. Accuracy: {model.score(X_test, y_test):.2f}")

def predict_fraud(input_data):
    model = joblib.load('models/fraud_model.pkl')
    columns = joblib.load('models/fraud_columns.pkl')
    
    # Create dataframe from input
    df_input = pd.DataFrame([input_data])
    
    # One-hot encoding to match training columns
    df_encoded = pd.get_dummies(df_input)
    
    # Add missing columns with 0
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Ensure column order matches
    df_final = df_encoded[columns]
    
    prob = model.predict_proba(df_final)[0][1]
    return prob

if __name__ == "__main__":
    train_fraud_model()
