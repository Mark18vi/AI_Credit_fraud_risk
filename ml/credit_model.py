import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_credit_model():
    # Load data
    df = pd.read_csv('data/credit.csv')
    
    X = df.drop(['customer_id', 'risk_level'], axis=1)
    y = df['risk_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model: Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/credit_model.pkl')
    
    print(f"Credit model trained. Accuracy: {model.score(X_test, y_test):.2f}")

def predict_credit_risk(input_data):
    model = joblib.load('models/credit_model.pkl')
    
    df_input = pd.DataFrame([input_data])
    risk = model.predict(df_input)[0]
    return risk

if __name__ == "__main__":
    train_credit_model()
