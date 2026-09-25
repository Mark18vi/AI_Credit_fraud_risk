import pandas as pd
import numpy as np
import os

def create_fraud_dataset():
    np.random.seed(42)
    n_rows = 1000
    data = {
        'transaction_id': range(1, n_rows + 1),
        'amount': np.random.uniform(1, 10000, n_rows),
        'transaction_type': np.random.choice(['transfer', 'payment', 'cash_out', 'cash_in'], n_rows),
        'country': np.random.choice(['US', 'UK', 'IN', 'CA', 'DE'], n_rows),
        'merchant': np.random.choice(['Amazon', 'Walmart', 'Apple', 'Target', 'Netflix'], n_rows),
        'time': np.random.randint(0, 86400, n_rows),
        'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_rows),
        'previous_fraud_count': np.random.randint(0, 5, n_rows),
        'is_fraud': np.random.choice([0, 1], n_rows, p=[0.95, 0.05])
    }
    df = pd.DataFrame(data)
    df.to_csv('data/fraud.csv', index=False)
    print("Created data/fraud.csv")

def create_credit_dataset():
    np.random.seed(42)
    n_rows = 1000
    data = {
        'customer_id': range(1, n_rows + 1),
        'income': np.random.uniform(20000, 150000, n_rows),
        'loan_amount': np.random.uniform(1000, 50000, n_rows),
        'credit_score': np.random.randint(300, 850, n_rows),
        'employment_years': np.random.randint(0, 40, n_rows),
        'debt_ratio': np.random.uniform(0.1, 0.6, n_rows),
        'risk_level': np.random.choice(['Low Risk', 'Medium Risk', 'High Risk'], n_rows)
    }
    df = pd.DataFrame(data)
    df.to_csv('data/credit.csv', index=False)
    print("Created data/credit.csv")

def create_identity_dataset():
    np.random.seed(42)
    n_rows = 1000
    data = {
        'user_id': range(1, n_rows + 1),
        'device_id': [f"dev_{np.random.randint(1, 500)}" for _ in range(n_rows)],
        'location': np.random.choice(['New York', 'London', 'Mumbai', 'Toronto', 'Berlin'], n_rows),
        'login_time': np.random.randint(0, 86400, n_rows),
        'ip_address': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(n_rows)],
        'browser': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge'], n_rows),
        'is_suspicious': np.random.choice([0, 1], n_rows, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)
    df.to_csv('data/identity.csv', index=False)
    print("Created data/identity.csv")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    create_fraud_dataset()
    create_credit_dataset()
    create_identity_dataset()
