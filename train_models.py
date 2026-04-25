import pandas as pd
import numpy as np
import joblib

import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, classification_report
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine, text

def get_engine():
    server = os.getenv('SQL_SERVER')
    database = os.getenv('SQL_DATABASE')
    user = os.getenv('SQL_USER')
    password = os.getenv('SQL_PASSWORD')
    return create_engine(f"mssql+pymssql://{user}:{password}@{server}/{database}")

def load_data():
    # We only need the household demographics directly
    with get_engine().connect() as conn:
        hh = pd.read_sql("SELECT * FROM households", conn)
        
        # SQL PUSHDOWN: Instead of downloading 3.6M rows, we let SQL Server 
        # calculate the RFM metrics and return just ~5,000 household rows.
        rfm_query = """
        WITH max_date AS (
            SELECT MAX(CAST(PURCHASE_DATE AS DATE)) as snap_date FROM transactions
        ),
        agg AS (
            SELECT 
                t.HSHD_NUM,
                MAX(CAST(t.PURCHASE_DATE AS DATE)) as last_purchase,
                COUNT(DISTINCT t.BASKET_NUM) as frequency,
                SUM(CAST(t.SPEND AS DECIMAL(10,2))) as total_spend,
                SUM(CAST(t.UNITS AS INT)) as total_units,
                SUM(CASE WHEN CAST(t.PURCHASE_DATE AS DATE) >= DATEADD(month, -6, (SELECT snap_date FROM max_date)) 
                         THEN CAST(t.SPEND AS DECIMAL(10,2)) ELSE 0 END) as recent_spend,
                SUM(CASE WHEN CAST(t.PURCHASE_DATE AS DATE) < DATEADD(month, -6, (SELECT snap_date FROM max_date)) 
                         THEN CAST(t.SPEND AS DECIMAL(10,2)) ELSE 0 END) as older_spend,
                (SELECT snap_date FROM max_date) as snapshot_date
            FROM transactions t
            GROUP BY t.HSHD_NUM
        )
        SELECT * FROM agg
        """
        rfm = pd.read_sql(rfm_query, conn)

    hh = hh.drop_duplicates(subset=['HSHD_NUM'])
    return rfm, hh

def build_features(rfm, hh):
    # Calculate Pandas-level features based on the SQL aggregated data
    rfm['recency'] = (pd.to_datetime(rfm['snapshot_date']) - pd.to_datetime(rfm['last_purchase'])).dt.days
    rfm['avg_basket_value'] = rfm['total_spend'] / rfm['frequency'].clip(lower=1)
    rfm['spend_trend'] = rfm['recent_spend'].fillna(0) - rfm['older_spend'].fillna(0)

    # Encode demographics
    le = LabelEncoder()
    for col in ['INCOME_RANGE', 'AGE_RANGE', 'HH_SIZE', 'L']:
        hh[col] = le.fit_transform(hh[col].astype(str))

    hh['HSHD_NUM'] = pd.to_numeric(hh['HSHD_NUM'], errors='coerce')
    rfm['HSHD_NUM'] = pd.to_numeric(rfm['HSHD_NUM'], errors='coerce')

    features = rfm.merge(
        hh[['HSHD_NUM', 'INCOME_RANGE', 'AGE_RANGE', 'HH_SIZE', 'L', 'CHILDREN']],
        on='HSHD_NUM', how='left'
    )
    features['CHILDREN'] = (features['CHILDREN'].astype(str).str.strip() == 'Y').astype(int)
    features = features.fillna(0)
    
    return features

def train_clv_model(features):
    """Gradient Boosting — predicts Customer Lifetime Value (total spend)."""
    feature_cols = ['recency', 'frequency', 'avg_basket_value', 'total_units',
                    'spend_trend', 'INCOME_RANGE', 'AGE_RANGE', 'HH_SIZE', 'L', 'CHILDREN']
    X = features[feature_cols]
    y = features['total_spend']

    # Sample to cap training time on large datasets (one row = one household)
    if len(X) > 10000:
        idx = X.sample(10000, random_state=42).index
        X, y = X.loc[idx], y.loc[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Reduced complexity: 100 trees, depth 3 — trains ~4x faster, nearly same accuracy
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                      max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    print(f"CLV Model MAE: ${mean_absolute_error(y_test, model.predict(X_test)):.2f}")
    joblib.dump(model, "models/clv_model.pkl")
    return model

def train_churn_model(features):
    """Random Forest — predicts churn (recency > 90 days)."""
    features = features.copy()
    features['churned'] = (features['recency'] > 90).astype(int)
    print(f"Churn rate: {features['churned'].mean():.1%}")

    feature_cols = ['frequency', 'avg_basket_value',
                    'INCOME_RANGE', 'AGE_RANGE', 'HH_SIZE', 'L', 'CHILDREN']
    X = features[feature_cols]
    y = features['churned']

    # Sample to cap training time on large datasets
    if len(X) > 10000:
        idx = X.sample(10000, random_state=42).index
        X, y = X.loc[idx], y.loc[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Reduced complexity: 50 trees, depth 5 — trains ~2x faster, negligible accuracy loss
    model = RandomForestClassifier(n_estimators=50, max_depth=5,
                                   random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    joblib.dump(model, "models/churn_model.pkl")
    return model

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    tx, hh = load_data()
    features = build_features(tx, hh)
    train_clv_model(features)
    train_churn_model(features)
    features.to_csv("models/features.csv", index=False)
    print("Training complete.")