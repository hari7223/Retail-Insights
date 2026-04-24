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

def get_db():
    return get_engine().connect()

def load_data():
    conn = get_db()
    tx = pd.read_sql("SELECT * FROM transactions", conn)
    hh = pd.read_sql("SELECT * FROM households", conn)
    conn.close()

    # Convert types after loading since all columns are VARCHAR in DB
    tx['SPEND'] = pd.to_numeric(tx['SPEND'], errors='coerce')
    tx['UNITS'] = pd.to_numeric(tx['UNITS'], errors='coerce')
    tx['WEEK_NUM'] = pd.to_numeric(tx['WEEK_NUM'], errors='coerce')
    tx['HSHD_NUM'] = pd.to_numeric(tx['HSHD_NUM'], errors='coerce')
    tx['PURCHASE_DATE'] = pd.to_datetime(tx['PURCHASE_DATE'], format='%d-%b-%y', errors='coerce')

    # Guard against duplicate rows from repeated uploads
    tx = tx.drop_duplicates()
    hh = hh.drop_duplicates(subset=['HSHD_NUM'])

    return tx, hh

def build_features(tx, hh):
    snapshot_date = tx['PURCHASE_DATE'].max()

    rfm = tx.groupby('HSHD_NUM').agg(
        recency=('PURCHASE_DATE', lambda x: (snapshot_date - x.max()).days),
        frequency=('BASKET_NUM', 'nunique'),
        total_spend=('SPEND', 'sum'),
        avg_basket_value=('SPEND', lambda x: x.sum() / max(tx.loc[x.index, 'BASKET_NUM'].nunique(), 1)),
        total_units=('UNITS', 'sum')
    ).reset_index()

    # Spend trend: last 6 months vs prior 6 months
    cutoff = snapshot_date - pd.DateOffset(months=6)
    recent = tx[tx['PURCHASE_DATE'] >= cutoff].groupby('HSHD_NUM')['SPEND'].sum().rename('recent_spend')
    older = tx[tx['PURCHASE_DATE'] < cutoff].groupby('HSHD_NUM')['SPEND'].sum().rename('older_spend')
    rfm = rfm.merge(recent, on='HSHD_NUM', how='left').merge(older, on='HSHD_NUM', how='left')
    rfm['spend_trend'] = rfm['recent_spend'].fillna(0) - rfm['older_spend'].fillna(0)

    # Encode demographics
    le = LabelEncoder()
    for col in ['INCOME_RANGE', 'AGE_RANGE', 'HH_SIZE', 'L']:
        hh[col] = le.fit_transform(hh[col].astype(str))

    hh['HSHD_NUM'] = pd.to_numeric(hh['HSHD_NUM'], errors='coerce')

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    print(f"CLV Model MAE: ${mean_absolute_error(y_test, model.predict(X_test)):.2f}")
    joblib.dump(model, "models/clv_model.pkl")
    return model

def train_churn_model(features):
    """Random Forest — predicts churn (recency > 90 days)."""
    features['churned'] = (features['recency'] > 90).astype(int)
    print(f"Churn rate: {features['churned'].mean():.1%}")

    feature_cols = ['frequency', 'avg_basket_value',
                    'INCOME_RANGE', 'AGE_RANGE', 'HH_SIZE', 'L', 'CHILDREN']
    X = features[feature_cols]
    y = features['churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
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