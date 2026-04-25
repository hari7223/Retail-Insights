import joblib
import pandas as pd

FEATURE_COLS_CLV   = ['recency', 'frequency', 'avg_basket_value', 'total_units',
                       'spend_trend', 'INCOME_RANGE', 'AGE_RANGE', 'HH_SIZE', 'L', 'CHILDREN']
FEATURE_COLS_CHURN = ['frequency', 'avg_basket_value',
                       'INCOME_RANGE', 'AGE_RANGE', 'HH_SIZE', 'L', 'CHILDREN']

_CHURN_FEATURE_LABELS = {
    'frequency':        'Visit Frequency',
    'avg_basket_value': 'Avg Basket Value',
    'INCOME_RANGE':     'Income Range',
    'AGE_RANGE':        'Age Range',
    'HH_SIZE':          'Household Size',
    'L':                'Loyalty Status',
    'CHILDREN':         'Has Children',
}

_clv_model   = None
_churn_model = None

def _load_models():
    global _clv_model, _churn_model
    if _clv_model is not None:
        return True
    try:
        _clv_model   = joblib.load("models/clv_model.pkl")
        _churn_model = joblib.load("models/churn_model.pkl")
        return True
    except Exception:
        # Fails gracefully instead of hijacking the HTTP request to train models
        return False

def _load_features():
    features = pd.read_csv("models/features.csv")
    return features.drop_duplicates(subset=['HSHD_NUM']).reset_index(drop=True)

def get_all_predictions():
    _load_models()
    features = _load_features()
    features['clv_score']    = _clv_model.predict(features[FEATURE_COLS_CLV])
    features['churn_prob']   = _churn_model.predict_proba(features[FEATURE_COLS_CHURN])[:, 1]
    features['risk_segment'] = pd.cut(
        features['churn_prob'],
        bins=[-0.1, 0.3, 0.6, 1.0],  # Changed 0 to -0.1 here
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    return features[['HSHD_NUM', 'clv_score', 'churn_prob', 'risk_segment',
                      'recency', 'frequency', 'total_spend']]

def get_churn_importances():
    if not _load_models():
        return []
    return sorted(
        [{'feature': _CHURN_FEATURE_LABELS[c], 'importance': round(float(v), 4)}
         for c, v in zip(FEATURE_COLS_CHURN, _churn_model.feature_importances_)],
        key=lambda x: -x['importance']
    )

def get_churn_correlations():
    try:
        features = _load_features()
        features['churned'] = (features['recency'] > 90).astype(int)
        num_cols = ['recency', 'frequency', 'avg_basket_value', 'total_spend', 'spend_trend']
        labels   = ['Recency (days)', 'Visit Frequency', 'Avg Basket Value',
                     'Total Spend', 'Spend Trend']
        corr = features[num_cols + ['churned']].corr()['churned'].drop('churned')
        return [{'feature': lbl, 'correlation': round(float(val), 3)}
                for lbl, val in zip(labels, corr.values)]
    except Exception:
        return []