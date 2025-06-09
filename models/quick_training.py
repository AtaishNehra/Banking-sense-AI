"""
Quick training for banking ML models using authentic PaySim data.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_models():
    """Train fraud detection and credit risk models quickly."""
    
    # Load authentic PaySim data
    df = pd.read_csv("data/processed/paysim_processed.csv")
    logger.info(f"Training on {len(df)} authentic banking transactions")
    
    # Fraud detection model
    feature_cols = ['amount', 'amount_log', 'step', 'type_encoded', 
                   'balance_diff_orig', 'is_transfer_or_cash_out']
    X = df[feature_cols]
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Quick fraud model
    fraud_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, scale_pos_weight=30)
    fraud_model.fit(X_train, y_train)
    
    fraud_auc = roc_auc_score(y_test, fraud_model.predict_proba(X_test)[:, 1])
    logger.info(f"Fraud model AUC: {fraud_auc:.3f}")
    
    # Credit risk model (synthetic labels based on transaction patterns)
    df['credit_risk'] = ((df['amount'] > df['amount'].quantile(0.8)) & 
                        (df['balance_diff_orig'] < 0)).astype(int)
    
    cr_X = df[['amount', 'balance_diff_orig', 'type_encoded', 'step']]
    cr_y = df['credit_risk']
    
    cr_X_train, cr_X_test, cr_y_train, cr_y_test = train_test_split(cr_X, cr_y, test_size=0.2, random_state=42)
    
    credit_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    credit_model.fit(cr_X_train, cr_y_train)
    
    credit_auc = roc_auc_score(cr_y_test, credit_model.predict_proba(cr_X_test)[:, 1])
    logger.info(f"Credit risk model AUC: {credit_auc:.3f}")
    
    # Save models
    Path("models").mkdir(exist_ok=True)
    
    with open("models/fraud_model.pkl", 'wb') as f:
        pickle.dump(fraud_model, f)
    
    with open("models/credit_risk_model.pkl", 'wb') as f:
        pickle.dump(credit_model, f)
    
    # Save test data
    with open("models/fraud_test_data.pkl", 'wb') as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)
    
    with open("models/credit_risk_test_data.pkl", 'wb') as f:
        pickle.dump({'X_test': cr_X_test, 'y_test': cr_y_test}, f)
    
    logger.info("Models trained and saved successfully!")
    return fraud_auc, credit_auc

if __name__ == "__main__":
    train_models()