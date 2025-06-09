"""
Simple fraud detection training using authentic PaySim data.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_fraud_model():
    """
    Train fraud detection model on authentic PaySim data.
    """
    # Load processed data
    data_path = "data/processed/paysim_processed.csv"
    logger.info(f"Loading processed PaySim data from {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} authentic banking transactions")
    logger.info(f"Fraud rate: {df['isFraud'].mean():.4f}")
    
    # Prepare features
    feature_columns = [
        'amount', 'amount_log', 'step', 'type_encoded',
        'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'balance_diff_orig', 'balance_diff_dest', 'is_transfer_or_cash_out',
        'orig_balance_zero', 'dest_balance_zero', 'is_round_amount'
    ]
    
    X = df[feature_columns]
    y = df['isFraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Training fraud rate: {y_train.mean():.4f}")
    
    # Train XGBoost model with optimized parameters for fraud detection
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=50,  # Handle class imbalance
        random_state=42,
        eval_metric='logloss'
    )
    
    logger.info("Training fraud detection model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"Test AUC Score: {auc_score:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 5 Most Important Features:")
    for _, row in feature_importance.head().iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model and test data
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "fraud_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    
    # Save test data for evaluation
    test_data_path = models_dir / "fraud_test_data.pkl"
    with open(test_data_path, 'wb') as f:
        pickle.dump({
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_columns': feature_columns
        }, f)
    logger.info(f"Test data saved to {test_data_path}")
    
    return model_path, auc_score

if __name__ == "__main__":
    train_fraud_model()