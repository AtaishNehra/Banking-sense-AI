"""
Train XGBoost model for fraud detection.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
# Using grid search instead of optuna for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionTrainer:
    """
    Trainer for fraud detection model using XGBoost.
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
    def prepare_features(self, df):
        """
        Prepare features for fraud detection model.
        """
        # Select relevant features
        feature_columns = []
        
        # Amount-based features
        if 'amount' in df.columns:
            feature_columns.append('amount')
        if 'log_amount' in df.columns:
            feature_columns.append('log_amount')
        if 'amount_rounded' in df.columns:
            feature_columns.append('amount_rounded')
            
        # Time-based features
        if 'hour' in df.columns:
            feature_columns.append('hour')
        if 'day' in df.columns:
            feature_columns.append('day')
        if 'step' in df.columns:
            feature_columns.append('step')
            
        # Balance features
        balance_features = ['balance_change_orig', 'balance_ratio_orig', 
                          'balance_change_dest', 'balance_ratio_dest']
        for feat in balance_features:
            if feat in df.columns:
                feature_columns.append(feat)
        
        # Rolling count feature
        if 'rolling_count' in df.columns:
            feature_columns.append('rolling_count')
            
        # Type dummy variables
        type_columns = [col for col in df.columns if col.startswith('type_')]
        feature_columns.extend(type_columns)
        
        # Other numerical features
        other_features = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        for feat in other_features:
            if feat in df.columns:
                feature_columns.append(feat)
        
        self.feature_columns = feature_columns
        logger.info(f"Selected {len(feature_columns)} features: {feature_columns}")
        
        return df[feature_columns]
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Optuna objective function for hyperparameter optimization.
        """
        # Suggest hyperparameters
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'eval_metric': 'auc',
            'use_label_encoder': False
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=50
        )
        
        # Predict and evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        return auc_score
    
    def train(self, data_path, test_size=0.3, val_size=0.5, n_trials=100):
        """
        Train the fraud detection model.
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Check for target column
        if 'isFraud' not in df.columns:
            raise ValueError("Target column 'isFraud' not found in dataset")
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = df['isFraud']
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Fraud rate: {y.mean():.4f}")
        
        # Split data: 70% train, 15% val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 ≈ 0.15/0.85
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Hyperparameter optimization with Optuna
        logger.info("Starting hyperparameter optimization")
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials
        )
        
        logger.info(f"Best AUC: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'random_state': 42,
            'eval_metric': 'auc',
            'use_label_encoder': False
        })
        
        self.model = xgb.XGBClassifier(**best_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True,
            early_stopping_rounds=50
        )
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        y_val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        logger.info("Validation set performance:")
        logger.info(classification_report(y_val, y_val_pred))
        logger.info(f"AUC-ROC: {roc_auc_score(y_val, y_val_pred_proba):.4f}")
        
        # Save model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "fraud_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'best_params': best_params
            }, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Save test data for evaluation
        test_data = {
            'X_test': X_test,
            'y_test': y_test
        }
        test_path = model_dir / "fraud_test_data.pkl"
        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f)
        
        logger.info(f"Test data saved to {test_path}")
        
        return self.model

def main():
    """
    Main function to train fraud detection model.
    """
    trainer = FraudDetectionTrainer()
    trainer.train("data/processed/transactions_processed.csv", n_trials=50)

if __name__ == "__main__":
    main()
