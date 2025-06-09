"""
Train credit risk scoring model with synthetic labels.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditRiskTrainer:
    """
    Trainer for credit risk scoring model.
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
    
    def create_synthetic_labels(self, df):
        """
        Create synthetic credit risk labels based on transaction patterns.
        """
        logger.info("Creating synthetic credit risk labels")
        
        # Group by customer (nameOrig)
        if 'nameOrig' not in df.columns:
            raise ValueError("nameOrig column required for credit risk modeling")
        
        # Calculate customer-level statistics
        customer_stats = df.groupby('nameOrig').agg({
            'amount': ['mean', 'sum', 'count', 'std'],
            'step': ['min', 'max'],
            'isFraud': 'sum'
        }).round(4)
        
        # Flatten column names
        customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns]
        customer_stats = customer_stats.reset_index()
        
        # Calculate daily transaction count and average daily amount
        if 'step_max' in customer_stats.columns and 'step_min' in customer_stats.columns:
            customer_stats['days_active'] = (customer_stats['step_max'] - customer_stats['step_min']) / 24 + 1
            customer_stats['daily_transaction_count'] = customer_stats['amount_count'] / customer_stats['days_active']
            customer_stats['daily_amount_avg'] = customer_stats['amount_sum'] / customer_stats['days_active']
        else:
            # Fallback if step columns don't exist
            customer_stats['daily_transaction_count'] = customer_stats['amount_count']
            customer_stats['daily_amount_avg'] = customer_stats['amount_mean']
        
        # Define high risk criteria:
        # 1. Average daily amount in top 25th percentile
        # 2. More than 10 transactions per day
        amount_threshold = customer_stats['daily_amount_avg'].quantile(0.75)
        transaction_threshold = 10
        
        customer_stats['credit_risk'] = (
            (customer_stats['daily_amount_avg'] >= amount_threshold) & 
            (customer_stats['daily_transaction_count'] > transaction_threshold)
        ).astype(int)
        
        # Add fraud history as additional risk factor
        customer_stats['has_fraud_history'] = (customer_stats['isFraud_sum'] > 0).astype(int)
        customer_stats['credit_risk'] = np.maximum(
            customer_stats['credit_risk'], 
            customer_stats['has_fraud_history']
        )
        
        risk_rate = customer_stats['credit_risk'].mean()
        logger.info(f"Credit risk rate: {risk_rate:.4f}")
        logger.info(f"High risk customers: {customer_stats['credit_risk'].sum()}")
        
        return customer_stats
    
    def prepare_features(self, customer_stats):
        """
        Prepare features for credit risk model.
        """
        feature_columns = [
            'amount_mean', 'amount_sum', 'amount_count', 'amount_std',
            'daily_transaction_count', 'daily_amount_avg',
            'has_fraud_history'
        ]
        
        # Add days_active if available
        if 'days_active' in customer_stats.columns:
            feature_columns.append('days_active')
        
        # Only include columns that exist
        feature_columns = [col for col in feature_columns if col in customer_stats.columns]
        
        # Handle missing values
        features_df = customer_stats[feature_columns].fillna(0)
        
        self.feature_columns = feature_columns
        logger.info(f"Selected {len(feature_columns)} features: {feature_columns}")
        
        return features_df
    
    def train(self, data_path):
        """
        Train the credit risk model.
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Create synthetic labels
        customer_stats = self.create_synthetic_labels(df)
        
        # Prepare features
        X = self.prepare_features(customer_stats)
        y = customer_stats['credit_risk']
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"High risk rate: {y.mean():.4f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        logger.info("Training credit risk model")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        logger.info("Model performance:")
        logger.info(classification_report(y_test, y_pred))
        logger.info(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Save model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "credit_risk_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns
            }, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Save test data for evaluation
        test_data = {
            'X_test': X_test,
            'y_test': y_test
        }
        test_path = model_dir / "credit_risk_test_data.pkl"
        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f)
        
        logger.info(f"Test data saved to {test_path}")
        
        return self.model

def main():
    """
    Main function to train credit risk model.
    """
    trainer = CreditRiskTrainer()
    trainer.train("data/processed/transactions_processed.csv")

if __name__ == "__main__":
    main()
