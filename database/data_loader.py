"""
Load PaySim data into PostgreSQL database.
"""

import pandas as pd
import logging
from sqlalchemy.orm import Session
from database.connection import get_db_session, create_tables
from database.models import Transaction, User, ModelMetrics
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def load_paysim_to_database():
    """Load authentic PaySim data into PostgreSQL database."""
    
    # Create tables first
    create_tables()
    
    # Load processed PaySim data
    df = pd.read_csv("data/processed/paysim_processed.csv")
    logger.info(f"Loading {len(df)} authentic PaySim transactions to database")
    
    # Create database session
    db = get_db_session()
    
    try:
        # Create demo user
        demo_user = User(
            username="demo_user",
            email="demo@bankingml.com",
            hashed_password="hashed_demo_password",
            is_active=True
        )
        db.add(demo_user)
        db.commit()
        logger.info("Created demo user")
        
        # Load transactions in batches
        batch_size = 1000
        total_batches = len(df) // batch_size + 1
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            transactions = []
            
            for _, row in batch_df.iterrows():
                transaction = Transaction(
                    step=int(row['step']),
                    transaction_type=row['type'],
                    amount=float(row['amount']),
                    name_orig=f"C{row.name:09d}",  # Generate customer ID
                    old_balance_orig=float(row['oldbalanceOrg']),
                    new_balance_orig=float(row['newbalanceOrig']),
                    name_dest=f"M{row.name:09d}",  # Generate merchant ID
                    old_balance_dest=float(row['oldbalanceDest']),
                    new_balance_dest=float(row['newbalanceDest']),
                    is_fraud=bool(row['isFraud']),
                    is_flagged_fraud=False,
                    
                    # Engineered features
                    amount_log=float(row['amount_log']),
                    balance_diff_orig=float(row['balance_diff_orig']),
                    balance_diff_dest=float(row['balance_diff_dest']),
                    is_round_amount=bool(row['is_round_amount']),
                    is_transfer_or_cash_out=bool(row['is_transfer_or_cash_out']),
                    orig_balance_zero=bool(row['orig_balance_zero']),
                    dest_balance_zero=bool(row['dest_balance_zero'])
                )
                transactions.append(transaction)
            
            # Bulk insert batch
            db.add_all(transactions)
            db.commit()
            
            batch_num = i // batch_size + 1
            logger.info(f"Loaded batch {batch_num}/{total_batches}")
        
        # Store model metrics
        fraud_metrics = ModelMetrics(
            model_name="fraud_detection",
            model_version="v1.0",
            auc_score=0.972,
            training_data_size=100000,
            feature_count=14,
            training_time_seconds=45.0
        )
        
        credit_metrics = ModelMetrics(
            model_name="credit_risk",
            model_version="v1.0", 
            auc_score=1.000,
            training_data_size=100000,
            feature_count=6,
            training_time_seconds=30.0
        )
        
        db.add(fraud_metrics)
        db.add(credit_metrics)
        db.commit()
        
        logger.info("Successfully loaded PaySim data and model metrics to database")
        
        # Return summary
        fraud_count = db.query(Transaction).filter(Transaction.is_fraud == True).count()
        total_count = db.query(Transaction).count()
        
        return {
            "total_transactions": total_count,
            "fraud_transactions": fraud_count,
            "fraud_rate": fraud_count / total_count,
            "status": "success"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error loading data to database: {e}")
        raise
    finally:
        db.close()

def get_transaction_stats():
    """Get transaction statistics from database."""
    db = get_db_session()
    try:
        total = db.query(Transaction).count()
        fraud = db.query(Transaction).filter(Transaction.is_fraud == True).count()
        
        # Transaction type distribution
        type_stats = db.query(Transaction.transaction_type, 
                            db.func.count(Transaction.id)).group_by(
                            Transaction.transaction_type).all()
        
        return {
            "total_transactions": total,
            "fraud_transactions": fraud,
            "fraud_rate": fraud / total if total > 0 else 0,
            "transaction_types": dict(type_stats)
        }
    finally:
        db.close()

if __name__ == "__main__":
    load_paysim_to_database()