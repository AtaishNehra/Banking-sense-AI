"""
Quick preprocessing for PaySim sample data.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_sample():
    """
    Preprocess PaySim sample for ML training.
    """
    input_path = "data/processed/paysim_sample.csv"
    output_path = "data/processed/paysim_processed.csv"
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading PaySim sample data...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} authentic banking transactions")
    logger.info(f"Fraud rate: {df['isFraud'].mean():.4f}")
    
    # Clean data
    df = df.dropna()
    
    # Feature engineering
    df['amount_log'] = np.log1p(df['amount'])
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Encode transaction type
    type_encoder = LabelEncoder()
    df['type_encoded'] = type_encoder.fit_transform(df['type'])
    
    # Binary features
    df['is_transfer_or_cash_out'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
    df['orig_balance_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
    df['dest_balance_zero'] = (df['oldbalanceDest'] == 0).astype(int)
    df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = ['amount', 'amount_log', 'step', 'oldbalanceOrg', 'newbalanceOrig', 
                     'oldbalanceDest', 'newbalanceDest', 'balance_diff_orig', 'balance_diff_dest']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save processed data
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    
    # Save preprocessor
    preprocessor_path = output_dir / "preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'type_encoder': type_encoder
        }, f)
    
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Fraud transactions: {df['isFraud'].sum()}")
    
    return str(output_path)

if __name__ == "__main__":
    preprocess_sample()