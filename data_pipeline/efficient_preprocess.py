"""
Memory-efficient preprocessing for large PaySim dataset.
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

def process_data_efficiently():
    """
    Process PaySim data efficiently for the banking ML platform.
    """
    input_path = "data/raw/PS_20174392719_1491204439457_log.csv"
    output_path = "data/processed/paysim_processed.csv"
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Processing authentic PaySim dataset efficiently...")
    
    # Read data in chunks to manage memory
    chunk_size = 50000
    processed_chunks = []
    
    # Initialize encoders
    type_encoder = LabelEncoder()
    scaler = StandardScaler()
    
    # First pass: fit encoders and collect sample for scaling
    logger.info("First pass: analyzing data structure...")
    sample_data = []
    
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        if i == 0:
            logger.info(f"Dataset columns: {list(chunk.columns)}")
            logger.info(f"Total rows to process: {len(chunk)} per chunk")
            
            # Fit type encoder on first chunk
            type_encoder.fit(chunk['type'])
            
            # Collect sample for scaler fitting
            sample_data.append(chunk.sample(min(1000, len(chunk))))
        
        if i >= 10:  # Use first 10 chunks for analysis
            break
    
    # Fit scaler on sample data
    sample_df = pd.concat(sample_data, ignore_index=True)
    
    # Basic feature engineering on sample
    sample_df['amount_log'] = np.log1p(sample_df['amount'])
    sample_df['balance_diff_orig'] = sample_df['oldbalanceOrg'] - sample_df['newbalanceOrig']
    sample_df['balance_diff_dest'] = sample_df['newbalanceDest'] - sample_df['oldbalanceDest']
    
    # Fit scaler
    numerical_cols = ['amount', 'amount_log', 'step', 'oldbalanceOrg', 'newbalanceOrig', 
                     'oldbalanceDest', 'newbalanceDest', 'balance_diff_orig', 'balance_diff_dest']
    scaler.fit(sample_df[numerical_cols])
    
    logger.info("Second pass: processing and saving data...")
    
    # Second pass: process all data
    first_chunk = True
    total_processed = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        # Basic cleaning
        chunk = chunk.dropna()
        
        # Feature engineering
        chunk['amount_log'] = np.log1p(chunk['amount'])
        chunk['balance_diff_orig'] = chunk['oldbalanceOrg'] - chunk['newbalanceOrig']
        chunk['balance_diff_dest'] = chunk['newbalanceDest'] - chunk['oldbalanceDest']
        chunk['type_encoded'] = type_encoder.transform(chunk['type'])
        
        # Add binary features
        chunk['is_transfer_or_cash_out'] = chunk['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
        chunk['orig_balance_zero'] = (chunk['oldbalanceOrg'] == 0).astype(int)
        chunk['dest_balance_zero'] = (chunk['oldbalanceDest'] == 0).astype(int)
        chunk['is_round_amount'] = (chunk['amount'] % 1000 == 0).astype(int)
        
        # Normalize numerical features
        chunk[numerical_cols] = scaler.transform(chunk[numerical_cols])
        
        # Save chunk
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk.to_csv(output_path, mode=mode, header=header, index=False)
        
        total_processed += len(chunk)
        first_chunk = False
        
        if total_processed % 500000 == 0:
            logger.info(f"Processed {total_processed} records...")
    
    logger.info(f"Processing complete! Total records: {total_processed}")
    
    # Save preprocessor objects
    preprocessor_path = output_dir / "preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'type_encoder': type_encoder
        }, f)
    
    # Log final statistics
    final_df = pd.read_csv(output_path, nrows=10000)  # Sample for stats
    fraud_rate = final_df['isFraud'].mean()
    logger.info(f"Fraud rate in sample: {fraud_rate:.4f}")
    logger.info(f"Processed data saved to {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    process_data_efficiently()