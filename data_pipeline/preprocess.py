"""
Preprocess the PaySim dataset for ML training.
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

class PaySimPreprocessor:
    """
    Preprocessor for PaySim banking transaction data.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def validate_data(self, df):
        """
        Basic data validation for the dataset.
        """
        logger.info("Validating data quality...")
        
        # Check required columns exist
        required_columns = ['step', 'type', 'amount', 'isFraud']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Validate data types and values
        if not pd.api.types.is_numeric_dtype(df['amount']):
            logger.error("Amount column must be numeric")
            return False
            
        if df['amount'].isnull().any():
            logger.error("Amount column contains null values")
            return False
            
        valid_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
        if not df['type'].isin(valid_types).all():
            logger.error(f"Type column contains invalid values. Valid types: {valid_types}")
            return False
            
        if not df['isFraud'].isin([0, 1]).all():
            logger.error("isFraud column must contain only 0 or 1 values")
            return False
        
        logger.info("Data validation passed successfully!")
        return True
        
    def clean_data(self, df):
        """
        Clean the dataset by handling nulls and outliers.
        """
        logger.info("Cleaning data...")
        
        # Handle missing values
        df = df.dropna()
        
        # Remove extreme outliers (amounts beyond 99.9th percentile)
        amount_threshold = df['amount'].quantile(0.999)
        df = df[df['amount'] <= amount_threshold]
        
        # Remove zero-amount transactions that aren't fraud
        df = df[~((df['amount'] == 0) & (df['isFraud'] == 0))]
        
        logger.info(f"Data cleaned. Remaining records: {len(df)}")
        return df
        
    def engineer_features(self, df):
        """
        Engineer features from the raw data.
        """
        logger.info("Engineering features...")
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_sqrt'] = np.sqrt(df['amount'])
        
        # Balance-based features
        df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        # Flag for round amounts
        df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
        
        # Transaction type features
        df['is_transfer_or_cash_out'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
        
        # Zero balance flags
        df['orig_balance_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
        df['dest_balance_zero'] = (df['oldbalanceDest'] == 0).astype(int)
        
        logger.info("Feature engineering completed")
        return df
        
    def encode_categorical(self, df):
        """
        Encode categorical variables.
        """
        logger.info("Encoding categorical variables...")
        
        # Encode transaction type
        if 'type' not in self.label_encoders:
            self.label_encoders['type'] = LabelEncoder()
            df['type_encoded'] = self.label_encoders['type'].fit_transform(df['type'])
        else:
            df['type_encoded'] = self.label_encoders['type'].transform(df['type'])
        
        logger.info("Categorical encoding completed")
        return df
        
    def normalize_features(self, df, fit=True):
        """
        Normalize numerical features.
        """
        logger.info("Normalizing features...")
        
        numerical_columns = [
            'amount', 'amount_log', 'amount_sqrt', 'step',
            'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'balance_diff_orig', 'balance_diff_dest'
        ]
        
        # Only use columns that exist in the dataframe
        available_columns = [col for col in numerical_columns if col in df.columns]
        
        if fit:
            df[available_columns] = self.scaler.fit_transform(df[available_columns])
        else:
            df[available_columns] = self.scaler.transform(df[available_columns])
        
        logger.info("Feature normalization completed")
        return df
        
    def preprocess(self, input_path, output_path):
        """
        Main preprocessing pipeline.
        """
        logger.info(f"Starting preprocessing of {input_path}")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Validate data
        if not self.validate_data(df):
            raise ValueError("Data validation failed")
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categorical variables
        df = self.encode_categorical(df)
        
        # Normalize features
        df = self.normalize_features(df, fit=True)
        
        # Save preprocessed data
        df.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to {output_path}")
        
        # Save preprocessor objects
        preprocessor_path = output_dir / "preprocessor.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoders': self.label_encoders
            }, f)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Log dataset statistics
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Fraud rate: {df['isFraud'].mean():.4f}")
        
        return str(output_path)

def main():
    """
    Main function to run preprocessing pipeline.
    """
    input_path = "data/raw/PS_20174392719_1491204439457_log.csv"
    output_path = "data/processed/paysim_processed.csv"
    
    preprocessor = PaySimPreprocessor()
    preprocessor.preprocess(input_path, output_path)

if __name__ == "__main__":
    main()