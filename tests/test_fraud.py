"""
Tests for fraud detection model.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.fraud_training import FraudDetectionTrainer
from models.evaluate_fraud import FraudModelEvaluator

class TestFraudDetection:
    """
    Test cases for fraud detection model.
    """
    
    def test_fraud_training_execution(self):
        """
        Test that fraud training executes end-to-end without errors.
        """
        # Skip if processed data doesn't exist
        data_path = Path("data/processed/transactions_processed.csv")
        if not data_path.exists():
            pytest.skip("Processed data not available")
        
        try:
            trainer = FraudDetectionTrainer()
            
            # Use small number of trials for testing
            model = trainer.train(str(data_path), n_trials=5)
            
            # Check that model was trained
            assert model is not None
            assert trainer.feature_columns is not None
            assert len(trainer.feature_columns) > 0
            
            # Check that model file was saved
            model_path = Path("models/fraud_model.pkl")
            assert model_path.exists()
            
        except Exception as e:
            pytest.fail(f"Fraud training failed: {e}")
    
    def test_fraud_evaluation_returns_metrics(self):
        """
        Test that fraud evaluation returns correct metrics.
        """
        # Skip if model doesn't exist
        model_path = Path("models/fraud_model.pkl")
        test_data_path = Path("models/fraud_test_data.pkl")
        
        if not (model_path.exists() and test_data_path.exists()):
            pytest.skip("Fraud model or test data not available")
        
        try:
            evaluator = FraudModelEvaluator()
            metrics = evaluator.evaluate(str(model_path), str(test_data_path))
            
            # Check that all required metrics are present
            required_metrics = ['precision', 'recall', 'f1', 'auc']
            for metric in required_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], (int, float))
                assert 0 <= metrics[metric] <= 1
            
            # Check that files were created
            assert Path("docs/figures/fraud_confusion_matrix.png").exists()
            assert Path("docs/figures/fraud_roc_curve.png").exists()
            assert Path("docs/metrics/fraud_metrics.json").exists()
            
        except Exception as e:
            pytest.fail(f"Fraud evaluation failed: {e}")
    
    def test_fraud_model_prediction(self):
        """
        Test that fraud model can make predictions.
        """
        model_path = Path("models/fraud_model.pkl")
        if not model_path.exists():
            pytest.skip("Fraud model not available")
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            feature_columns = model_data['feature_columns']
            
            # Create sample input
            sample_data = {col: 0.0 for col in feature_columns}
            sample_data.update({
                'amount': 1000.0,
                'log_amount': np.log1p(1000.0),
                'type_TRANSFER': 1.0
            })
            
            # Ensure all required features are present
            for col in feature_columns:
                if col not in sample_data:
                    sample_data[col] = 0.0
            
            df = pd.DataFrame([sample_data])
            df = df[feature_columns]  # Ensure correct order
            
            # Make prediction
            prediction = model.predict(df)
            probability = model.predict_proba(df)
            
            # Check outputs
            assert len(prediction) == 1
            assert prediction[0] in [0, 1]
            assert probability.shape == (1, 2)
            assert 0 <= probability[0][1] <= 1
            
        except Exception as e:
            pytest.fail(f"Fraud model prediction failed: {e}")
    
    def test_feature_engineering(self):
        """
        Test feature engineering for fraud detection.
        """
        # Create sample data
        sample_data = {
            'amount': [100, 1000, 10000],
            'step': [1, 25, 49],
            'oldbalanceOrg': [1000, 2000, 3000],
            'newbalanceOrig': [900, 1000, 0],
            'oldbalanceDest': [0, 1000, 2000],
            'newbalanceDest': [100, 2000, 12000],
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
            'isFraud': [0, 0, 1]
        }
        
        df = pd.DataFrame(sample_data)
        
        try:
            trainer = FraudDetectionTrainer()
            features = trainer.prepare_features(df)
            
            # Check that features were created
            assert len(features.columns) > 0
            assert len(features) == 3
            
            # Check for specific engineered features
            expected_features = ['amount', 'log_amount']
            for feature in expected_features:
                if feature in trainer.feature_columns:
                    assert not features[feature].isnull().any()
                    
        except Exception as e:
            pytest.fail(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
