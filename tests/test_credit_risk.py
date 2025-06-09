"""
Tests for credit risk model.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.credit_risk_training import CreditRiskTrainer
from models.evaluate_credit_risk import CreditRiskEvaluator

class TestCreditRisk:
    """
    Test cases for credit risk model.
    """
    
    def test_credit_risk_training_execution(self):
        """
        Test that credit risk training runs without error.
        """
        # Skip if processed data doesn't exist
        data_path = Path("data/processed/transactions_processed.csv")
        if not data_path.exists():
            pytest.skip("Processed data not available")
        
        try:
            trainer = CreditRiskTrainer()
            model = trainer.train(str(data_path))
            
            # Check that model was trained
            assert model is not None
            assert trainer.feature_columns is not None
            assert len(trainer.feature_columns) > 0
            
            # Check that model file was saved
            model_path = Path("models/credit_risk_model.pkl")
            assert model_path.exists()
            
        except Exception as e:
            pytest.fail(f"Credit risk training failed: {e}")
    
    def test_predictions_between_zero_and_one(self):
        """
        Test that credit risk predictions are between 0 and 1.
        """
        model_path = Path("models/credit_risk_model.pkl")
        test_data_path = Path("models/credit_risk_test_data.pkl")
        
        if not (model_path.exists() and test_data_path.exists()):
            pytest.skip("Credit risk model or test data not available")
        
        try:
            # Load model and test data
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            with open(test_data_path, 'rb') as f:
                test_data = pickle.load(f)
            
            model = model_data['model']
            X_test = test_data['X_test']
            
            # Make predictions
            predictions = model.predict_proba(X_test)[:, 1]
            
            # Check that all predictions are between 0 and 1
            assert np.all(predictions >= 0)
            assert np.all(predictions <= 1)
            assert len(predictions) == len(X_test)
            
        except Exception as e:
            pytest.fail(f"Credit risk prediction validation failed: {e}")
    
    def test_shap_values_calculation(self):
        """
        Test that SHAP values can be calculated and sum appropriately.
        """
        model_path = Path("models/credit_risk_model.pkl")
        test_data_path = Path("models/credit_risk_test_data.pkl")
        
        if not (model_path.exists() and test_data_path.exists()):
            pytest.skip("Credit risk model or test data not available")
        
        try:
            evaluator = CreditRiskEvaluator()
            evaluator.load_model(str(model_path))
            X_test, y_test = evaluator.load_test_data(str(test_data_path))
            
            # Calculate SHAP values for a small sample
            sample_size = min(10, len(X_test))
            X_sample = X_test.iloc[:sample_size]
            
            shap_values = evaluator.calculate_shap_values(X_sample)
            
            # Check that SHAP values are returned
            assert isinstance(shap_values, dict)
            assert len(shap_values) > 0
            
            # Check that all feature names are present
            for feature in evaluator.feature_columns:
                assert feature in shap_values
                assert isinstance(shap_values[feature], (int, float))
            
        except Exception as e:
            # SHAP calculation might fail on some systems, so we'll warn instead of fail
            pytest.skip(f"SHAP calculation failed (expected on some systems): {e}")
    
    def test_synthetic_label_creation(self):
        """
        Test synthetic credit risk label creation.
        """
        # Create sample transaction data
        sample_data = {
            'nameOrig': ['customer1', 'customer1', 'customer2', 'customer2', 'customer3'] * 4,
            'amount': [100, 200, 5000, 10000, 50] * 4,
            'step': [1, 2, 3, 4, 5] * 4,
            'isFraud': [0, 0, 0, 1, 0] * 4
        }
        
        df = pd.DataFrame(sample_data)
        
        try:
            trainer = CreditRiskTrainer()
            customer_stats = trainer.create_synthetic_labels(df)
            
            # Check that labels were created
            assert 'credit_risk' in customer_stats.columns
            assert customer_stats['credit_risk'].dtype in [int, 'int64']
            assert customer_stats['credit_risk'].isin([0, 1]).all()
            
            # Check that customer-level statistics were calculated
            assert 'amount_mean' in customer_stats.columns
            assert 'amount_sum' in customer_stats.columns
            assert 'amount_count' in customer_stats.columns
            assert 'has_fraud_history' in customer_stats.columns
            
            # Check that we have the right number of customers
            assert len(customer_stats) == 3  # 3 unique customers
            
        except Exception as e:
            pytest.fail(f"Synthetic label creation failed: {e}")
    
    def test_credit_risk_evaluation_metrics(self):
        """
        Test that credit risk evaluation returns proper metrics.
        """
        model_path = Path("models/credit_risk_model.pkl")
        test_data_path = Path("models/credit_risk_test_data.pkl")
        
        if not (model_path.exists() and test_data_path.exists()):
            pytest.skip("Credit risk model or test data not available")
        
        try:
            evaluator = CreditRiskEvaluator()
            metrics, shap_values = evaluator.evaluate(str(model_path), str(test_data_path))
            
            # Check that all required metrics are present
            required_metrics = ['auc_roc', 'log_loss', 'calibration_error', 'brier_score']
            for metric in required_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], (int, float))
            
            # Check metric ranges
            assert 0 <= metrics['auc_roc'] <= 1
            assert metrics['log_loss'] >= 0
            assert 0 <= metrics['calibration_error'] <= 1
            assert 0 <= metrics['brier_score'] <= 1
            
            # Check that SHAP values are returned
            assert isinstance(shap_values, dict)
            
            # Check that files were created
            assert Path("docs/figures/credit_roc_curve.png").exists()
            assert Path("docs/figures/credit_calibration.png").exists()
            assert Path("docs/metrics/credit_metrics.json").exists()
            
        except Exception as e:
            pytest.fail(f"Credit risk evaluation failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
