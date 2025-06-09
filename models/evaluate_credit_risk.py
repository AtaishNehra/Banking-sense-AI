"""
Evaluate credit risk model performance.
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, log_loss, 
    calibration_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
import shap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditRiskEvaluator:
    """
    Evaluator for credit risk model.
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
    
    def load_model(self, model_path):
        """
        Load trained credit risk model.
        """
        logger.info(f"Loading model from {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        
        logger.info("Model loaded successfully")
    
    def load_test_data(self, test_data_path):
        """
        Load test dataset.
        """
        logger.info(f"Loading test data from {test_data_path}")
        
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        return test_data['X_test'], test_data['y_test']
    
    def calculate_calibration_error(self, y_true, y_prob, n_bins=10):
        """
        Calculate calibration error (Expected Calibration Error).
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        # Calculate bin sizes
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper, fraction_pos, mean_pred in zip(
            bin_lowers, bin_uppers, fraction_of_positives, mean_predicted_value
        ):
            # Find samples in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = fraction_pos
                avg_confidence_in_bin = mean_pred
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def evaluate(self, model_path="models/credit_risk_model.pkl", 
                test_data_path="models/credit_risk_test_data.pkl"):
        """
        Evaluate the credit risk model.
        """
        # Load model and test data
        self.load_model(model_path)
        X_test, y_test = self.load_test_data(test_data_path)
        
        logger.info(f"Evaluating on test set with {len(X_test)} samples")
        
        # Make predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        calibration_error = self.calculate_calibration_error(y_test, y_pred_proba)
        brier_score = brier_score_loss(y_test, y_pred_proba)
        
        metrics = {
            'auc_roc': auc_roc,
            'log_loss': logloss,
            'calibration_error': calibration_error,
            'brier_score': brier_score
        }
        
        logger.info("Model Performance:")
        logger.info(f"AUC-ROC: {auc_roc:.4f}")
        logger.info(f"Log Loss: {logloss:.4f}")
        logger.info(f"Calibration Error: {calibration_error:.4f}")
        logger.info(f"Brier Score: {brier_score:.4f}")
        
        # Generate plots
        self.plot_roc_curve(y_test, y_pred_proba, auc_roc)
        self.plot_calibration_curve(y_test, y_pred_proba)
        
        # Calculate SHAP values
        shap_values = self.calculate_shap_values(X_test)
        
        # Save metrics
        self.save_metrics(metrics)
        
        return metrics, shap_values
    
    def plot_roc_curve(self, y_true, y_pred_proba, auc_score):
        """
        Plot and save ROC curve.
        """
        # Create figures directory
        figures_dir = Path("docs/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Credit Risk Model - ROC Curve')
        plt.legend(loc="lower right")
        
        # Save plot
        output_path = figures_dir / "credit_roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {output_path}")
    
    def plot_calibration_curve(self, y_true, y_pred_proba):
        """
        Plot and save calibration curve.
        """
        # Create figures directory
        figures_dir = Path("docs/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label="Credit Risk Model", color='darkorange')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Credit Risk Model - Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = figures_dir / "credit_calibration.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration plot saved to {output_path}")
    
    def calculate_shap_values(self, X_test):
        """
        Calculate SHAP values for model interpretability.
        """
        logger.info("Calculating SHAP values")
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_test)
            
            # For binary classification, take the positive class SHAP values
            if len(shap_values) == 2:
                shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP values for feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance dictionary
            shap_dict = {
                feature: importance 
                for feature, importance in zip(self.feature_columns, feature_importance)
            }
            
            logger.info("SHAP values calculated successfully")
            return shap_dict
            
        except Exception as e:
            logger.warning(f"Failed to calculate SHAP values: {e}")
            # Return dummy SHAP values
            return {feature: 0.0 for feature in self.feature_columns}
    
    def save_metrics(self, metrics):
        """
        Save metrics to JSON file.
        """
        # Create metrics directory
        metrics_dir = Path("docs/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        output_path = metrics_dir / "credit_metrics.json"
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")

def main():
    """
    Main function to evaluate credit risk model.
    """
    evaluator = CreditRiskEvaluator()
    metrics, shap_values = evaluator.evaluate()
    
    print("\nFinal Credit Risk Model Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    print("\nTop Feature Importances (SHAP):")
    sorted_features = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:5]:
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    main()
