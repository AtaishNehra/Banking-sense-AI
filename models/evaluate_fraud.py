"""
Evaluate fraud detection model performance.
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
    confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudModelEvaluator:
    """
    Evaluator for fraud detection model.
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
    
    def load_model(self, model_path):
        """
        Load trained fraud detection model.
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
    
    def evaluate(self, model_path="models/fraud_model.pkl", 
                test_data_path="models/fraud_test_data.pkl"):
        """
        Evaluate the fraud detection model.
        """
        # Load model and test data
        self.load_model(model_path)
        X_test, y_test = self.load_test_data(test_data_path)
        
        logger.info(f"Evaluating on test set with {len(X_test)} samples")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_roc
        }
        
        logger.info("Model Performance:")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"AUC-ROC: {auc_roc:.4f}")
        
        # Generate confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Generate ROC curve
        self.plot_roc_curve(y_test, y_pred_proba)
        
        # Save metrics
        self.save_metrics(metrics)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot and save confusion matrix.
        """
        # Create figures directory
        figures_dir = Path("docs/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Fraud', 'Fraud'],
                   yticklabels=['Not Fraud', 'Fraud'])
        plt.title('Fraud Detection - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot
        output_path = figures_dir / "fraud_confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """
        Plot and save ROC curve.
        """
        # Create figures directory
        figures_dir = Path("docs/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Fraud Detection - ROC Curve')
        plt.legend(loc="lower right")
        
        # Save plot
        output_path = figures_dir / "fraud_roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {output_path}")
    
    def save_metrics(self, metrics):
        """
        Save metrics to JSON file.
        """
        # Create metrics directory
        metrics_dir = Path("docs/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        output_path = metrics_dir / "fraud_metrics.json"
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")

def main():
    """
    Main function to evaluate fraud detection model.
    """
    evaluator = FraudModelEvaluator()
    metrics = evaluator.evaluate()
    
    print("\nFinal Fraud Detection Model Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()
