"""
Enhanced credit risk service with proper database integration.
"""

import json
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from database.connection import get_db_session
from database.models import CreditAssessment, CustomerProfile
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EnhancedCreditService:
    """Enhanced credit risk assessment with database persistence."""
    
    def save_assessment(self, customer_id: str, assessment_data: Dict[str, Any]) -> str:
        """Save credit assessment to database."""
        try:
            db = get_db_session()
            
            # Create assessment record
            assessment = CreditAssessment(
                customer_id=customer_id,
                amount_mean=assessment_data.get('amount_mean', 0),
                amount_sum=assessment_data.get('amount_sum', 0), 
                amount_count=assessment_data.get('amount_count', 0),
                daily_transaction_count=assessment_data.get('daily_transaction_count', 0),
                daily_amount_avg=assessment_data.get('daily_amount_avg', 0),
                has_fraud_history=assessment_data.get('has_fraud_history', 0),
                risk_score=assessment_data.get('risk_score', 0),
                risk_category=assessment_data.get('risk_category', 'UNKNOWN'),
                shap_values=json.dumps(assessment_data.get('shap_values', {})),
                assessment_notes=f"Assessment completed at {datetime.now()}",
                recommendation=self._generate_recommendation(assessment_data.get('risk_score', 0))
            )
            
            db.add(assessment)
            db.commit()
            
            assessment_id = assessment.id
            db.close()
            
            logger.info(f"Saved assessment {assessment_id} for customer {customer_id}")
            return assessment_id
            
        except Exception as e:
            logger.error(f"Error saving assessment: {e}")
            if 'db' in locals():
                db.rollback()
                db.close()
            return None
    
    def get_customer_assessments(self, customer_id: str) -> list:
        """Get all assessments for a customer."""
        try:
            db = get_db_session()
            
            assessments = db.query(CreditAssessment).filter(
                CreditAssessment.customer_id == customer_id
            ).order_by(CreditAssessment.created_at.desc()).limit(10).all()
            
            result = []
            for assessment in assessments:
                result.append({
                    'id': assessment.id,
                    'risk_score': assessment.risk_score,
                    'risk_category': assessment.risk_category,
                    'created_at': assessment.created_at.isoformat(),
                    'recommendation': assessment.recommendation
                })
            
            db.close()
            return result
            
        except Exception as e:
            logger.error(f"Error getting assessments: {e}")
            if 'db' in locals():
                db.close()
            return []
    
    def _generate_recommendation(self, risk_score: float) -> str:
        """Generate recommendation based on risk score."""
        if risk_score < 0.3:
            return "Low risk customer. Approve credit with standard terms."
        elif risk_score < 0.6:
            return "Medium risk customer. Approve with careful monitoring and higher interest rate."
        else:
            return "High risk customer. Require additional documentation and collateral."

# Global service instance
enhanced_credit_service = EnhancedCreditService()