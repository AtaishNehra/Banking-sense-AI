"""
Customer authentication and data management service.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from database.models import CustomerProfile, CreditAssessment, DeletedCustomerData
from database.connection import get_db_session

logger = logging.getLogger(__name__)

class CustomerAuthService:
    """
    Service for customer authentication and data management.
    """
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return hashlib.sha256(password.encode()).hexdigest() == hashed
    
    @staticmethod
    def hash_security_answer(answer: str) -> str:
        """Hash security question answer."""
        return hashlib.sha256(answer.lower().strip().encode()).hexdigest()
    
    def create_customer_profile(
        self, 
        customer_id: str, 
        password: str,
        security_q1: str,
        security_a1: str,
        security_q2: str,
        security_a2: str
    ) -> bool:
        """Create new customer profile with authentication."""
        try:
            db = get_db_session()
            
            # Check if customer already exists
            existing = db.query(CustomerProfile).filter(
                CustomerProfile.customer_id == customer_id
            ).first()
            
            if existing:
                db.close()
                return False
            
            # Create new customer profile
            profile = CustomerProfile(
                customer_id=customer_id,
                hashed_password=self.hash_password(password),
                security_question_1=security_q1,
                security_answer_1_hash=self.hash_security_answer(security_a1),
                security_question_2=security_q2,
                security_answer_2_hash=self.hash_security_answer(security_a2)
            )
            
            db.add(profile)
            db.commit()
            db.close()
            
            logger.info(f"Created customer profile for {customer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating customer profile: {e}")
            return False
    
    def authenticate_customer(self, customer_id: str, password: str) -> bool:
        """Authenticate customer with password."""
        try:
            db = get_db_session()
            
            profile = db.query(CustomerProfile).filter(
                CustomerProfile.customer_id == customer_id,
                CustomerProfile.is_active == True
            ).first()
            
            if not profile:
                db.close()
                return False
            
            # Verify password
            stored_password_hash = str(profile.hashed_password)
            is_valid = self.verify_password(password, stored_password_hash)
            logger.info(f"Authentication attempt for {customer_id}: {'success' if is_valid else 'failed'}")
            
            if is_valid:
                # Update last login
                profile.last_login = datetime.now()
                db.commit()
            
            db.close()
            return is_valid
            
        except Exception as e:
            logger.error(f"Error authenticating customer: {e}")
            return False
    
    def get_security_questions(self, customer_id: str) -> Optional[Tuple[str, str]]:
        """Get security questions for customer."""
        try:
            db = get_db_session()
            
            profile = db.query(CustomerProfile).filter(
                CustomerProfile.customer_id == customer_id,
                CustomerProfile.is_active == True
            ).first()
            
            if not profile:
                db.close()
                return None
            
            questions = (profile.security_question_1, profile.security_question_2)
            db.close()
            return questions
            
        except Exception as e:
            logger.error(f"Error getting security questions: {e}")
            return None
    
    def verify_security_answers(
        self, 
        customer_id: str, 
        answer1: str, 
        answer2: str
    ) -> bool:
        """Verify security question answers."""
        try:
            db = get_db_session()
            
            profile = db.query(CustomerProfile).filter(
                CustomerProfile.customer_id == customer_id,
                CustomerProfile.is_active == True
            ).first()
            
            if not profile:
                db.close()
                return False
            
            # Verify both answers
            answer1_valid = self.hash_security_answer(answer1) == profile.security_answer_1_hash
            answer2_valid = self.hash_security_answer(answer2) == profile.security_answer_2_hash
            
            db.close()
            return answer1_valid and answer2_valid
            
        except Exception as e:
            logger.error(f"Error verifying security answers: {e}")
            return False
    
    def get_customer_assessments(self, customer_id: str) -> List[Dict]:
        """Get all credit assessments for customer."""
        try:
            db = get_db_session()
            
            assessments = db.query(CreditAssessment).filter(
                CreditAssessment.customer_id == customer_id
            ).order_by(CreditAssessment.created_at.desc()).all()
            
            result = []
            for assessment in assessments:
                shap_values = {}
                try:
                    if assessment.shap_values:
                        shap_values = json.loads(assessment.shap_values)
                except:
                    pass
                
                result.append({
                    'id': assessment.id,
                    'risk_score': assessment.risk_score,
                    'risk_category': assessment.risk_category,
                    'assessment_notes': assessment.assessment_notes,
                    'recommendation': assessment.recommendation,
                    'shap_values': shap_values,
                    'created_at': assessment.created_at.isoformat(),
                    'input_features': {
                        'amount_mean': assessment.amount_mean,
                        'amount_sum': assessment.amount_sum,
                        'amount_count': assessment.amount_count,
                        'daily_transaction_count': assessment.daily_transaction_count,
                        'daily_amount_avg': assessment.daily_amount_avg,
                        'has_fraud_history': assessment.has_fraud_history
                    }
                })
            
            db.close()
            return result
            
        except Exception as e:
            logger.error(f"Error getting customer assessments: {e}")
            return []
    
    def delete_customer_data(self, customer_id: str, reason: str = "Security verification failed") -> bool:
        """Delete customer data and archive to admin-only table."""
        try:
            db = get_db_session()
            
            # Get customer profile and assessments
            profile = db.query(CustomerProfile).filter(
                CustomerProfile.customer_id == customer_id
            ).first()
            
            if not profile:
                db.close()
                return False
            
            assessments = db.query(CreditAssessment).filter(
                CreditAssessment.customer_id == customer_id
            ).all()
            
            # Prepare archived data
            archived_assessments = []
            for assessment in assessments:
                archived_assessments.append({
                    'id': assessment.id,
                    'risk_score': assessment.risk_score,
                    'risk_category': assessment.risk_category,
                    'shap_values': assessment.shap_values,
                    'assessment_notes': assessment.assessment_notes,
                    'recommendation': assessment.recommendation,
                    'created_at': assessment.created_at.isoformat(),
                    'input_features': {
                        'amount_mean': assessment.amount_mean,
                        'amount_sum': assessment.amount_sum,
                        'amount_count': assessment.amount_count,
                        'daily_transaction_count': assessment.daily_transaction_count,
                        'daily_amount_avg': assessment.daily_amount_avg,
                        'has_fraud_history': assessment.has_fraud_history
                    }
                })
            
            archived_profile = {
                'customer_id': profile.customer_id,
                'created_at': profile.created_at.isoformat(),
                'last_login': profile.last_login.isoformat() if profile.last_login else None,
                'security_questions': [profile.security_question_1, profile.security_question_2]
            }
            
            # Create archive record
            deleted_data = DeletedCustomerData(
                original_customer_id=customer_id,
                deletion_reason=reason,
                archived_assessments=json.dumps(archived_assessments),
                archived_profile_data=json.dumps(archived_profile)
            )
            
            db.add(deleted_data)
            
            # Delete actual data
            for assessment in assessments:
                db.delete(assessment)
            db.delete(profile)
            
            db.commit()
            db.close()
            
            logger.info(f"Deleted customer data for {customer_id}, reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting customer data: {e}")
            return False
    
    def customer_exists(self, customer_id: str) -> bool:
        """Check if customer profile exists."""
        try:
            db = get_db_session()
            
            profile = db.query(CustomerProfile).filter(
                CustomerProfile.customer_id == customer_id,
                CustomerProfile.is_active == True
            ).first()
            
            exists = profile is not None
            db.close()
            return exists
            
        except Exception as e:
            logger.error(f"Error checking customer existence: {e}")
            return False

# Global service instance
customer_auth_service = CustomerAuthService()