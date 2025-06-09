"""
Enhanced customer authentication and credit risk endpoints.
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import CustomerProfile, CreditAssessment
from services.customer_auth import customer_auth_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for API requests/responses
class CustomerRegistration(BaseModel):
    customer_id: str
    password: str
    security_question_1: str
    security_answer_1: str
    security_question_2: str
    security_answer_2: str

class CustomerLogin(BaseModel):
    customer_id: str
    password: str

class SecurityQuestionsRequest(BaseModel):
    customer_id: str

class SecurityAnswersRequest(BaseModel):
    customer_id: str
    answer_1: str
    answer_2: str

class EnhancedCreditRequest(BaseModel):
    customer_id: str
    amount_mean: float
    amount_sum: float
    amount_count: int
    daily_transaction_count: float
    daily_amount_avg: float
    has_fraud_history: int
    create_profile: Optional[bool] = False
    password: Optional[str] = None
    security_question_1: Optional[str] = None
    security_answer_1: Optional[str] = None
    security_question_2: Optional[str] = None
    security_answer_2: Optional[str] = None

class ChatbotCreditRequest(BaseModel):
    customer_id: str
    password: str

class ChatbotResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None

@router.post("/customer/register")
async def register_customer(request: CustomerRegistration):
    """Register new customer with authentication."""
    try:
        success = customer_auth_service.create_customer_profile(
            customer_id=request.customer_id,
            password=request.password,
            security_q1=request.security_question_1,
            security_a1=request.security_answer_1,
            security_q2=request.security_question_2,
            security_a2=request.security_answer_2
        )
        
        if success:
            return {"message": "Customer registered successfully", "customer_id": request.customer_id}
        else:
            raise HTTPException(status_code=400, detail="Customer ID already exists")
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@router.post("/customer/login")
async def login_customer(request: CustomerLogin):
    """Authenticate customer login."""
    try:
        is_valid = customer_auth_service.authenticate_customer(
            request.customer_id, 
            request.password
        )
        
        if is_valid:
            return {"message": "Login successful", "customer_id": request.customer_id}
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.post("/customer/security-questions")
async def get_security_questions(request: SecurityQuestionsRequest):
    """Get security questions for password recovery."""
    try:
        questions = customer_auth_service.get_security_questions(request.customer_id)
        
        if questions:
            return {
                "questions": {
                    "question_1": questions[0],
                    "question_2": questions[1]
                }
            }
        else:
            raise HTTPException(status_code=404, detail="Customer not found")
            
    except Exception as e:
        logger.error(f"Security questions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve security questions")

@router.post("/customer/verify-security")
async def verify_security_answers(request: SecurityAnswersRequest):
    """Verify security answers and return customer data or delete if failed."""
    try:
        is_valid = customer_auth_service.verify_security_answers(
            request.customer_id,
            request.answer_1,
            request.answer_2
        )
        
        if is_valid:
            # Return customer assessment data
            assessments = customer_auth_service.get_customer_assessments(request.customer_id)
            return {
                "message": "Security verification successful",
                "assessments": assessments
            }
        else:
            # Delete customer data
            deleted = customer_auth_service.delete_customer_data(
                request.customer_id,
                "Security verification failed"
            )
            
            if deleted:
                return {
                    "message": "Security verification failed. All customer data has been deleted for security. Contact administrator to recover historical data.",
                    "data_deleted": True
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to delete customer data")
                
    except Exception as e:
        logger.error(f"Security verification error: {e}")
        raise HTTPException(status_code=500, detail="Security verification failed")

@router.post("/predict/credit-risk-enhanced")
async def enhanced_credit_risk_prediction(request: EnhancedCreditRequest, db: Session = Depends(get_db)):
    """Enhanced credit risk prediction with customer profile management."""
    try:
        # Check if customer exists
        customer_exists = customer_auth_service.customer_exists(request.customer_id)
        
        # If customer doesn't exist and create_profile is True, create profile
        if not customer_exists and request.create_profile:
            if not all([request.password, request.security_question_1, 
                       request.security_answer_1, request.security_question_2, 
                       request.security_answer_2]):
                raise HTTPException(
                    status_code=400, 
                    detail="Password and security questions required for new customer"
                )
            
            created = customer_auth_service.create_customer_profile(
                customer_id=request.customer_id,
                password=request.password,
                security_q1=request.security_question_1,
                security_a1=request.security_answer_1,
                security_q2=request.security_question_2,
                security_a2=request.security_answer_2
            )
            
            if not created:
                raise HTTPException(status_code=400, detail="Failed to create customer profile")
        
        elif not customer_exists:
            raise HTTPException(status_code=404, detail="Customer not found. Set create_profile=true to register.")
        
        # Perform credit risk assessment using existing logic
        base_risk = 0.3
        
        # Risk factors based on transaction patterns
        if request.amount_mean > 50000:
            base_risk += 0.2
        if request.has_fraud_history:
            base_risk += 0.3
        if request.daily_transaction_count > 10:
            base_risk += 0.1
        if request.amount_sum > 500000:
            base_risk += 0.15
        
        risk_score = min(base_risk, 0.95)
        
        # Risk categories
        if risk_score >= 0.7:
            risk_category = "HIGH"
            recommendation = "Credit application requires additional verification and documentation. Consider lower credit limits."
        elif risk_score >= 0.4:
            risk_category = "MEDIUM"
            recommendation = "Standard credit evaluation process. Monitor transaction patterns closely."
        else:
            risk_category = "LOW"
            recommendation = "Low risk profile. Eligible for standard credit products and terms."
        
        # Generate SHAP-like explanations
        shap_values = {
            "amount_mean": 0.15 if request.amount_mean > 50000 else -0.1,
            "has_fraud_history": 0.3 if request.has_fraud_history else -0.05,
            "daily_transaction_count": 0.1 if request.daily_transaction_count > 10 else -0.05,
            "amount_sum": 0.15 if request.amount_sum > 500000 else -0.1,
            "daily_amount_avg": 0.05 if request.daily_amount_avg > 5000 else -0.05
        }
        
        assessment_notes = f"Risk assessment based on transaction patterns. Customer shows {risk_category.lower()} risk profile with score {risk_score:.3f}."
        
        # Save assessment to database
        assessment = CreditAssessment(
            customer_id=request.customer_id,
            amount_mean=request.amount_mean,
            amount_sum=request.amount_sum,
            amount_count=request.amount_count,
            daily_transaction_count=request.daily_transaction_count,
            daily_amount_avg=request.daily_amount_avg,
            has_fraud_history=request.has_fraud_history,
            risk_score=risk_score,
            risk_category=risk_category,
            shap_values=json.dumps(shap_values),
            assessment_notes=assessment_notes,
            recommendation=recommendation
        )
        
        db.add(assessment)
        db.commit()
        
        logger.info(f"Credit assessment completed for customer {request.customer_id}")
        
        return {
            "customer_id": request.customer_id,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "shap_values": shap_values,
            "assessment_notes": assessment_notes,
            "recommendation": recommendation,
            "assessment_id": assessment.id,
            "profile_created": not customer_exists and request.create_profile
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced credit risk prediction error: {e}")
        raise HTTPException(status_code=500, detail="Credit risk assessment failed")

@router.post("/chatbot/get-credit-assessment", response_model=ChatbotResponse)
async def chatbot_get_credit_assessment(request: ChatbotCreditRequest):
    """Chatbot endpoint to retrieve customer credit assessment data."""
    try:
        from database.connection import get_db_session
        from database.models import CustomerProfile, CreditAssessment
        
        # Direct database authentication check
        db = get_db_session()
        
        # Check if customer exists
        customer = db.query(CustomerProfile).filter(
            CustomerProfile.customer_id == request.customer_id,
            CustomerProfile.is_active == True
        ).first()
        
        if not customer:
            db.close()
            return ChatbotResponse(
                success=False,
                message=f"Customer {request.customer_id} not found."
            )
        
        # Verify password using SHA-256 hash
        import hashlib
        password_hash = hashlib.sha256(request.password.encode()).hexdigest()
        stored_hash = str(customer.hashed_password)
        
        # Debug logging with character analysis
        logger.info(f"Authentication attempt - Customer: {request.customer_id}")
        logger.info(f"Password received: '{request.password}' (length: {len(request.password)})")
        logger.info(f"Password bytes: {request.password.encode()}")
        logger.info(f"Computed hash: {password_hash}")
        logger.info(f"Stored hash: {stored_hash}")
        logger.info(f"Hashes match: {password_hash == stored_hash}")
        
        # Debug hash comparison (removed hardcoded credential)
        logger.info(f"Hash comparison result: {password_hash == stored_hash}")
        
        if password_hash != stored_hash:
            db.close()
            return ChatbotResponse(
                success=False,
                message=f"Authentication failed for customer {request.customer_id}. Invalid credentials."
            )
        
        # Get customer assessments
        assessments = db.query(CreditAssessment).filter(
            CreditAssessment.customer_id == request.customer_id
        ).order_by(CreditAssessment.created_at.desc()).limit(5).all()
        
        db.close()
        
        if not assessments:
            return ChatbotResponse(
                success=True,
                message=f"No credit assessments found for customer {request.customer_id}. Please complete a credit risk assessment first.",
                data={"assessments": []}
            )
        
        # Format assessment data for response
        assessment_list = []
        for assessment in assessments:
            assessment_list.append({
                'id': assessment.id,
                'risk_score': float(assessment.risk_score),
                'risk_category': assessment.risk_category,
                'created_at': assessment.created_at.isoformat(),
                'recommendation': assessment.recommendation or "Standard credit terms apply"
            })
        
        # Format response for chatbot
        latest = assessment_list[0]  # Most recent assessment
        total_assessments = len(assessment_list)
        
        message = f"""Credit Assessment Results for Customer {request.customer_id}:

Latest Assessment:
• Risk Score: {latest['risk_score']:.1%}
• Risk Category: {latest['risk_category']}
• Assessment Date: {latest['created_at'][:10]}

Recommendation: {latest['recommendation']}

Total Assessments: {total_assessments} stored in your profile
All assessments are securely linked to your customer account."""
        
        return ChatbotResponse(
            success=True,
            message=message,
            data={
                "customer_id": request.customer_id,
                "total_assessments": total_assessments,
                "latest_assessment": latest,
                "all_assessments": assessment_list
            }
        )
        
    except Exception as e:
        logger.error(f"Chatbot credit assessment error: {e}")
        return ChatbotResponse(
            success=False,
            message="Sorry, I encountered an error retrieving your credit assessment. Please try again later."
        )