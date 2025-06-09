"""
Database models for banking ML platform.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class User(Base):
    """User accounts for the banking platform."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    transactions = relationship("Transaction", back_populates="user")
    fraud_predictions = relationship("FraudPrediction", back_populates="user")
    credit_assessments = relationship("CreditAssessment", back_populates="user")

class Transaction(Base):
    """Banking transactions from PaySim dataset."""
    __tablename__ = "transactions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    # PaySim transaction fields
    step = Column(Integer, nullable=False)
    transaction_type = Column(String(20), nullable=False)  # PAYMENT, TRANSFER, etc.
    amount = Column(Float, nullable=False)
    name_orig = Column(String(50), nullable=False)
    old_balance_orig = Column(Float, nullable=False)
    new_balance_orig = Column(Float, nullable=False)
    name_dest = Column(String(50), nullable=False)
    old_balance_dest = Column(Float, nullable=False)
    new_balance_dest = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    is_flagged_fraud = Column(Boolean, nullable=False)
    
    # Engineered features
    amount_log = Column(Float)
    balance_diff_orig = Column(Float)
    balance_diff_dest = Column(Float)
    is_round_amount = Column(Boolean)
    is_transfer_or_cash_out = Column(Boolean)
    orig_balance_zero = Column(Boolean)
    dest_balance_zero = Column(Boolean)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="transactions")
    fraud_predictions = relationship("FraudPrediction", back_populates="transaction")

class FraudPrediction(Base):
    """Fraud detection predictions."""
    __tablename__ = "fraud_predictions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    transaction_id = Column(String, ForeignKey("transactions.id"), nullable=True)
    
    # Input features
    amount = Column(Float, nullable=False)
    transaction_type = Column(String(20), nullable=False)
    old_balance_orig = Column(Float)
    new_balance_orig = Column(Float)
    old_balance_dest = Column(Float)
    new_balance_dest = Column(Float)
    step = Column(Integer)
    
    # Prediction results
    fraud_probability = Column(Float, nullable=False)
    is_fraud_predicted = Column(Boolean, nullable=False)
    risk_level = Column(String(10), nullable=False)  # LOW, MEDIUM, HIGH
    model_version = Column(String(20), default="v1.0")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="fraud_predictions")
    transaction = relationship("Transaction", back_populates="fraud_predictions")

class CustomerProfile(Base):
    """Customer profiles with authentication."""
    __tablename__ = "customer_profiles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Security questions
    security_question_1 = Column(String(255), nullable=False)
    security_answer_1_hash = Column(String(255), nullable=False)
    security_question_2 = Column(String(255), nullable=False)
    security_answer_2_hash = Column(String(255), nullable=False)
    
    # Profile metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    
    # Relationships
    credit_assessments = relationship("CreditAssessment", back_populates="customer_profile")

class CreditAssessment(Base):
    """Credit risk assessments."""
    __tablename__ = "credit_assessments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    customer_id = Column(String(50), ForeignKey("customer_profiles.customer_id"), nullable=False)
    
    # Input features
    amount_mean = Column(Float, nullable=False)
    amount_sum = Column(Float, nullable=False)
    amount_count = Column(Integer, nullable=False)
    daily_transaction_count = Column(Float, nullable=False)
    daily_amount_avg = Column(Float, nullable=False)
    has_fraud_history = Column(Integer, nullable=False)
    
    # Assessment results
    risk_score = Column(Float, nullable=False)
    risk_category = Column(String(20), nullable=False)
    shap_values = Column(Text)  # JSON string of SHAP values
    model_version = Column(String(20), default="v1.0")
    
    # Additional customer context
    assessment_notes = Column(Text)
    recommendation = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="credit_assessments")
    customer_profile = relationship("CustomerProfile", back_populates="credit_assessments")

class DeletedCustomerData(Base):
    """Archive for deleted customer data (admin access only)."""
    __tablename__ = "deleted_customer_data"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    original_customer_id = Column(String(50), nullable=False)
    deletion_reason = Column(String(100), nullable=False)
    
    # Archived assessment data (JSON)
    archived_assessments = Column(Text, nullable=False)
    archived_profile_data = Column(Text, nullable=False)
    
    # Deletion metadata
    deleted_at = Column(DateTime(timezone=True), server_default=func.now())
    deleted_by_system = Column(Boolean, default=True)

class ModelMetrics(Base):
    """Model performance metrics."""
    __tablename__ = "model_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    # Performance metrics
    auc_score = Column(Float)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Training metadata
    training_data_size = Column(Integer)
    feature_count = Column(Integer)
    training_time_seconds = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class APILog(Base):
    """API request logging."""
    __tablename__ = "api_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Float)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")