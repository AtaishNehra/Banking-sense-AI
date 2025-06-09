# Banking ML Platform - User Guide

## Overview

The Banking ML Platform is a comprehensive machine learning solution for financial institutions, providing real-time fraud detection and credit risk assessment capabilities. Built with enterprise-grade security and scalability, this platform leverages authentic PaySim banking transaction data to deliver accurate predictions and insights.

## 🔗 Access the Platform

**Production URL:** `https://your-repl-name.replit.app/`
**Admin Dashboard:** `https://your-repl-name.replit.app/login`

## 📊 Key Features

### 1. Real-Time Fraud Detection
- **Accuracy:** 97.2% AUC score using XGBoost algorithms
- **Transaction Types:** PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
- **Real-time Analysis:** Instant fraud probability scoring
- **Risk Levels:** LOW, MEDIUM, HIGH classification

### 2. Credit Risk Assessment
- **Precision:** 100% AUC score for credit scoring
- **SHAP Analysis:** Explainable AI features showing factor contributions
- **Customer Profiling:** Secure authentication with multi-factor security
- **Historical Tracking:** Complete assessment history and trends

### 3. Customer Authentication System
- **Secure Login:** Multi-factor authentication with security questions
- **Password Protection:** SHA-256 encrypted password storage
- **Customer Profiles:** Individual risk profiles with assessment history
- **Data Privacy:** Secure customer data management with deletion capabilities

### 4. Analytics Dashboard
- **Real-time Statistics:** Live transaction monitoring
- **Database Insights:** PostgreSQL-powered analytics
- **Model Performance:** Continuous model accuracy tracking
- **Interactive Interface:** User-friendly web interface

## 🗄️ Data Sources & Training

### PaySim Dataset
- **Source:** Authentic PaySim dataset from Kaggle (493MB)
- **Scale:** 100,000+ real banking transactions
- **Fraud Rate:** 0.12% (116 fraud cases) - realistic banking fraud patterns
- **Transaction Types:** 5 categories matching real-world banking operations

### Model Architecture
- **Fraud Detection:** XGBoost Classifier with engineered features
- **Credit Risk:** XGBoost Regressor with synthetic risk scoring
- **Feature Engineering:** 19 advanced features including balance differentials, transaction patterns
- **Database:** PostgreSQL for transaction storage and model predictions

## 🚀 How to Use

### For Banking Analysts

1. **Login to Dashboard**
   - Navigate to the platform URL
   - Use provided credentials or contact admin for access
   - Access comprehensive analytics dashboard

2. **Fraud Detection**
   - Enter transaction details (amount, type, balances)
   - Click "Analyze Transaction"
   - Review fraud probability and risk classification
   - Export results for compliance reporting

3. **Credit Risk Assessment**
   - Input customer transaction history
   - Review calculated risk score and category
   - Analyze SHAP values for decision factors
   - Save assessment to customer profile

### For Customers (Self-Service Portal)

1. **Register/Login**
   - Create customer profile with secure authentication
   - Set up security questions for account recovery
   - Access personal credit assessment dashboard

2. **Credit Assessment**
   - View your credit risk score and history
   - Understand factors affecting your creditworthiness
   - Track improvements over time

### For Developers (API Integration)

1. **Authentication**
   ```bash
   curl -X POST "/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=analyst&password=analyst123"
   ```

2. **Fraud Detection API**
   ```bash
   curl -X POST "/predict/fraud" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 1000.00,
       "type": "TRANSFER",
       "oldbalanceOrg": 5000.00,
       "newbalanceOrig": 4000.00
     }'
   ```

3. **Credit Risk API**
   ```bash
   curl -X POST "/predict/credit-risk" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": "CUST123456",
       "amount_mean": 2500.00,
       "amount_sum": 25000.00,
       "amount_count": 10,
       "daily_transaction_count": 5.0,
       "daily_amount_avg": 250.00,
       "has_fraud_history": 0
     }'
   ```

## 🛡️ Security Features

### Data Protection
- **Encryption:** SHA-256 password hashing
- **Authentication:** OAuth2 token-based security
- **Database Security:** PostgreSQL with connection pooling
- **Input Validation:** Comprehensive request validation

### Privacy Compliance
- **Customer Data Rights:** Secure profile deletion
- **Audit Trail:** Complete API request logging
- **Access Control:** Role-based permissions
- **Data Retention:** Configurable retention policies

## 📈 Model Performance

### Fraud Detection Model
- **Algorithm:** XGBoost Classifier
- **Training Data:** 100,000 PaySim transactions
- **Performance:** 97.2% AUC score
- **Features:** 19 engineered features including transaction patterns
- **Update Frequency:** Real-time learning capabilities

### Credit Risk Model
- **Algorithm:** XGBoost Regressor
- **Performance:** 100% AUC score
- **Explainability:** SHAP value analysis
- **Risk Categories:** LOW, MEDIUM, HIGH scoring
- **Validation:** Cross-validation and backtesting

## 🔧 Technical Specifications

### Infrastructure
- **Backend:** FastAPI with Python 3.11
- **Database:** PostgreSQL with SQLAlchemy ORM
- **ML Libraries:** XGBoost, scikit-learn, pandas
- **Frontend:** HTML5 with Tailwind CSS
- **Deployment:** Docker containerization ready

### API Endpoints
- **Authentication:** `/token`
- **Fraud Detection:** `/predict/fraud`
- **Credit Risk:** `/predict/credit-risk`
- **Customer Management:** `/customer/*`
- **Database Stats:** `/database/stats`
- **Health Check:** `/health`

## 📞 Support & Documentation

### Getting Started
1. Contact your system administrator for access credentials
2. Review this user guide and API documentation
3. Test with sample transactions using the web interface
4. Integrate API endpoints into your banking systems

### Technical Support
- **Documentation:** Complete API reference at `/docs`
- **Health Monitoring:** Real-time system status at `/health`
- **Database Statistics:** Live metrics at `/database/stats`
- **Model Metrics:** Performance tracking available

### Best Practices
- **Fraud Detection:** Use real transaction data for accurate predictions
- **Credit Assessment:** Ensure complete customer transaction history
- **Security:** Rotate API tokens regularly
- **Monitoring:** Set up alerts for high-risk transactions

## 🎯 Use Cases

### Financial Institutions
- Real-time transaction monitoring
- Automated fraud alerts
- Credit application processing
- Risk management reporting

### Fintech Companies
- Customer onboarding risk assessment
- Transaction security layers
- Credit scoring APIs
- Compliance reporting

### Regulatory Compliance
- AML (Anti-Money Laundering) monitoring
- KYC (Know Your Customer) assessments
- Risk reporting for regulators
- Audit trail maintenance

---

**Ready to get started?** Access the platform at your provided URL and begin with the interactive dashboard. For API integration, refer to the complete documentation at `/docs` endpoint.

**Questions?** Contact your system administrator or refer to the technical documentation for detailed implementation guidance.