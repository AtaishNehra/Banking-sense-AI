# Banking ML Platform - Demo Credentials

## Public Demo Access

The platform is now secured with proper authentication. Use these demo credentials for testing:

### Demo Customer Accounts

**Account 1: Standard Customer**
- Customer ID: `DEMO001`
- Password: `demo123`
- Description: Regular customer with transaction history

**Account 2: High-Risk Customer** 
- Customer ID: `DEMO002`
- Password: `demo456`
- Description: Customer with fraud history for testing risk assessment

**Account 3: Premium Customer**
- Customer ID: `DEMO003` 
- Password: `demo789`
- Description: High-value customer with extensive transaction patterns

### Admin/Analyst Access

**API Testing:**
- Username: `analyst`
- Password: `analyst123`
- Use for API endpoint testing via `/docs` interface

### Security Features

- All passwords are SHA-256 encrypted in database
- Multi-factor authentication with security questions
- Session-based authentication for web interface
- Protected dashboard route requiring valid login

### How to Test

1. **Visit:** https://banking-intelligence-AtaishNehra.replit.app/
2. **Login:** Use any demo credential above
3. **Test Features:** 
   - Fraud detection with sample transactions
   - Credit risk assessment with customer data
   - View database analytics and insights
4. **API Testing:** Access `/docs` for complete API documentation

### Sample Test Data

**Fraud Detection:**
- High Risk: Amount $50,000, Type: CASH_OUT
- Medium Risk: Amount $10,000, Type: TRANSFER  
- Low Risk: Amount $500, Type: PAYMENT

**Credit Assessment:**
- High Risk: Large amounts, frequent transactions, fraud history
- Low Risk: Moderate amounts, regular patterns, no fraud history

The platform automatically redirects to secure login and protects all sensitive functionality.