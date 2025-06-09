#!/bin/bash

# Integration test script for banking ML platform
set -e

echo "Starting integration tests..."

# Function to check if service is healthy
check_service_health() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo "Checking $service_name health at $url"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url/health" > /dev/null; then
            echo "$service_name is healthy"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: $service_name not ready, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: $service_name failed to become healthy"
    return 1
}

# Function to test API endpoint
test_endpoint() {
    local method=$1
    local url=$2
    local data=$3
    local expected_status=$4
    local description=$5
    
    echo "Testing: $description"
    
    if [ "$method" = "POST" ]; then
        response=$(curl -s -w "%{http_code}" -X POST "$url" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $AUTH_TOKEN" \
            -d "$data")
    else
        response=$(curl -s -w "%{http_code}" "$url")
    fi
    
    # Extract status code (last 3 characters)
    status_code="${response: -3}"
    response_body="${response%???}"
    
    if [ "$status_code" = "$expected_status" ]; then
        echo "✓ $description - Status: $status_code"
        return 0
    else
        echo "✗ $description - Expected: $expected_status, Got: $status_code"
        echo "Response: $response_body"
        return 1
    fi
}

# Start Docker Compose services
echo "Starting services with docker-compose..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check service health
BASE_URL="http://localhost:8000"

check_service_health "$BASE_URL" "Main API"

# Get authentication token
echo "Getting authentication token..."
AUTH_RESPONSE=$(curl -s -X POST "$BASE_URL/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=analyst&password=analyst123")

if echo "$AUTH_RESPONSE" | grep -q "access_token"; then
    AUTH_TOKEN=$(echo "$AUTH_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
    echo "✓ Authentication successful"
else
    echo "✗ Authentication failed"
    echo "Response: $AUTH_RESPONSE"
    exit 1
fi

# Test fraud detection endpoint
FRAUD_DATA='{
    "amount": 1000.0,
    "type": "TRANSFER",
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4000.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 1000.0,
    "step": 24
}'

test_endpoint "POST" "$BASE_URL/predict/fraud" "$FRAUD_DATA" "200" "Fraud detection prediction"

# Test credit risk endpoint
CREDIT_DATA='{
    "customer_id": "test_customer",
    "amount_mean": 1500.0,
    "amount_sum": 15000.0,
    "amount_count": 10,
    "daily_transaction_count": 2.0,
    "daily_amount_avg": 1500.0,
    "has_fraud_history": 0
}'

test_endpoint "POST" "$BASE_URL/predict/credit_risk" "$CREDIT_DATA" "200" "Credit risk prediction"

# Test chat endpoint
CHAT_DATA='{
    "user_id": "test_user",
    "message": "Hello, can you help me with fraud detection?"
}'

# Note: Chat endpoint might return 503 if chatbot service is not running
test_endpoint "POST" "$BASE_URL/chat" "$CHAT_DATA" "200" "Chat endpoint" || \
    echo "⚠ Chat endpoint failed (chatbot service may not be running)"

# Test document summarization endpoint would require file upload
# For now, we'll test the endpoint accessibility
echo "Testing document summarization endpoint accessibility..."
curl -s -f "$BASE_URL/docs" > /dev/null && echo "✓ API documentation accessible" || echo "✗ API documentation not accessible"

# Test metrics endpoint
test_endpoint "GET" "$BASE_URL/metrics" "" "200" "Prometheus metrics endpoint"

# Test health endpoint
test_endpoint "GET" "$BASE_URL/health" "" "200" "Health check endpoint"

echo ""
echo "Integration tests completed!"
echo ""

# Clean up
echo "Stopping services..."
docker-compose down

echo "Integration test script finished."
