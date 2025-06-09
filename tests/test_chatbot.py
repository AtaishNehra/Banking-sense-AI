"""
Tests for chatbot service.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from services.chatbot_service.main import ChatbotService, app
from fastapi.testclient import TestClient

class TestChatbotService:
    """
    Test cases for chatbot service.
    """
    
    def test_chatbot_initialization(self):
        """
        Test that chatbot service initializes properly.
        """
        try:
            chatbot = ChatbotService()
            
            # Chatbot should initialize even if model loading fails
            assert chatbot is not None
            
        except Exception as e:
            pytest.fail(f"Chatbot initialization failed: {e}")
    
    @patch('services.chatbot_service.main.ChatbotService.generate_response')
    def test_chat_endpoint_returns_json_with_response(self, mock_generate):
        """
        Test that chat endpoint returns JSON with response key.
        """
        # Mock the LLM to echo input
        mock_generate.return_value = "This is a test response"
        
        client = TestClient(app)
        
        # First, get authentication token
        login_data = {"username": "analyst", "password": "analyst123"}
        token_response = client.post("/token", data=login_data)
        
        assert token_response.status_code == 200
        token = token_response.json()["access_token"]
        
        # Make chat request
        chat_data = {
            "user_id": "test_user",
            "message": "Hello, can you help me with fraud detection?"
        }
        
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post("/chat", json=chat_data, headers=headers)
        
        assert response.status_code == 200
        
        result = response.json()
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
        assert result["user_id"] == "test_user"
    
    def test_fallback_response_generation(self):
        """
        Test fallback response generation when model is not available.
        """
        chatbot = ChatbotService()
        chatbot.model = None  # Force fallback mode
        
        test_messages = [
            "Hello",
            "What is fraud detection?",
            "Tell me about my account balance",
            "I need help with credit",
            "How do I transfer money?"
        ]
        
        for message in test_messages:
            response = chatbot.generate_response("test_user", message)
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert "error" not in response.lower() or "help" in response.lower()
    
    def test_conversation_history_management(self):
        """
        Test that conversation history is properly managed.
        """
        chatbot = ChatbotService()
        user_id = "test_conversation_user"
        
        # Send multiple messages
        messages = [
            "Hello",
            "What is fraud detection?",
            "Can you help me with my account?",
            "Thank you"
        ]
        
        for message in messages:
            response = chatbot.generate_response(user_id, message)
            assert isinstance(response, str)
            assert len(response) > 0
        
        # Check that conversation was stored
        from services.chatbot_service.main import conversations
        assert user_id in conversations
        assert len(conversations[user_id]) > 0
        
        # Each message should create 2 entries (user + assistant)
        assert len(conversations[user_id]) == len(messages) * 2
    
    def test_authentication_required(self):
        """
        Test that authentication is required for chat endpoint.
        """
        client = TestClient(app)
        
        chat_data = {
            "user_id": "test_user",
            "message": "Hello"
        }
        
        # Request without authentication should fail
        response = client.post("/chat", json=chat_data)
        assert response.status_code == 401
    
    def test_invalid_authentication(self):
        """
        Test that invalid authentication is rejected.
        """
        client = TestClient(app)
        
        chat_data = {
            "user_id": "test_user",
            "message": "Hello"
        }
        
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.post("/chat", json=chat_data, headers=headers)
        assert response.status_code == 401
    
    def test_banking_context_responses(self):
        """
        Test that chatbot provides appropriate responses to banking queries.
        """
        chatbot = ChatbotService()
        
        banking_queries = [
            ("fraud", "fraud"),
            ("suspicious activity", "security"),
            ("account balance", "account"),
            ("credit score", "credit"),
            ("transfer money", "transfer"),
            ("hello", "hello")
        ]
        
        for query, expected_keyword in banking_queries:
            response = chatbot._generate_fallback_response(query)
            
            assert isinstance(response, str)
            assert len(response) > 0
            # Response should be relevant to banking
            assert any(word in response.lower() for word in [
                "banking", "account", "fraud", "security", "credit", 
                "transfer", "help", "assist", "service"
            ])
    
    def test_health_endpoint(self):
        """
        Test health check endpoint.
        """
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        result = response.json()
        assert "status" in result
        assert result["status"] == "healthy"
        assert "service" in result

if __name__ == "__main__":
    pytest.main([__file__])
