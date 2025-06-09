"""
LLM-powered chatbot service for banking queries.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Simple user database
fake_users_db = {
    "analyst": {
        "username": "analyst",
        "password": "analyst123",  # In production, use hashed passwords
        "role": "analyst"
    },
    "manager": {
        "username": "manager", 
        "password": "manager123",
        "role": "manager"
    }
}

# In-memory conversation storage
conversations: Dict[str, List[Dict]] = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    user_id: str
    timestamp: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatbotService:
    """
    LLM-powered chatbot service.
    """
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """
        Load the LLM model for chatbot responses.
        """
        try:
            # Use a smaller model that can run on CPU/limited resources
            model_name = "microsoft/DialoGPT-medium"  # Fallback model
            
            # Try to use a more capable model if available
            alternative_models = [
                "microsoft/DialoGPT-large",
                "facebook/blenderbot-400M-distill",
                "microsoft/DialoGPT-medium"
            ]
            
            for model_name in alternative_models:
                try:
                    logger.info(f"Attempting to load model: {model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                    
                    # Add padding token if not present
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    logger.info(f"Successfully loaded model: {model_name}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            if self.model is None:
                raise Exception("Failed to load any suitable model")
                
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            # Fallback to a simple echo bot
            self.model = None
            self.tokenizer = None
    
    def generate_response(self, user_id: str, message: str) -> str:
        """
        Generate chatbot response to user message.
        """
        # Add banking context to the conversation
        banking_context = """You are a helpful banking assistant. You can help with:
        - Account inquiries
        - Transaction history
        - Fraud detection questions
        - Credit risk information
        - General banking services
        
        Please provide helpful, accurate responses about banking topics."""
        
        # Get conversation history
        if user_id not in conversations:
            conversations[user_id] = []
        
        conversation = conversations[user_id]
        
        # Add current message to conversation
        conversation.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            if self.model is not None:
                # Use the loaded model
                response = self._generate_with_model(message, conversation)
            else:
                # Fallback to rule-based responses
                response = self._generate_fallback_response(message)
            
            # Add response to conversation
            conversation.append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 exchanges to manage memory
            if len(conversation) > 20:
                conversation = conversation[-20:]
                conversations[user_id] = conversation
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing your request right now. Please try again later."
    
    def _generate_with_model(self, message: str, conversation: List[Dict]) -> str:
        """
        Generate response using the loaded LLM model.
        """
        try:
            # Prepare conversation history for the model
            conversation_text = ""
            for turn in conversation[-6:]:  # Use last 3 exchanges
                if turn["role"] == "user":
                    conversation_text += f"User: {turn['content']}\n"
                else:
                    conversation_text += f"Assistant: {turn['content']}\n"
            
            conversation_text += "Assistant:"
            
            # Tokenize and generate
            inputs = self.tokenizer.encode(conversation_text, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            if not response:
                return self._generate_fallback_response(message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in model generation: {e}")
            return self._generate_fallback_response(message)
    
    def _generate_fallback_response(self, message: str) -> str:
        """
        Generate simple rule-based responses as fallback.
        """
        message_lower = message.lower()
        
        # Banking-specific responses
        if any(word in message_lower for word in ["fraud", "suspicious", "security"]):
            return "For fraud-related concerns, please contact our security team immediately at 1-800-SECURITY. We take all security matters very seriously."
        
        elif any(word in message_lower for word in ["balance", "account", "money"]):
            return "I can help you with account inquiries. For real-time balance information, please log into your online banking or visit one of our branches."
        
        elif any(word in message_lower for word in ["credit", "loan", "mortgage"]):
            return "For credit and lending services, I'd be happy to direct you to our lending specialists. They can help assess your credit options and requirements."
        
        elif any(word in message_lower for word in ["transfer", "payment", "send"]):
            return "For transfers and payments, you can use our online banking platform or mobile app. Make sure to verify recipient details before confirming any transfers."
        
        elif any(word in message_lower for word in ["hello", "hi", "help"]):
            return "Hello! I'm your banking assistant. I can help you with account questions, fraud concerns, credit information, and general banking services. How can I assist you today?"
        
        else:
            return f"Thank you for your inquiry about '{message}'. I'm here to help with banking-related questions. Could you please provide more specific details about what you need assistance with?"

# Initialize the chatbot service
chatbot = ChatbotService()

# FastAPI app
app = FastAPI(title="Banking Chatbot Service", version="1.0.0")

def authenticate_user(username: str, password: str):
    """
    Authenticate user credentials.
    """
    user = fake_users_db.get(username)
    if user and user["password"] == password:
        return user
    return None

def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Get current authenticated user.
    """
    # Simple token validation (in production, use proper JWT)
    for username, user_data in fake_users_db.items():
        if token == f"token_{username}":
            return user_data
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials"
    )

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 login endpoint.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Simple token (in production, use proper JWT)
    access_token = f"token_{user['username']}"
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Chat endpoint for LLM-powered responses.
    """
    try:
        # Generate response
        response = chatbot.generate_response(request.user_id, request.message)
        
        return ChatResponse(
            response=response,
            user_id=request.user_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "service": "chatbot"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
