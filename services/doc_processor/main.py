"""
FastAPI service for document processing.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn

from .summarizer import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingResponse(BaseModel):
    summary: str
    key_clauses: list
    document_name: str

# Initialize document processor
processor = DocumentProcessor()

# FastAPI app
app = FastAPI(title="Document Processing Service", version="1.0.0")

@app.post("/summarize", response_model=ProcessingResponse)
async def summarize_document(file: UploadFile = File(...)):
    """
    Upload and process document for summarization and AML clause extraction.
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.doc'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Process document
        result = processor.process_document(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {result['error']}"
            )
        
        return ProcessingResponse(
            summary=result["summary"],
            key_clauses=result["key_clauses"],
            document_name=file.filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "service": "document_processor"}

@app.get("/")
async def root():
    """
    Root endpoint with service information.
    """
    return {
        "service": "Document Processing Service",
        "version": "1.0.0",
        "endpoints": {
            "POST /summarize": "Upload and process documents",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
