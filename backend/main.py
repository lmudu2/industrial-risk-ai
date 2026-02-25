from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
import models  # This imports all our table definitions

# ─────────────────────────────────────────────
# CREATE THE FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Enterprise Asset Management Platform",
    description="AI-Powered Predictive Maintenance System",
    version="1.0.0"
)

# ─────────────────────────────────────────────
# CORS MIDDLEWARE (Allows frontend to talk to backend)
# ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins (OK for demo)
    allow_methods=["*"],      # Allow all HTTP methods
    allow_headers=["*"],      # Allow all headers
)

# ─────────────────────────────────────────────
# CREATE DATABASE TABLES ON STARTUP
# ─────────────────────────────────────────────

@app.on_event("startup")
def startup():
    """
    This function runs ONCE when the server starts
    
    It looks at all the models (Industries, Assets, Sensors, etc.)
    and creates the actual tables in the database
    """
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    print("📊 Tables: industries, assets, sensors, maintenance_records, work_orders, cost_records")

# ─────────────────────────────────────────────
# ROOT ENDPOINT (Just to test server is running)
# ─────────────────────────────────────────────

@app.get("/")
def root():
    """
    Home endpoint - returns basic info
    
    Visit: http://127.0.0.1:8000
    """
    return {
        "message": "EAM Platform API is running! ✅",
        "status": "healthy",
        "version": "1.0.0"
    }


# ─────────────────────────────────────────────
# CHAT ENDPOINT
# ─────────────────────────────────────────────

from pydantic import BaseModel
from typing import Optional, Dict, Any
from chatbot_service import generate_response

class ChatRequest(BaseModel):
    query: str
    context_data: Optional[Dict[str, Any]] = None

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    NLP Chatbot endpoint
    Uses LLM to convert natural language to SQL
    """
    response = generate_response(request.query, context=request.context_data)
    return {"response": response}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}