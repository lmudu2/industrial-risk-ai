#!/bin/bash
# Start backend api
source /Users/puneethsmacbook/.venv/bin/activate
uvicorn backend.chatbot_service:app --host 127.0.0.1 --port 8000 &
echo "Uvicorn started on port 8000"
