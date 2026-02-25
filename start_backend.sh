#!/bin/bash
# Start backend api
# Try to activate local venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

python3 -m uvicorn backend.chatbot_service:app --host 127.0.0.1 --port 8000 &
echo "Uvicorn started on port 8000"
