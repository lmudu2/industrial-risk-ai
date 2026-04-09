#!/bin/bash
# Start backend api

# Activate venv2 if it exists
if [ -f "venv2/bin/activate" ]; then
    source venv2/bin/activate
fi

python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
