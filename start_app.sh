#!/bin/bash
echo "🚀 Starting EAM AI Platform..."

# Activate venv2 if it exists
if [ -f "venv2/bin/activate" ]; then
    source venv2/bin/activate
fi

# 1. Start the FastAPI backend in the background
echo "Starting backend service on port 8000..."
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# 2. Link monitoring
echo "Starting Streamlit frontend on port 8501..."
# We run from root to ensure relative imports WORK correctly
python3 -m streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1

# When Streamlit exits, also kill the backend
kill $BACKEND_PID
echo "🛑 EAM AI Platform stopped."
