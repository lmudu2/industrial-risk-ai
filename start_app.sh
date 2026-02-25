#!/bin/bash
echo "🚀 Starting EAM AI Platform..."

# Start the FastAPI backend in the background
echo "Starting backend service on port 8000..."
cd backend
python3 -m uvicorn main:app --reload &
BACKEND_PID=$!

# Move back to root directory
cd ..

# Wait a moment for backend to initialize
sleep 2

# Start the Streamlit frontend
echo "Starting Streamlit frontend on port 8501..."
cd frontend
streamlit run app.py

# When Streamlit exits, also kill the backend
kill $BACKEND_PID
echo "🛑 EAM AI Platform stopped."
