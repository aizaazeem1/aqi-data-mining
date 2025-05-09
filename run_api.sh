#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install requirements if not already installed
pip install -r requirements.txt

# Run the FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload 