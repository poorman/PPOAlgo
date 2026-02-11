#!/bin/bash
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then 
    nvidia-smi || echo "GPU detected but nvidia-smi failed (standard in WSL2)"
else 
    echo "nvidia-smi not found"
fi
echo "Running GPU warmup..."
python3 /app/gpu_warmup.py
echo "Starting server with 1 worker..."
exec uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 60
