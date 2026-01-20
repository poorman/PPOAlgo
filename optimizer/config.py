#!/usr/bin/env python3
"""
Configuration module for Stock Optimizer GUI.
Contains environment variables, database settings, and constants.
"""

import os

# API Configuration
PPOALGO_API = os.getenv("PPOALGO_API", "http://ppoalgo_api_1:8000")
ALPACA_API_KEY_ID = os.getenv("ALPACA_API_KEY_ID", "PKYAH5AZB64NRG2UB6QT3ISHPG")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
RESULTS_DIR = "/app/results"

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "ppoalgo"),
    "user": os.getenv("POSTGRES_USER", "ppoalgo"),
    "password": os.getenv("POSTGRES_PASSWORD", "ppoalgo"),
    "host": os.getenv("POSTGRES_HOST", "ppoalgo_db_1"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

# Widesurf API configuration
WIDESURF_API_KEY = os.getenv("WIDESURF_API_KEY", "GV6VaiYR7Vp1ZpLnXKCe6rt1Q4c0Ymv34I5Io5jyx0I")
WIDESURF_API_URL = os.getenv("WIDESURF_API_URL", "http://10.0.0.94:8020")

# Massive.com API configuration
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY", "vQtz66lpyexhpplKWhLL7rOXdfnClQsh")
MASSIVE_API_URL = "https://api.massive.com"

# GPU availability flags (set at import time)
GPU_AVAILABLE = False
GPU_BACKTEST_AVAILABLE = False

try:
    import cupy
    GPU_AVAILABLE = True
except ImportError:
    pass

try:
    from gpu_backtester import GPUBacktester
    GPU_BACKTEST_AVAILABLE = True
except ImportError:
    pass
