#!/usr/bin/env python3
"""
Pydantic models for Stock Optimizer GUI.
"""

from typing import List, Optional
from pydantic import BaseModel


class OptimizationRequest(BaseModel):
    """Request model for optimization endpoint."""
    symbols: List[str]
    start_date: str
    end_date: str
    backtest_start_date: Optional[str] = None
    backtest_end_date: Optional[str] = None
    capital: float = 100000
    n_trials: int = 200
    optimization_metric: str = "sharpe"
    use_gpu: bool = False
    smart_timing: bool = False
    timing_approach: str = "sequential"
    algo: str = "default"
