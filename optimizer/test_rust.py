#!/usr/bin/env python3
"""Quick test of the Rust optimizer bridge."""
from rust_optimizer import run_rust_vwap_optimization, RUST_BINARY, RUST_AVAILABLE
import json

print(f"Rust binary: {RUST_BINARY}")
print(f"Available: {RUST_AVAILABLE}")

bars = [
    {"o": 100, "h": 105, "l": 98, "c": 103, "price_10am": 102, "vwap": 101, "vwap_940": 100.5, "or_high": 103, "or_low": 99},
    {"o": 103, "h": 108, "l": 101, "c": 106, "price_10am": 104.5, "vwap": 103.5, "vwap_940": 103, "or_high": 106, "or_low": 101},
]

result = run_rust_vwap_optimization(bars, 100000, (0.01, 0.03, 0.01), (0.01, 0.03, 0.01), "total_return")
print(json.dumps(result, indent=2))
print("✅ Rust optimizer test PASSED")
