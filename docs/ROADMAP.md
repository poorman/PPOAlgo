# Roadmap

## Completed

- [x] GPU-accelerated grid search (CuPy)
- [x] Rust Rayon-parallel grid search
- [x] Hybrid GPU/CPU dispatch
- [x] VWAP 4-condition entry signal
- [x] Buy trigger as functional 5th condition (ADR-001)
- [x] PostgreSQL history persistence
- [x] WebSocket real-time progress
- [x] Widesurf V2 data source integration
- [x] Batch optimization (300+ stocks)

## Planned

### Short-term
- [ ] Optimize alpha/gamma as tunable parameters (currently hardcoded 0.50/0.25)
- [ ] Add stop-loss parameter to VWAP Rust grid search
- [ ] Walk-forward validation (train on period A, test on period B)
- [ ] Export optimization results to CSV/Excel

### Medium-term
- [ ] Multi-timeframe VWAP (5-min, 15-min aggregation windows)
- [ ] Portfolio-level optimization (optimize across multiple stocks jointly)
- [ ] Risk-adjusted position sizing (Kelly criterion or volatility-scaled)
- [ ] Automated test suite for optimizer correctness

### Long-term
- [ ] Reinforcement learning strategy (PPO agent)
- [ ] LSTM price predictor integration
- [ ] Live paper trading with Alpaca API
- [ ] Multi-GPU support for batch optimization
