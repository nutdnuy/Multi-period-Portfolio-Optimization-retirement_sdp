# Retirement SDP — Multi-Period Portfolio Optimization

Optimises a retirement portfolio across multiple life phases using **Stochastic Dynamic Programming (SDP)** with real market data from Alpha Vantage.

## Features

| Feature | Details |
|---|---|
| **Market data** | Monthly adjusted prices (SPY, AGG, GLD) via Alpha Vantage; local Parquet cache |
| **Inflation** | CPI-based or fixed-rate real cash-flow deflators |
| **SDP engine** | CRRA utility, log-spaced 500-node wealth grid, 100-portfolio efficient frontier |
| **Bellman** | Vectorised backward induction with Monte Carlo (2 000 samples/step) |
| **PDF propagation** | Gaussian KDE wealth distributions from 20 000 forward paths |
| **Ruin analysis** | Empirical lifetime ruin probability; absorbing ruin state |
| **Safe Withdrawal Rate** | Brent root-finding over withdrawal rates |
| **Figures** | Efficient frontier, wealth PDF panels, ruin-probability curve |

---

## Project Layout

```
retirement_sdp/
├── config/
│   ├── config.example.yaml     # Template — copy and fill in your key
│   └── config.yaml             # Your local config (git-ignored)
├── src/retirement_sdp/
│   ├── data/                   # API client, Parquet cache, preprocessor
│   ├── models/                 # Return model, inflation, cash flows
│   ├── sdp/                    # State/action spaces, utility, Bellman, policy
│   ├── analysis/               # PDF propagation, ruin, SWR
│   └── visualization/          # Three output figures
├── outputs/
│   ├── cache/                  # Parquet price data
│   └── figures/                # PNG outputs
├── main.py                     # CLI entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get an Alpha Vantage API key

Free tier at <https://www.alphavantage.co/support/#api-key> (25 requests/day).

### 3. Configure

```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml — set your api_key
```

### 4. Run

```bash
# Full run (fetches data, solves SDP, generates figures)
python main.py

# Use cached data only (no API calls)
python main.py --skip-fetch

# Fast debug mode (smaller grid / fewer samples)
python main.py --fast

# Custom config location
python main.py --config my_config.yaml
```

Expected runtime: **60–180 s** on a modern laptop (full parameters).

---

## Configuration Reference

```yaml
alpha_vantage:
  api_key: "YOUR_KEY"           # Required

assets:
  tickers: ["SPY", "AGG", "GLD"]

phases:                         # Life-phase schedule
  - name: "Early Career"
    years: 15
    type: accumulation           # or drawdown
  ...

cashflow:
  initial_wealth: 100_000
  annual_contribution: 20_000   # Real dollars, inflation-adjusted
  annual_withdrawal: 50_000

sdp:
  wealth_grid_points: 500       # Nodes on log-spaced grid
  num_portfolios: 100           # Efficient-frontier portfolios
  gamma: 3.0                    # CRRA risk aversion (higher = more conservative)
  beta: 0.97                    # Annual discount factor
  mc_samples: 2_000             # Monte Carlo samples per Bellman step

analysis:
  num_paths: 20_000             # Forward simulation paths
  swr_tolerance: 0.05           # Max acceptable ruin probability
  swr_bracket: [0.01, 0.15]     # SWR search interval
```

---

## Output Figures

### `efficient_frontier.png`
Mean-variance efficient frontier with individual assets marked and portfolio composition bars.

### `wealth_pdf.png`
5-panel wealth probability density at key life-phase snapshots, with per-period ruin fraction annotated.

### `ruin_curve.png`
Lifetime ruin probability as a function of withdrawal rate, with the Safe Withdrawal Rate highlighted.

---

## Console Output (example)

```
============================================================
  Safe Withdrawal Rate:  4.12%  of initial wealth
  = $4,120 / year (real)
  Ruin probability at SWR: 5.00%
  Lifetime ruin (base case): 2.31%
============================================================
```

---

## Design Notes

### SDP Formulation

- **State**: `(W, t)` — 500 log-spaced wealth nodes × T annual periods
- **Action**: weight vector from efficient-frontier portfolios
- **Transition**: `W' = (W ± cf_t) × R_p`  where `R_p ~ LogNormal(μ_p, σ_p²)`
- **Utility**: CRRA `U(W) = W^(1-γ)/(1-γ)`, `U(0) = −∞`
- **Bellman**: `V(W,t) = max_a { β · E[V(W', t+1)] }` via MC expectation

### Ruin State
W = 0 is an absorbing state. Any path with W ≤ 0 stays ruined forever. Ruin probability is monotone non-decreasing.

### Safe Withdrawal Rate
Brent's method solves `P_ruin(r) = tolerance` over `r ∈ [r_lo, r_hi]`. Each evaluation re-runs forward simulation with the candidate withdrawal amount.

---

## Caching

All Alpha Vantage responses are stored as Parquet files under `outputs/cache/`. Subsequent runs reuse cached data. With the free tier (25 calls/day), a full first run uses ~4 calls (3 tickers + CPI).

---

## License

MIT
