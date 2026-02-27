# Stock Market Regime Detection MVP

Detect market regimes (e.g. bull / bear) using a Gaussian Hidden Markov Model and a KMeans baseline, with an interactive Streamlit dashboard.

## Architecture

```
project/
├── src/                  # Core library
│   ├── config.py         # Constants, paths, seeds
│   ├── utils.py          # File I/O helpers
│   ├── data_ingestion.py # yfinance download + persistence
│   ├── features.py       # Feature engineering pipeline
│   ├── models.py         # HMM and KMeans training
│   └── backtest.py       # Simple regime-aware backtest
├── app/
│   └── streamlit_app.py  # Interactive dashboard
├── scripts/              # CLI entry points
│   ├── pull_data.py
│   ├── make_features.py
│   └── train_models.py
├── tests/                # pytest suite
│   ├── data/             # Sample CSV for tests
│   ├── test_features.py
│   ├── test_io.py
│   └── test_models.py
├── data/                 # Generated at runtime (git-ignored)
│   ├── raw/              # OHLCV parquet files
│   ├── processed/        # Feature tables
│   └── outputs/          # Regime predictions
├── requirements.txt
└── .gitignore
```

## Quick Start

```bash
# 1. Clone and set up
git clone <your-repo-url>
cd StockMarket-RegimeDetection-main

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull data
python scripts/pull_data.py SPY QQQ AAPL

# 5. Build features
python scripts/make_features.py SPY

# 6. Train models
python scripts/train_models.py SPY

# 7. Launch dashboard
streamlit run app/streamlit_app.py

# 8. Run tests
pytest
```

## Features Computed

| Feature | Description |
|---|---|
| `log_return` | `ln(Adj Close).diff()` |
| `rolling_vol_20` | 20-day std of log return, annualised (×√252) |
| `volume_change` | Percent change of trading volume |
| `ema_fast` | Exponential moving average, span=12 |
| `ema_slow` | Exponential moving average, span=26 |

## Models

**Gaussian HMM** — Learns hidden states from the sequence of feature vectors. Provides posterior probabilities for each regime.

**KMeans** — Baseline clustering approach. Groups days by feature similarity. No temporal structure.

Both models standardise features with `StandardScaler` before fitting. A fixed `SEED = 42` ensures reproducibility.

