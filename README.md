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

## Models

**Gaussian HMM** — Learns hidden states from the sequence of feature vectors. Provides posterior probabilities for each regime.

**KMeans** — Baseline clustering approach. Groups days by feature similarity. No temporal structure.

Both models standardise features with `StandardScaler` before fitting. A fixed `SEED = 42` ensures reproducibility.

