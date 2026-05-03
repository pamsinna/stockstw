"""Shared fixtures: synthetic OHLCV / institutional / revenue dataframes.

Pure deterministic data — no network, no SQLite, no time.sleep. Tests built on
these fixtures should run in well under 1s each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """120 trading days of an uptrending price series with realistic OHLCV.

    Long enough for MA60 / 20-day breakout / KD / MACD / RSI to all populate.
    """
    n = 120
    rng = np.random.default_rng(seed=42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    drift = np.linspace(100, 150, n)
    noise = rng.normal(0, 1.2, n)
    close = drift + noise
    open_ = close - rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.0, n)
    volume = rng.integers(800_000, 1_500_000, n).astype(float)
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def synthetic_institutional(synthetic_ohlcv) -> pd.DataFrame:
    """Institutional buy/sell aligned with the synthetic price dates."""
    n = len(synthetic_ohlcv)
    rng = np.random.default_rng(seed=7)
    return pd.DataFrame({
        "date": synthetic_ohlcv["date"],
        "foreign": rng.integers(-500, 1500, n).astype(float),
        "trust": rng.integers(-200, 800, n).astype(float),
        "dealer": rng.integers(-100, 400, n).astype(float),
    })


@pytest.fixture
def synthetic_revenue() -> pd.DataFrame:
    """36 months of monthly revenue with revenue_yoy already provided."""
    months = pd.date_range("2022-01-01", periods=36, freq="MS")
    base = np.linspace(100_000, 200_000, 36)
    return pd.DataFrame({
        "date": months,
        "revenue": base,
        "revenue_yoy": np.linspace(5, 30, 36),
    })


@pytest.fixture
def handcrafted_price_with_signal() -> pd.DataFrame:
    """Tiny OHLCV dataframe with explicit signal — for engine unit tests.

    Signal day = day 2. Engine should enter at day 3 open (with slippage).
    """
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-02", "2024-01-03", "2024-01-04",
            "2024-01-05", "2024-01-08", "2024-01-09",
            "2024-01-10", "2024-01-11", "2024-01-12",
            "2024-01-15", "2024-01-16", "2024-01-17",
        ]),
        "open":   [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        "high":   [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        "low":    [ 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "close":  [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        "signal": [False, False, True, False, False, False,
                   False, False, False, False, False, False],
    })
