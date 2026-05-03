"""Signal-function smoke + invariant tests.

These don't assert specific entry days (signal logic itself can evolve) — they
assert that:
  - each signal returns its expected output column with bool dtype
  - market_filter ANDs correctly (False days zero out signals)
  - signals degrade gracefully when optional inputs are missing
"""
import pandas as pd
import pytest

from technical.signals import (
    STRATEGIES,
    signal_longterm_quality_entry,
    signal_revenue_momentum,
    signal_short_vol_breakout,
    signal_swing_dual_inst,
    signal_swing_ma_kd_inst,
)


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s["name"])
def test_signal_function_produces_expected_column(strategy, synthetic_ohlcv,
                                                  synthetic_institutional,
                                                  synthetic_revenue):
    """Every registered strategy must populate its signal_col with a bool series."""
    extra = {}
    if strategy.get("needs_revenue"):
        extra["rev_df"] = synthetic_revenue
    out = strategy["signal_fn"](
        synthetic_ohlcv,
        inst_df=synthetic_institutional,
        **extra,
    )
    col = strategy["signal_col"]
    assert col in out.columns, f"{strategy['name']}: missing {col}"
    assert out[col].dtype == bool, f"{strategy['name']}: {col} must be bool"
    assert len(out) == len(synthetic_ohlcv)


def test_market_filter_ands_with_signal(synthetic_ohlcv, synthetic_institutional):
    """market_filter=False on every day must force every signal to False."""
    all_false = pd.Series(False, index=synthetic_ohlcv["date"])
    out = signal_longterm_quality_entry(
        synthetic_ohlcv,
        inst_df=synthetic_institutional,
        market_filter=all_false,
    )
    assert not out["signal_long"].any()


def test_market_filter_none_does_not_alter_signal(synthetic_ohlcv, synthetic_institutional):
    """market_filter=None must be a no-op (some signals can still fire)."""
    no_filter = signal_longterm_quality_entry(
        synthetic_ohlcv, inst_df=synthetic_institutional, market_filter=None,
    )
    # All True filter ⇒ same shape as no-filter
    all_true = pd.Series(True, index=synthetic_ohlcv["date"])
    with_filter = signal_longterm_quality_entry(
        synthetic_ohlcv, inst_df=synthetic_institutional, market_filter=all_true,
    )
    assert no_filter["signal_long"].equals(with_filter["signal_long"])


def test_revenue_momentum_returns_false_when_rev_df_missing(synthetic_ohlcv):
    """signal_revenue_momentum must not crash with rev_df=None — just no signals."""
    out = signal_revenue_momentum(synthetic_ohlcv, rev_df=None)
    assert "signal_rev" in out.columns
    assert not out["signal_rev"].any()


def test_dual_inst_returns_false_when_inst_df_missing(synthetic_ohlcv):
    """signal_swing_dual_inst requires inst_df — must return False, not crash."""
    out = signal_swing_dual_inst(synthetic_ohlcv, inst_df=None)
    assert "signal_dual_inst" in out.columns
    assert not out["signal_dual_inst"].any()


def test_short_vol_breakout_runs_without_inst_df(synthetic_ohlcv):
    """Short signal can run without institutional data (treats inst_buy as True)."""
    out = signal_short_vol_breakout(synthetic_ohlcv, inst_df=None)
    assert "signal_short" in out.columns


def test_swing_ma_kd_runs_without_inst_df(synthetic_ohlcv):
    out = signal_swing_ma_kd_inst(synthetic_ohlcv, inst_df=None)
    assert "signal_swing" in out.columns
