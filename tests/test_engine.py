"""Engine invariants — these are the rules CLAUDE.md calls out as load-bearing.

If any of these break, backtest results stop being meaningful. The most
expensive bugs in this category are silent (look-ahead bias, fee math drift) —
that's why these tests focus on numeric assertions, not just "it runs".
"""
import pandas as pd
import pytest

from backtest.engine import Trade, run_backtest
from config import (
    FEE_RATE_BUY, FEE_RATE_SELL,
    SLIPPAGE,
    TAX_EMERGING, TAX_TWSE_OTC,
)


def test_entry_fills_at_next_day_open_with_slippage(handcrafted_price_with_signal):
    """No look-ahead: signal at T → entry at T+1 open * (1+SLIPPAGE)."""
    df = handcrafted_price_with_signal
    result = run_backtest(
        df, signal_col="signal",
        take_profit=10.0, stop_loss=10.0, max_hold_days=999,
        start="2024-01-01", end="2024-12-31",
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    # signal day = 2024-01-04 (index 2). Entry should be next bar's open.
    assert trade.entry_date == pd.Timestamp("2024-01-05")
    assert trade.entry_price == pytest.approx(103 * (1 + SLIPPAGE))


def test_take_profit_exit_uses_tp_price_not_high(handcrafted_price_with_signal):
    df = handcrafted_price_with_signal.copy()
    # Spike day 5 high to clearly trigger 5% TP from entry.
    df.loc[5, "high"] = 200.0
    result = run_backtest(
        df, signal_col="signal",
        take_profit=0.05, stop_loss=0.99, max_hold_days=999,
        start="2024-01-01", end="2024-12-31",
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "take_profit"
    assert trade.exit_price == pytest.approx(trade.entry_price * 1.05)


def test_stop_loss_exit_uses_sl_price(handcrafted_price_with_signal):
    df = handcrafted_price_with_signal.copy()
    # Drop day 5 low below 5% stop.
    df.loc[5, "low"] = 50.0
    result = run_backtest(
        df, signal_col="signal",
        take_profit=0.99, stop_loss=0.05, max_hold_days=999,
        start="2024-01-01", end="2024-12-31",
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(trade.entry_price * 0.95)


def test_max_hold_exits_at_next_open(handcrafted_price_with_signal):
    df = handcrafted_price_with_signal
    # max_hold_days = 1: should exit one bar after entry (at next open).
    result = run_backtest(
        df, signal_col="signal",
        take_profit=0.99, stop_loss=0.99, max_hold_days=1,
        start="2024-01-01", end="2024-12-31",
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "max_hold"
    # entry at 2024-01-05, max_hold check fires on 2024-01-08 (≥1 day),
    # exits at NEXT bar's open (2024-01-09 open = 105).
    assert trade.exit_price == pytest.approx(105 * (1 + SLIPPAGE))


def test_pnl_pct_subtracts_buy_sell_and_tax_for_twse():
    trade = Trade(
        stock_id="2330",
        entry_date=pd.Timestamp("2024-01-01"),
        entry_price=100.0,
        exit_date=pd.Timestamp("2024-01-10"),
        exit_price=110.0,
        market="TWSE",
    )
    expected = 0.10 - (FEE_RATE_BUY + FEE_RATE_SELL + TAX_TWSE_OTC)
    assert trade.pnl_pct == pytest.approx(expected)


def test_pnl_pct_uses_emerging_tax_rate():
    """興櫃稅率不同 — engine must look up TAX_EMERGING when market='Emerging'."""
    trade = Trade(
        stock_id="9999",
        entry_date=pd.Timestamp("2024-01-01"),
        entry_price=100.0,
        exit_date=pd.Timestamp("2024-01-10"),
        exit_price=110.0,
        market="Emerging",
    )
    expected = 0.10 - (FEE_RATE_BUY + FEE_RATE_SELL + TAX_EMERGING)
    assert trade.pnl_pct == pytest.approx(expected)
    assert TAX_EMERGING != TAX_TWSE_OTC, "tax rates must differ for this test to be meaningful"


def test_no_signal_produces_no_trades():
    df = pd.DataFrame({
        "date":   pd.bdate_range("2024-01-01", periods=20),
        "open":   [100.0] * 20,
        "high":   [101.0] * 20,
        "low":    [ 99.0] * 20,
        "close":  [100.0] * 20,
        "signal": [False] * 20,
    })
    result = run_backtest(
        df, signal_col="signal",
        take_profit=0.05, stop_loss=0.05, max_hold_days=10,
        start="2024-01-01", end="2024-12-31",
    )
    assert result.trades == []


def test_open_trade_force_closed_at_end_of_period(handcrafted_price_with_signal):
    """Trades still open at end of period must be force-closed at last close."""
    df = handcrafted_price_with_signal
    result = run_backtest(
        df, signal_col="signal",
        take_profit=0.99, stop_loss=0.99, max_hold_days=999,
        start="2024-01-01", end="2024-12-31",
    )
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "end_of_period"
    assert result.trades[0].exit_price == pytest.approx(111.0)
