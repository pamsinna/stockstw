"""Regression tests for issue #3 — _normalize_and_save_revenue must handle
all documented FinMind column variants and log a warning on the early-return
path so future schema drift surfaces.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from backtest import run_backtest


@pytest.fixture
def captured_save():
    """Patch save_monthly_revenue so we capture what would have been written."""
    with patch.object(run_backtest, "save_monthly_revenue") as mock_save:
        yield mock_save


def _raw_revenue(column_name: str) -> pd.DataFrame:
    """Synthetic FinMind-shaped response with a configurable revenue column."""
    months = pd.date_range("2022-01-01", periods=24, freq="MS")
    return pd.DataFrame({
        "date": months,
        column_name: range(100_000, 100_000 + 24 * 1000, 1000),
    })


@pytest.mark.parametrize("column_name", [
    "revenue",
    "Revenue",
    "revenue_month",
    "monthly_revenue",
])
def test_known_revenue_aliases_are_normalized_and_saved(column_name, captured_save):
    raw = _raw_revenue(column_name)
    run_backtest._normalize_and_save_revenue("2330", raw)

    captured_save.assert_called_once()
    saved_id, saved_df = captured_save.call_args[0]
    assert saved_id == "2330"
    assert "revenue" in saved_df.columns, (
        f"alias {column_name!r} was not normalized to 'revenue'"
    )
    assert len(saved_df) == 24


def test_unknown_revenue_column_logs_warning_and_skips_save(captured_save, caplog):
    """Issue #3 root cause: unknown column names must NOT silently drop data."""
    raw = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=12, freq="MS"),
        "rev_unknown_field": range(12),
    })
    with caplog.at_level("WARNING", logger=run_backtest.logger.name):
        run_backtest._normalize_and_save_revenue("2330", raw)

    captured_save.assert_not_called()
    assert any("revenue column missing" in rec.message for rec in caplog.records), \
        "expected a warning when revenue column is missing"


def test_revenue_yoy_computed_when_missing(captured_save):
    raw = _raw_revenue("revenue")
    run_backtest._normalize_and_save_revenue("2330", raw)

    saved_df = captured_save.call_args[0][1]
    assert "revenue_yoy" in saved_df.columns


def test_existing_revenue_yoy_preserved(captured_save):
    raw = _raw_revenue("revenue")
    raw["revenue_yoy"] = 12.5
    run_backtest._normalize_and_save_revenue("2330", raw)

    saved_df = captured_save.call_args[0][1]
    assert (saved_df["revenue_yoy"] == 12.5).all()
