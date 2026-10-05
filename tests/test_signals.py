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
    signal_growth_breakout,
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


# ─── 營收公布日：DB date 已是「申報月」（3 月營收 → 04-01）─────────────────────
# 回歸測試：舊版又 +1 個月，S5/S6 營收訊號整整晚一個月（2026-10 修正）。

def _flat_ohlcv(start="2023-01-02", end="2024-06-28"):
    dates = pd.bdate_range(start, end)
    n = len(dates)
    close = pd.Series(range(n), dtype=float) * 0.1 + 100
    return pd.DataFrame({"date": dates, "open": close - 0.5, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1_000_000.0})


def _rev_jump_at(label: str):
    """2022-01 ~ 2024-06 月營收；只有申報月 label 那一筆跳升 3 倍（單月爆量）。"""
    labels = pd.date_range("2022-01-01", "2024-06-01", freq="MS")
    rev = [100.0 * (1.01 ** i) for i in range(len(labels))]
    df = pd.DataFrame({"date": labels, "revenue": rev})
    df.loc[df["date"] == label, "revenue"] *= 3
    df["revenue_yoy"] = df["revenue"].pct_change(12) * 100
    return df


def test_growth_breakout_uses_report_month_publish_date():
    price = _flat_ohlcv()
    rev = _rev_jump_at("2024-04-01")  # 3 月營收，法定 4/10 前公布
    df = signal_growth_breakout(price, None, rev)
    g = df.set_index("date")["rev_3m_growth"]
    assert g.loc["2024-04-10"] < 50          # 公布日前還看不到
    assert g.loc["2024-04-11"] > 50          # 申報月 11 日起可用（不是 5/11）


def test_revenue_momentum_signal_day_in_report_month():
    price = _flat_ohlcv()
    rev = _rev_jump_at("2024-04-01")
    df = signal_revenue_momentum(price, None, rev)
    days = df.loc[df["signal_rev"], "date"]
    assert pd.Timestamp("2024-04-10") in set(days)
    assert not any((days >= "2024-05-01") & (days <= "2024-05-31"))


def test_revenue_burst_triggers_on_first_breakout_after_publish():
    from technical.signals import signal_revenue_burst
    dates = pd.bdate_range("2023-01-02", "2024-06-28")
    close = pd.Series(100.0, index=range(len(dates)))
    i0 = dates.get_indexer([pd.Timestamp("2024-04-15")])[0]
    close.iloc[i0:] = 130.0                       # 4/15 放量突破
    vol = pd.Series(1_000_000.0, index=close.index)
    vol.iloc[i0] = 3_000_000.0
    price = pd.DataFrame({"date": dates, "open": close, "high": close + 1, "low": close - 1,
                          "close": close, "volume": vol})
    labels = pd.date_range("2022-01-01", "2024-06-01", freq="MS")
    rev = pd.DataFrame({"date": labels, "revenue": 100.0})
    rev.loc[rev["date"] >= "2024-02-01", "revenue"] = 300.0   # 申報月 02～04 營收三倍 → 04-01 那筆 3M 合計創高
    out = signal_revenue_burst(price, rev)
    hits = out.loc[out["signal_burst"], "date"].tolist()
    assert pd.Timestamp("2024-04-15") in hits
    assert all(d >= pd.Timestamp("2024-02-10") for d in hits)          # 不早於公布日
    assert out.loc[out["date"] == "2024-04-15", "burst_yoy"].iloc[0] > 50


# ─── 法人佈局雷達（條件凍結 R1）────────────────────────────────────────────────

def _radar_inputs(trust_last=(1, 1, 1, 1), last_move=1.0, retail=(30.0, 29.8), yoy=30.0):
    import numpy as np
    dates = pd.bdate_range("2025-06-02", periods=300)
    # 先漲到 120、回檔到 100，最後 60 天緩升到 108（距 52 週高 10%、站上季線、近 20 日小漲）
    path = np.r_[np.linspace(80, 120, 200), np.linspace(120, 100, 40), np.linspace(100, 108, 60)]
    path[-1] = path[-2] * last_move
    price = pd.DataFrame({"date": dates, "open": path, "high": path, "low": path, "close": path,
                          "volume": 1e6})
    trust = np.zeros(300)
    trust[-len(trust_last):] = trust_last
    trust[-len(trust_last) - 1] = -1                        # 連買起點前一天是賣
    inst = pd.DataFrame({"date": dates, "foreign_": 0.0, "trust": np.array(trust) * 1e5, "dealer": 0.0})
    labels = pd.date_range("2024-06-01", "2026-07-01", freq="MS")
    rev = pd.DataFrame({"date": labels, "revenue": 100.0, "revenue_yoy": yoy})
    bench = pd.Series(100.0, index=dates)
    sh = pd.DataFrame({"date": [dates[-12], dates[-6]], "retail_pct": list(retail),
                       "total_shares": [1e8, 1e8]})
    return price, inst, rev, bench, sh


def test_layout_radar_hits_when_all_conditions_hold():
    from technical.signals import layout_radar_today
    r = layout_radar_today(*_radar_inputs())
    assert r is not None and r["trust_days"] == 4
    assert 5 <= r["dist52_pct"] <= 20 and r["since_start_pct"] < 8 and r["retail_wchg"] < 0


@pytest.mark.parametrize("kw", [
    {"trust_last": (1, 1)},          # 投信只連買 2 天
    {"last_move": 1.12},             # 投信進場後已漲 12%
    {"retail": (30.0, 30.5)},        # 散戶比例上升（在接）
    {"yoy": 5.0},                    # 營收年增不夠
])
def test_layout_radar_rejects(kw):
    from technical.signals import layout_radar_today
    assert layout_radar_today(*_radar_inputs(**kw)) is None


def test_layout_radar_ignores_unpublished_revenue():
    from technical.signals import layout_radar_today
    price, inst, rev, bench, sh = _radar_inputs(yoy=5.0)
    # 最新一筆（申報月在價格最後一天之後）年增很高，但還沒公布 → 不能用
    rev = pd.concat([rev, pd.DataFrame({"date": [price["date"].iloc[-1] + pd.offsets.MonthBegin(1)],
                                        "revenue": [300.0], "revenue_yoy": [200.0]})])
    assert layout_radar_today(price, inst, rev, bench, sh) is None


def test_foreign_layout_candidate_uses_foreign_flow_not_trust():
    from technical.signals import layout_radar_foreign_candidate, foreign_flow_ratio
    price, inst, rev, bench, sh = _radar_inputs(trust_last=(0, 0, 0, 0))   # 投信完全沒買
    inst["foreign_"] = 50_000.0                                             # 外資天天買
    r = layout_radar_foreign_candidate(price, inst, rev, bench, sh)
    assert r is not None and r["foreign_ratio"] == foreign_flow_ratio(price, inst) > 0
    inst["foreign_"] = -50_000.0                                            # 外資賣 → 不是候選
    assert layout_radar_foreign_candidate(price, inst, rev, bench, sh) is None
