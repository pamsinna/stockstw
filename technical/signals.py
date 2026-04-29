"""
訊號產生器：三個時間框架的進場邏輯
每個函數接受附加指標的 DataFrame，回傳加上 signal 欄位的 DataFrame

market_filter: pd.Series indexed by date (bool) — True = 大盤多頭，可買進
               若為 None 則不過濾（全部允許）
"""
import pandas as pd
from technical.indicators import add_all, merge_institutional


def _apply_market_filter(df: pd.DataFrame,
                          signal_col: str,
                          market_filter: "pd.Series | None") -> pd.DataFrame:
    """AND signal with market_filter (date-indexed bool Series)."""
    if market_filter is None or market_filter.empty:
        return df
    ok = df["date"].map(market_filter).fillna(False)
    df[signal_col] = df[signal_col] & ok
    return df


# ─── 短線策略（1～5天）────────────────────────────────────────────────────────
# 核心邏輯：量暴增突破 + 陽線（收 > 開） + 外資投信同步買進（過濾假突破）

def signal_short_vol_breakout(df: pd.DataFrame,
                               inst_df: pd.DataFrame | None = None,
                               vol_ratio: float = 2.5,
                               breakout_days: int = 20,
                               market_filter: pd.Series | None = None) -> pd.DataFrame:
    df = add_all(df)
    if inst_df is not None and not inst_df.empty:
        df = merge_institutional(df, inst_df)

    cond_vol    = df["vol_ratio"] >= vol_ratio          # 量大於均量 2.5 倍
    cond_break  = df[f"new_high_{breakout_days}"]       # 突破近 20 日高點
    cond_candle = df["close"] > df["open"]              # 必須收陽線（過濾假突破）
    cond_ma     = df["close"] > df["ma20"]

    # 嚴格要求外資 + 投信同步買超（不接受僅法人合計為正）
    if "inst_sync_buy" in df.columns:
        inst_buy = df["inst_sync_buy"]
    else:
        inst_buy = pd.Series(True, index=df.index)

    df["signal_short"] = (cond_vol & cond_break & cond_candle & cond_ma & inst_buy)
    return _apply_market_filter(df, "signal_short", market_filter)


# ─── 波段策略（1～4週）────────────────────────────────────────────────────────
# 核心邏輯：中期均線多頭 + KD 回測金叉（允許 K 最高到 65）+ 法人連續買超

def signal_swing_ma_kd_inst(df: pd.DataFrame,
                              inst_df: pd.DataFrame | None = None,
                              kd_threshold: float = 65,
                              inst_days: int = 2,
                              market_filter: pd.Series | None = None) -> pd.DataFrame:
    df = add_all(df)
    if inst_df is not None and not inst_df.empty:
        df = merge_institutional(df, inst_df)

    # 放寬：只要 ma20 > ma60（中期趨勢向上），不強求 5/10MA 也對齊
    cond_ma    = (df["close"] > df["ma20"]) & (df["ma20"] > df["ma60"])
    cond_kd    = df["kd_golden"] & (df["kd_k"] < kd_threshold)
    cond_macd  = df["macd_hist_up"]  # 只要 MACD 柱狀圖上升即可

    # 法人當日合計淨買超即可（不要求連續 N 天，避免與 KD 金叉日期無法對齊）
    inst_cond = pd.Series(True, index=df.index)
    if "inst_total" in df.columns:
        inst_cond = df["inst_total"] > 0

    df["signal_swing"] = (cond_ma & cond_kd & cond_macd & inst_cond)
    return _apply_market_filter(df, "signal_swing", market_filter)


# ─── 中長線策略（1～3個月）───────────────────────────────────────────────────
# 核心邏輯：均線多頭 + MACD 金叉翻正 + 布林下軌反彈（逢低進場）

def signal_longterm_quality_entry(df: pd.DataFrame,
                                   inst_df: pd.DataFrame | None = None,
                                   market_filter: pd.Series | None = None) -> pd.DataFrame:
    df = add_all(df)
    if inst_df is not None and not inst_df.empty:
        df = merge_institutional(df, inst_df)

    # 月線以上（趨勢向上）
    cond_above_ma60 = df["close"] > df["ma60"]
    # MACD 翻正
    cond_macd = df["macd_golden"] & df["macd_positive"]
    # 布林帶位置適中（不追高，在中軸以上）
    cond_bb = (df["bb_pct"] > 0.3) & (df["bb_pct"] < 0.85)
    # RSI 未過熱
    cond_rsi = df["rsi"] < 70

    df["signal_long"] = (cond_above_ma60 & cond_macd & cond_bb & cond_rsi)
    return _apply_market_filter(df, "signal_long", market_filter)


# ─── 波段加強版：外資 + 投信雙買（最強訊號）─────────────────────────────────

def signal_swing_dual_inst(df: pd.DataFrame,
                             inst_df: pd.DataFrame | None = None,
                             foreign_days: int = 3,
                             trust_days: int = 2,
                             market_filter: pd.Series | None = None) -> pd.DataFrame:
    """外資連買 N 天 + 投信連買 M 天，是波段最強訊號之一"""
    df = add_all(df)
    if inst_df is None or inst_df.empty:
        df["signal_dual_inst"] = False
        return df

    df = merge_institutional(df, inst_df)

    # 計算外資/投信各自連續買超天數
    for col, new_col in [("foreign_", "foreign_consec"), ("trust", "trust_consec")]:
        if col in df.columns:
            buy = (df[col] > 0).astype(int)
            groups = (buy != buy.shift(1)).cumsum()
            df[new_col] = buy.groupby(groups).cumsum() * buy
        else:
            df[new_col] = 0

    cond_foreign = df["foreign_consec"] >= foreign_days
    cond_trust   = df["trust_consec"]   >= trust_days
    cond_ma      = df["close"] > df["ma20"]

    df["signal_dual_inst"] = (cond_foreign & cond_trust & cond_ma)
    return _apply_market_filter(df, "signal_dual_inst", market_filter)


# ─── 策略五：月營收動能 + 法人確認（基本面轉折因子）─────────────────────────

def signal_revenue_momentum(
    df: pd.DataFrame,
    inst_df: pd.DataFrame | None = None,
    rev_df: pd.DataFrame | None = None,
    market_filter: pd.Series | None = None,
) -> pd.DataFrame:
    """
    策略五：月營收 YoY 加速 + 外資確認

    訊號只在每月 10 日（營收公布日）後的「第一個」交易日觸發，
    以確保資料已公開、且每月只進場一次（避免同一份資料重複觸發）。

    進場條件（全部滿足）：
      基本面：YoY > 20%，YoY 加速 > 10pp，過去 12 個月 ≥ 8 個月正成長
      籌碼：近 20 日外資累計買超 > 0
      技術：收盤 > MA60 或近 20 日跌幅 < -10%（不在崩跌中）
    """
    df = add_all(df).copy()
    df["signal_rev"] = False

    if rev_df is None or rev_df.empty:
        return df

    rev = rev_df.sort_values("date").copy()

    # ── 計算 YoY ───────────────────────────────────────────────────────────────
    if "revenue_yoy" in rev.columns and not rev["revenue_yoy"].isna().all():
        rev["yoy"] = pd.to_numeric(rev["revenue_yoy"], errors="coerce")
    else:
        rev["yoy"] = rev["revenue"].pct_change(12) * 100

    # 3 個月平均 YoY（用前 3 個月，不含當月，避免前視偏差）
    rev["yoy_3m_avg"] = rev["yoy"].shift(1).rolling(3, min_periods=2).mean()
    rev["yoy_accel"]  = rev["yoy"] - rev["yoy_3m_avg"]

    # 過去 12 個月正成長月數
    rev["consistency"] = (rev["yoy"] > 0).astype(int).rolling(12, min_periods=6).sum()

    # ── 公布日（月末 +1 個月取第 10 日）──────────────────────────────────────
    def _pub(d: pd.Timestamp) -> pd.Timestamp:
        y, m = d.year, d.month
        if m == 12:
            return pd.Timestamp(y + 1, 1, 10)
        return pd.Timestamp(y, m + 1, 10)

    rev["publish_date"] = rev["date"].apply(_pub)

    # ── 營收條件通過的公布日集合 ─────────────────────────────────────────────
    rev_ok = rev[
        (rev["yoy"] > 20) &
        (rev["yoy_accel"] > 10) &
        (rev["consistency"] >= 8)
    ]["publish_date"].tolist()

    if not rev_ok:
        return _apply_market_filter(df, "signal_rev", market_filter)

    # ── 法人：近 20 日外資累計買超 ───────────────────────────────────────────
    if inst_df is not None and not inst_df.empty:
        df = merge_institutional(df, inst_df)
        if "foreign_" in df.columns:
            df["_foreign_20d"] = df["foreign_"].rolling(20, min_periods=5).sum()
        else:
            df["_foreign_20d"] = 0.0
    else:
        df["_foreign_20d"] = 0.0

    # ── 技術面：不在崩跌中 ───────────────────────────────────────────────────
    df["_ret_20d"] = df["close"] / df["close"].shift(20) - 1
    cond_tech = (df["close"] > df["ma60"]) | (df["_ret_20d"] > -0.10)

    # ── 對應到第一個交易日 ────────────────────────────────────────────────────
    price_dates = df["date"].sort_values().values  # numpy datetime64 array

    for pub in rev_ok:
        pub_ts = pd.Timestamp(pub)
        # 找公布日後第一個有價格資料的交易日
        candidates = df[df["date"] >= pub_ts]
        if candidates.empty:
            continue
        signal_day = candidates.iloc[0]["date"]
        day_mask = df["date"] == signal_day

        tech_ok  = cond_tech[day_mask].any()
        inst_ok  = (df.loc[day_mask, "_foreign_20d"] > 0).any() or (df["_foreign_20d"] == 0).all()
        if tech_ok and inst_ok:
            df.loc[day_mask, "signal_rev"] = True

    df.drop(columns=["_foreign_20d", "_ret_20d"], errors="ignore", inplace=True)
    return _apply_market_filter(df, "signal_rev", market_filter)


# ─── 全策略清單（供批次回測用）────────────────────────────────────────────────

STRATEGIES = [
    {
        "name": "短線_量暴增突破",
        "signal_fn": signal_short_vol_breakout,
        "signal_col": "signal_short",
        "timeframe": "short",
        "default_tp": 0.08, "default_sl": 0.05, "default_hold": 5,
    },
    {
        "name": "波段_均線KD法人",
        "signal_fn": signal_swing_ma_kd_inst,
        "signal_col": "signal_swing",
        "timeframe": "swing",
        "default_tp": 0.15, "default_sl": 0.07, "default_hold": 20,
    },
    {
        "name": "波段_外資投信雙買",
        "signal_fn": signal_swing_dual_inst,
        "signal_col": "signal_dual_inst",
        "timeframe": "swing",
        "default_tp": 0.12, "default_sl": 0.06, "default_hold": 20,
    },
    {
        "name": "中長線_品質股低接",
        "signal_fn": signal_longterm_quality_entry,
        "signal_col": "signal_long",
        "timeframe": "long",
        "default_tp": 0.30, "default_sl": 0.10, "default_hold": 90,
    },
    {
        "name": "月營收動能",
        "signal_fn": signal_revenue_momentum,
        "signal_col": "signal_rev",
        "timeframe": "revenue",
        "default_tp": 0.40, "default_sl": 0.12, "default_hold": 120,
        "needs_revenue": True,  # 告知回測/選股迴圈需要額外載入月營收資料
    },
]
