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
]
