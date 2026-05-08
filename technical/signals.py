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

def _merge_per(df: pd.DataFrame, per_df: pd.DataFrame) -> pd.DataFrame:
    """把每日 PER 合併進價格 DataFrame（left join on date）。"""
    per = per_df[["date", "per"]].rename(columns={"per": "_per"})
    return df.merge(per, on="date", how="left")


def signal_longterm_quality_entry(df: pd.DataFrame,
                                   inst_df: pd.DataFrame | None = None,
                                   per_df: pd.DataFrame | None = None,
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
    # 法人 60 日累計淨買超：外資 + 投信合計 > 0（OR 改為加總，避免一買一大賣仍過關）
    if "foreign_" in df.columns or "trust" in df.columns:
        f_60d = (df["foreign_"].rolling(60, min_periods=30).sum()
                 if "foreign_" in df.columns else pd.Series(0.0, index=df.index))
        t_60d = (df["trust"].rolling(60, min_periods=30).sum()
                 if "trust" in df.columns else pd.Series(0.0, index=df.index))
        cond_inst_accum = (f_60d + t_60d) > 0
    else:
        cond_inst_accum = pd.Series(True, index=df.index)

    # PER 過濾：0 < PER < 20（有獲利且不過貴）；無資料時放行
    if per_df is not None and not per_df.empty:
        df = _merge_per(df, per_df)
        cond_per = (df["_per"] > 0) & (df["_per"] < 20) | df["_per"].isna()
    else:
        cond_per = pd.Series(True, index=df.index)

    df["signal_long"] = (cond_above_ma60 & cond_macd & cond_bb & cond_rsi & cond_inst_accum & cond_per)
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
    per_df: pd.DataFrame | None = None,
    market_filter: pd.Series | None = None,
) -> pd.DataFrame:
    """
    策略五：月營收 YoY 加速 + 外資確認（基本面轉折因子）

    訊號只在每月 11 日（公布日保守估計）後的「第一個」交易日觸發。
    11 日而非 10 日：部分公司在 10 日傍晚才公布，11 日確保全市場資料可用。

    進場條件（全部滿足）：
      基本面：
        1. YoY > 20%
        2. 加速 > 10pp（vs 前 3 個月平均）
        3. 過去 12 個月 ≥ 8 個月正成長
        4. 2 年 CAGR > 5%（消除基期效應，排除「去年太爛所以今年高 YoY」）
      籌碼：
        5. 近 20 日外資累計買超 > 近 20 日均量的 0.5%（相對強度，避免大型股雜訊）
      技術：
        6. 站上 MA60，或（近 5 日跌幅 > -8% 且當日收紅K）
           排除正在崩跌中的股票；當日紅K確認有買盤進來，不是盲目逢低接
    """
    df = add_all(df).copy()
    df["signal_rev"] = False

    if rev_df is None or rev_df.empty:
        return df

    rev = rev_df.sort_values("date").copy()
    rev["revenue"] = pd.to_numeric(rev["revenue"], errors="coerce")

    # ── 計算 YoY（百分比，如 20 = 20%）────────────────────────────────────────
    if "revenue_yoy" in rev.columns and not rev["revenue_yoy"].isna().all():
        rev["yoy"] = pd.to_numeric(rev["revenue_yoy"], errors="coerce")
    else:
        rev["yoy"] = rev["revenue"].pct_change(12) * 100

    # 3 個月平均 YoY（不含當月，避免前視偏差）
    rev["yoy_3m_avg"] = rev["yoy"].shift(1).rolling(3, min_periods=2).mean()
    rev["yoy_accel"]  = rev["yoy"] - rev["yoy_3m_avg"]

    # 過去 12 個月正成長月數
    rev["consistency"] = (rev["yoy"] > 0).astype(int).rolling(12, min_periods=6).sum()

    # 2 年 CAGR：消除基期效應（NaN 時不過濾，讓基本面其他條件決定）
    rev["rev_lag24"] = rev["revenue"].shift(24)
    rev["cagr_2y"] = (rev["revenue"] / rev["rev_lag24"]) ** 0.5 - 1

    # ── 公布日：有 fetched_date 用實際抓取日，否則退回次月 10 日（保守估計）──
    # fetched_date 由 save_monthly_revenue 在首次抓到時寫入，代表資料真實可用日
    def _pub(d: pd.Timestamp, fetched=None) -> pd.Timestamp:
        if fetched is not None and not pd.isna(fetched):
            return pd.Timestamp(fetched)
        y, m = d.year, d.month
        if m == 12:
            return pd.Timestamp(y + 1, 1, 10)
        return pd.Timestamp(y, m + 1, 10)

    has_fetched = "fetched_date" in rev.columns
    rev["publish_date"] = [
        _pub(row["date"], row["fetched_date"] if has_fetched else None)
        for _, row in rev.iterrows()
    ]

    # ── 所有四條基本面條件通過的公布日 ───────────────────────────────────────
    cagr_ok = rev["cagr_2y"].isna() | (rev["cagr_2y"] > 0.05)
    rev_ok = rev[
        (rev["yoy"] > 20) &
        (rev["yoy_accel"] > 10) &
        (rev["consistency"] >= 8) &
        cagr_ok
    ]["publish_date"].tolist()

    if not rev_ok:
        return _apply_market_filter(df, "signal_rev", market_filter)

    # ── 法人：外資 20 日累計買超 > 均量 0.5%（相對強度）────────────────────
    has_inst = inst_df is not None and not inst_df.empty
    if has_inst:
        df = merge_institutional(df, inst_df)
    if has_inst and "foreign_" in df.columns:
        df["_foreign_20d"] = df["foreign_"].rolling(20, min_periods=5).sum()
        # volume 單位是股，foreign_ 是張（1 張 = 1000 股），統一為張
        df["_vol_20d_avg"] = df["volume"].rolling(20, min_periods=5).mean() / 1000
        df["_inst_threshold"] = df["_vol_20d_avg"] * 0.005  # 均量 0.5%
    else:
        df["_foreign_20d"]    = 0.0
        df["_inst_threshold"] = -1.0  # 無資料時放行（-1 確保 > 永遠成立）

    # ── 技術面：站上 MA60，或近 5 日不崩跌且已止跌 ──────────────────────────
    df["_ret_5d"]   = df["close"] / df["close"].shift(5) - 1
    cond_above_ma60 = df["close"] > df["ma60"]
    cond_not_crash  = df["_ret_5d"] > -0.08
    cond_red_candle = df["close"] > df["open"]
    cond_tech = cond_above_ma60 | (cond_not_crash & cond_red_candle)

    # ── PER 過濾：0 < PER < 20；無資料時放行 ────────────────────────────────
    if per_df is not None and not per_df.empty:
        df = _merge_per(df, per_df)
    else:
        df["_per"] = float("nan")

    # ── 對應到 publish_date 後第一個交易日 ──────────────────────────────────
    for pub in rev_ok:
        pub_ts = pd.Timestamp(pub)
        candidates = df[df["date"] >= pub_ts]
        if candidates.empty:
            continue
        signal_day = candidates.iloc[0]["date"]
        day_mask = df["date"] == signal_day

        tech_ok = bool(cond_tech[day_mask].any())
        inst_ok = bool(
            (df.loc[day_mask, "_foreign_20d"] >=
             df.loc[day_mask, "_inst_threshold"]).any()
        )
        per_val = df.loc[day_mask, "_per"].iloc[0] if day_mask.any() else float("nan")
        per_ok  = pd.isna(per_val) or (0 < per_val < 20)

        if tech_ok and inst_ok and per_ok:
            df.loc[day_mask, "signal_rev"] = True

    df.drop(columns=["_foreign_20d", "_inst_threshold", "_vol_20d_avg",
                      "_ret_5d", "_per"], errors="ignore", inplace=True)
    return _apply_market_filter(df, "signal_rev", market_filter)


# ─── 策略六：短線反轉收貨 ────────────────────────────────────────────────────

def signal_reversal_inst(
    df: pd.DataFrame,
    inst_df: pd.DataFrame | None = None,
    market_filter: pd.Series | None = None,
) -> pd.DataFrame:
    """
    策略六：短線反轉收貨（目標持倉 ≤ 20 個交易日）

    三階段底部確認：
      階段一 - 跌破 MA20 支撐（近 10 日內曾在 MA20 上方，目前在 MA20 下方）
      階段二 - 量縮確認賣盤耗盡（近 2 日均量 < MA20量 × 0.7）
      階段三 - 第一根紅K翻轉（收 > 開 AND 收 > 昨收，量開始回升）
    法人：長線收貨（60日） + 短線丟出（10日）製造底部，OR 邏輯
    出場（engine）：停利 +15%、停損 -7%、最長 20日、連跌兩日次日出
    """
    df = add_all(df).copy()
    df["signal_reversal"] = False

    vol_ma20 = df["volume"].rolling(20, min_periods=10).mean()

    # ── 流動性 ────────────────────────────────────────────────────────
    cond_liquid = (vol_ma20 > 500) & (df["close"] > 15)

    # ── 階段一：跌破 MA20 支撐 ───────────────────────────────────────
    # 目前在 MA20 下方，且近 10 日最高曾在 MA20 上方（確認是跌破，不是長期空頭）
    below_ma20_now  = df["close"].shift(1) < df["ma20"].shift(1)
    was_above_ma20  = df["close"].shift(1).rolling(10, min_periods=5).max() > df["ma20"].shift(1)
    cond_broke_support = below_ma20_now & was_above_ma20

    # ── 階段二：量縮（賣盤力道耗盡）────────────────────────────────
    # 近 2 日成交量均低於 MA20量 × 0.7
    vol_shrink = (
        (df["volume"].shift(1) < vol_ma20 * 0.7) &
        (df["volume"].shift(2) < vol_ma20 * 0.7)
    )

    # ── 階段三：今日翻轉紅K ──────────────────────────────────────────
    cond_bullish = df["close"] > df["open"]           # 收紅（台股 = 收漲）
    cond_higher  = df["close"] > df["close"].shift(1) # 收高於昨收
    # 量開始回升：今日量 > 昨日量（量縮後有人接手），無需大量
    cond_vol_recover = df["volume"] > df["volume"].shift(1)

    # ── 法人：長線收貨 + 短線丟出（OR 邏輯）────────────────────────
    if inst_df is not None and not inst_df.empty:
        df = merge_institutional(df, inst_df)

    if "foreign_" in df.columns and "trust" in df.columns:
        f_60d = df["foreign_"].rolling(60, min_periods=30).sum()
        t_60d = df["trust"].rolling(60, min_periods=30).sum()
        f_10d = df["foreign_"].rolling(10, min_periods=5).sum()
        t_10d = df["trust"].rolling(10, min_periods=5).sum()
        inst_long  = (f_60d > 0) | (t_60d > 0)
        inst_short = (f_10d < 0) | (t_10d < 0)
    else:
        inst_long  = pd.Series(True, index=df.index)
        inst_short = pd.Series(True, index=df.index)

    df["signal_reversal"] = (
        cond_liquid &
        cond_broke_support &
        vol_shrink &
        cond_bullish &
        cond_higher &
        cond_vol_recover &
        inst_long &
        inst_short
    )

    return _apply_market_filter(df, "signal_reversal", market_filter)


# ─── 全策略清單（供批次回測用）────────────────────────────────────────────────

STRATEGIES = [
    {
        "name": "中長線_品質股低接",
        "signal_fn": signal_longterm_quality_entry,
        "signal_col": "signal_long",
        "timeframe": "long",
        "default_tp": 0.30, "default_sl": 0.10, "default_hold": 90,
        "strict_market": True,
        "needs_per": True,
        "needs_fundamental": True,
    },
    {
        "name": "月營收動能",
        "signal_fn": signal_revenue_momentum,
        "signal_col": "signal_rev",
        "timeframe": "revenue",
        "default_tp": 0.40, "default_sl": 0.12, "default_hold": 120,
        "needs_revenue": True,
        "needs_per": True,
        "needs_fundamental": True,
    },
]
