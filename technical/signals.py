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
    """把每日 PER 合併進價格 DataFrame（left join on date）。
    ffill 補齊最新價格日期沒有對應 PER 的情況（PER 更新頻率低於日 K）。"""
    per = per_df[["date", "per"]].rename(columns={"per": "_per"})
    merged = df.merge(per, on="date", how="left")
    merged["_per"] = merged["_per"].ffill()
    return merged


def signal_longterm_quality_entry(df: pd.DataFrame,
                                   inst_df: pd.DataFrame | None = None,
                                   per_df: pd.DataFrame | None = None,
                                   market_filter: pd.Series | None = None,
                                   inst_threshold: int = 0) -> pd.DataFrame:
    df = add_all(df)
    if inst_df is not None and not inst_df.empty:
        df = merge_institutional(df, inst_df)

    # 月線以上（趨勢向上）
    cond_above_ma60 = df["close"] > df["ma60"]
    # MACD 金叉（3 日視窗：允許金叉後幾天才突破月線；不要求 MACD > 0）
    cond_macd = df["macd_golden"].rolling(3, min_periods=1).max().astype(bool)
    # 布林帶位置適中（不追高，在中軸以上）
    cond_bb = (df["bb_pct"] > 0.3) & (df["bb_pct"] < 1.2)
    # RSI 未過熱
    cond_rsi = df["rsi"] < 70
    # 法人 60 日累計淨買超：外資 + 投信合計 > 0（OR 改為加總，避免一買一大賣仍過關）
    if "foreign_" in df.columns or "trust" in df.columns:
        f_60d = (df["foreign_"].rolling(60, min_periods=30).sum()
                 if "foreign_" in df.columns else pd.Series(0.0, index=df.index))
        t_60d = (df["trust"].rolling(60, min_periods=30).sum()
                 if "trust" in df.columns else pd.Series(0.0, index=df.index))
        df["f_60d"] = f_60d
        df["t_60d"] = t_60d
        cond_inst_accum = (f_60d + t_60d) > inst_threshold
    else:
        df["f_60d"] = 0.0
        df["t_60d"] = 0.0
        cond_inst_accum = pd.Series(True, index=df.index)

    # PER 過濾：0 < PER < 20（有獲利且不過貴）；無資料時放行
    if per_df is not None and not per_df.empty:
        df = _merge_per(df, per_df)
        cond_per = (df["_per"] > 0) & (df["_per"] < 20) | df["_per"].isna()
    else:
        cond_per = pd.Series(True, index=df.index)

    df["signal_long"] = (cond_above_ma60 & cond_macd & cond_bb & cond_rsi & cond_inst_accum & cond_per)
    if "_per" in df.columns:
        df.rename(columns={"_per": "per"}, inplace=True)
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
    yoy_min: float = 15.0,
    accel_min: float = 5.0,
    consist_min: int = 6,
) -> pd.DataFrame:
    """
    策略五：月營收 YoY 加速 + 外資確認（基本面轉折因子）

    訊號只在每月 11 日（公布日保守估計）後的「第一個」交易日觸發。
    11 日而非 10 日：部分公司在 10 日傍晚才公布，11 日確保全市場資料可用。

    進場條件（全部滿足）：
      基本面（門檻可調，預設經 OOS 掃描優化）：
        1. YoY > yoy_min (預設 15%)
        2. 加速 > accel_min pp (預設 5pp)
        3. 過去 12 個月 ≥ consist_min 個月正成長 (預設 6)
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
        y, m = d.year, d.month
        default = pd.Timestamp(y + 1, 1, 10) if m == 12 else pd.Timestamp(y, m + 1, 10)
        if fetched is not None and not pd.isna(fetched):
            fetched_ts = pd.Timestamp(fetched)
            # 只在 fetched_date 落於預期發布窗口 ±20 天內才採用
            # 避免 bootstrap 把所有歷史資料的 fetched_date 設成同一天造成訊號爆炸
            if abs((fetched_ts - default).days) <= 20:
                return fetched_ts
        return default

    has_fetched = "fetched_date" in rev.columns
    rev["publish_date"] = [
        _pub(row["date"], row["fetched_date"] if has_fetched else None)
        for _, row in rev.iterrows()
    ]

    # ── 所有四條基本面條件通過的公布日 ───────────────────────────────────────
    cagr_ok = rev["cagr_2y"].isna() | (rev["cagr_2y"] > 0.05)
    rev_ok = rev[
        (rev["yoy"] > yoy_min) &
        (rev["yoy_accel"] > accel_min) &
        (rev["consistency"] >= consist_min) &
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
            # 把當月 revenue_yoy 寫入 signal 行，供通知格式化使用
            rev_row = rev[rev["publish_date"] == pub_ts]
            if not rev_row.empty:
                df.loc[day_mask, "revenue_yoy"] = float(rev_row.iloc[-1]["yoy"])

    # 保留供通知顯示的欄位，內部輔助欄清除
    df.rename(columns={"_foreign_20d": "f_20d", "_per": "per"}, inplace=True)
    df.drop(columns=["_inst_threshold", "_vol_20d_avg", "_ret_5d"],
            errors="ignore", inplace=True)
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


# ─── 策略 S6：高成長突破（regime-conditional aggressor）────────────────────
# 補 S4 抓不到的高 PE / 高成長飆股。AI bull regime 下 OOS Sharpe 12.8，
# 但無 AI bull 的 IS 只有 Sharpe 2.6，視為「順勢加強」而非全週期 core。

def signal_growth_breakout(df: pd.DataFrame,
                            inst_df: pd.DataFrame | None = None,
                            rev_df: pd.DataFrame | None = None,
                            market_filter: pd.Series | None = None,
                            inst_threshold: int = 5_000_000,
                            rev_growth_min: float = 10.0,
                            breakout_days: int = 60,
                            vol_mult: float = 1.5) -> pd.DataFrame:
    """
    S6 高成長突破：抓 S4 因 PER<20 條件擋下的高 PE 飆股
    （鴻勁、奇鋐、世芯-KY 這類）

    進場條件（全部滿足）：
      1. 月營收 3M sum 比 3M-3M 前成長 > rev_growth_min%
         （用 3M-vs-3M 取代 YoY，IPO 新股 < 12 個月歷史也適用）
      2. 收盤 = breakout_days 日新高（突破，而非 S4 的金叉）
      3. 當日量 > 20 日均量 × vol_mult（量增配合突破）
      4. 收盤 > MA60（清晰上升趨勢）
      5. 外資 + 投信 60 日 > inst_threshold 張

    出場（在 STRATEGIES 設定）：
      trail_trigger 0.80 / trail_pct 0.15 — 漲 +80% 才啟動 trailing
      讓贏家跑得夠久才鎖利；之前固定停利的 80% 機會會被砍掉
      勝率僅 ~38%，靠少數大贏家拉抬，連虧 5-6 筆是正常
    """
    df = add_all(df).copy()
    if inst_df is not None and not inst_df.empty:
        df = merge_institutional(df, inst_df)
    df["signal_growth"] = False

    # 月營收 3M-vs-3M：需至少 6 個月歷史才能算
    if rev_df is None or rev_df.empty or len(rev_df) < 6:
        return _apply_market_filter(df, "signal_growth", market_filter)

    rev = rev_df.sort_values("date").copy()
    rev["revenue"] = pd.to_numeric(rev["revenue"], errors="coerce")
    rev["rev_3m_sum"] = rev["revenue"].rolling(3, min_periods=3).sum()
    rev["rev_3m_growth"] = (rev["rev_3m_sum"] / rev["rev_3m_sum"].shift(3) - 1) * 100
    rev["pub_date"] = rev["date"].apply(
        lambda d: pd.Timestamp(d.year + (1 if d.month == 12 else 0),
                               1 if d.month == 12 else d.month + 1, 11))
    rev_pub = rev[["pub_date", "rev_3m_growth"]].dropna().sort_values("pub_date")
    rev_pub = rev_pub.rename(columns={"pub_date": "date"})

    df = df.sort_values("date").reset_index(drop=True)
    df["rev_3m_growth"] = pd.merge_asof(df, rev_pub, on="date", direction="backward")["rev_3m_growth"].values

    cond_rev = df["rev_3m_growth"] > rev_growth_min
    cond_break = df["close"] >= df["close"].rolling(breakout_days, min_periods=breakout_days).max()
    cond_vol = df["volume"] > df["volume"].rolling(20, min_periods=10).mean() * vol_mult
    cond_ma60 = df["close"] > df["ma60"]

    if "foreign_" in df.columns or "trust" in df.columns:
        f60 = (df["foreign_"].rolling(60, min_periods=30).sum()
               if "foreign_" in df.columns else pd.Series(0.0, index=df.index))
        t60 = (df["trust"].rolling(60, min_periods=30).sum()
               if "trust" in df.columns else pd.Series(0.0, index=df.index))
        df["f_60d"] = f60
        df["t_60d"] = t60
        cond_inst = (f60 + t60) > inst_threshold
    else:
        df["f_60d"] = 0.0
        df["t_60d"] = 0.0
        cond_inst = pd.Series(False, index=df.index)  # 沒法人資料保守不發

    df["signal_growth"] = cond_rev & cond_break & cond_vol & cond_ma60 & cond_inst
    return _apply_market_filter(df, "signal_growth", market_filter)


# ─── 策略 S7：累積前夕（狙擊手型）────────────────────────────────────────────
# 抓「法人偷收貨、股價還沒反映」的早中期累積階段。
# 訊號量少（每年 ~5 筆）、勝率中等（~44%）、但贏家可放 200-500 天賺 +60% ~ +200%。

def signal_accumulation_eve(df: pd.DataFrame,
                             inst_df: pd.DataFrame | None = None,
                             market_filter: pd.Series | None = None,
                             inst_threshold: int = 3_000_000,
                             price_chg_low: float = -5.0,
                             price_chg_high: float = 25.0,
                             pos_max: float = 0.75,
                             aqs_min: float = 70.0) -> pd.DataFrame:
    """
    S7 累積前夕：在主升段啟動「之前」就埋伏，等突破時 trailing 接力跑

    進場條件（全部滿足）：
      1. 外資 + 投信 60 日累計 ≥ inst_threshold 股
      2. 股價 60 日漲幅 in [price_chg_low, price_chg_high]
         （-5% ~ +25%：法人在累積但股價還沒被 price-in）
      3. 收盤位置 < pos_max（離 60 日高還有空間，未突破）
      4. AQS 估計 ≥ aqs_min（dim1+dim2+18+dim4 ≥ 70）
      5. AQS dim4 ≥ 5（法人 vs 股價同向，無紅旗）
      6. 大盤 loose 多頭

    出場（在 STRATEGIES 設定）：
      SL -20%（寬：給累積期足夠空間，避免被正常波動洗出）
      max_hold 180 天
      trailing trigger +80% / trail -15%（突破後讓贏家跑）

    為什麼 SL 要寬：累積階段股價自然會在 ±10% 區間波動數個月。
    若 SL 設 -10%，正常洗盤就會觸發停損；改 -20% 後 IS Sharpe
    從 6.18 翻倍到 13.29、MaxDD 從 -16% 降到 -2.52%。
    """
    df = add_all(df).copy()
    if inst_df is not None and not inst_df.empty:
        df = merge_institutional(df, inst_df)
    df["signal_accum"] = False
    if "foreign_" not in df.columns:
        return _apply_market_filter(df, "signal_accum", market_filter)

    import numpy as np
    df = df.sort_values("date").reset_index(drop=True)

    # 法人 60 日累計（外資 + 投信）
    inst_net = df["foreign_"].fillna(0) + df["trust"].fillna(0) if "trust" in df.columns else df["foreign_"].fillna(0)
    df["accum_inst_60d"] = inst_net.rolling(60, min_periods=30).sum()

    # 股價 60 日漲幅
    df["accum_price_chg_60d"] = (df["close"] / df["close"].shift(60) - 1) * 100

    # 位置（60 日區間）
    high60 = df["close"].rolling(60, min_periods=60).max()
    low60  = df["close"].rolling(60, min_periods=60).min()
    df["accum_pos"] = (df["close"] - low60) / (high60 - low60)

    # AQS 估計：dim1 + dim2 + 18(中性) + dim4
    price_chg_d = df["close"].diff()
    up_vol = np.where(price_chg_d > 0, df["volume"], 0)
    dn_vol = np.where(price_chg_d < 0, df["volume"], 0)
    up_count = (price_chg_d > 0).astype(int)
    dn_count = (price_chg_d < 0).astype(int)
    up_vol_avg = pd.Series(up_vol, index=df.index).rolling(60).sum() / up_count.rolling(60).sum().replace(0, np.nan)
    dn_vol_avg = pd.Series(dn_vol, index=df.index).rolling(60).sum() / dn_count.rolling(60).sum().replace(0, np.nan)
    vp_ratio = up_vol_avg / dn_vol_avg
    dim1 = (vp_ratio - 0.5) * 20
    dim1 = dim1.clip(0, 20)

    inst_buy = (inst_net > 0).astype(int)
    dim2 = inst_buy.rolling(60, min_periods=30).mean() * 20

    # dim4: 法人 vs 股價同向
    inst_total = df["accum_inst_60d"]
    price_chg = df["accum_price_chg_60d"]
    dim4 = pd.Series(0.0, index=df.index)
    dim4[(inst_total > 0) & (price_chg.between(-20, 5))] = 20
    dim4[(inst_total > 0) & (price_chg.between(5, 40, inclusive="right"))] = 15
    dim4[(inst_total > 0) & (price_chg > 40)] = 5
    dim4[(inst_total <= 0) & (price_chg > 30)] = -20
    dim4[(inst_total <= 0) & (price_chg.between(0, 30))] = -10
    df["accum_aqs"] = (dim1 + dim2 + 18 + dim4).clip(0, 100)
    df["accum_dim4"] = dim4

    df["signal_accum"] = ((df["accum_inst_60d"] >= inst_threshold) &
                          (df["accum_price_chg_60d"].between(price_chg_low, price_chg_high)) &
                          (df["accum_pos"] < pos_max) &
                          (df["accum_aqs"] >= aqs_min) &
                          (dim4 >= 5))

    # 把 f_60d / t_60d 加進去（給通知用）
    if "f_60d" not in df.columns:
        df["f_60d"] = df["foreign_"].rolling(60, min_periods=30).sum() if "foreign_" in df.columns else 0
    if "t_60d" not in df.columns:
        df["t_60d"] = df["trust"].rolling(60, min_periods=30).sum() if "trust" in df.columns else 0

    return _apply_market_filter(df, "signal_accum", market_filter)


# ─── 全策略清單（供批次回測用）────────────────────────────────────────────────

STRATEGIES = [
    {
        "name": "中長線_品質股低接",
        "signal_fn": signal_longterm_quality_entry,
        "signal_col": "signal_long",
        "timeframe": "long",
        "default_tp": 0.30, "default_sl": 0.10, "default_hold": 90,
        "trail_trigger": 0.20,  # 漲 20% 啟動 trailing
        "trail_pct": 0.15,      # 從峰值跌 15% 出場
        "strict_market": True,
        "needs_per": True,
        "needs_fundamental": True,
        "inst_threshold": 1_000_000,
        "retail_max_pct": 50.0,  # 排除散戶持股 >50% 的票（套牢盤厚）
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
    {
        "name": "高成長突破",
        "signal_fn": signal_growth_breakout,
        "signal_col": "signal_growth",
        "timeframe": "growth",
        "default_tp": 0.30, "default_sl": 0.10, "default_hold": 90,
        "trail_trigger": 0.80,  # 漲 +80% 才啟動 trailing（讓贏家跑）
        "trail_pct": 0.15,
        "strict_market": False,  # 用 loose mf（取消 strict 在 IS 表現較好）
        "needs_revenue": True,
        "needs_fundamental": True,
        "inst_threshold": 5_000_000,
        "rev_growth_min": 10.0,
        # ⚠️ regime-conditional: 在 AI bull (2023-25) Sharpe 12.8、
        # 無 AI bull (2019-22) Sharpe 2.6。視為「順勢加強」而非全週期 core。
        # 勝率 ~38% — 連虧 5-6 筆是正常，部位應比 S4 小。
    },
    {
        "name": "累積前夕",
        "signal_fn": signal_accumulation_eve,
        "signal_col": "signal_accum",
        "timeframe": "accum",
        "default_tp": 0.30, "default_sl": 0.20, "default_hold": 180,
        "trail_trigger": 0.80,
        "trail_pct": 0.15,
        "strict_market": False,
        "needs_fundamental": True,
        "inst_threshold": 2_000_000,
        "aqs_min": 60.0,
        # AQS 門檻 60（激進版，AI bull 加強型）
        # 🚨🚨🚨 REGIME-CONDITIONAL 警告 🚨🚨🚨
        #
        # OOS 2023-2025 (AI bull): tr=222, Sh 27.55, ann +56%, DD -0.87%
        # IS  2019-2022 (含 COVID + 升息熊): tr=142, Sh 7.89, ann +14%, IS DD -4.80%
        # 2022 單一熊市年: tr=61, 勝率 18%, ann -20%, Sharpe -18, 連敗 15 筆
        #
        # 用戶明確選擇激進版 (option C)：
        #   1. 信任 AI bull regime 會持續 1-2 年
        #   2. 承諾大盤翻空時手動關閉 S7 通知
        #   3. 接受 OOS 訊號 ~70 筆/年 的高密度
        #
        # ⚠️ 若大盤 0050 跌破 MA60 持續 2 週以上 → 強烈建議在 telegram bot
        #    手動 mute S7 通知，避免重演 2022 連敗 15 筆的慘況。
    },
]
