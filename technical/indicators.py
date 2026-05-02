"""
技術指標計算：純 pandas/numpy，不依賴 ta-lib（避免安裝困難）
輸入 OHLCV DataFrame，輸出附加指標欄位的 DataFrame
"""
import pandas as pd


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """一次計算所有指標，回傳原始 df + 新增欄位"""
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = _moving_averages(df)
    df = _bollinger(df)
    df = _kd(df)
    df = _macd(df)
    df = _volume_features(df)
    df = _rsi(df)
    return df


# ─── 均線 ──────────────────────────────────────────────────────────────────────

def _moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    for n in [5, 10, 20, 60, 120, 240]:
        df[f"ma{n}"] = df["close"].rolling(n).mean()
    df["ma_aligned"] = (
        (df["ma5"] > df["ma10"]) &
        (df["ma10"] > df["ma20"]) &
        (df["ma20"] > df["ma60"])
    )
    return df


# ─── 布林通道 ─────────────────────────────────────────────────────────────────

def _bollinger(df: pd.DataFrame, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    df["bb_upper"] = mid + k * std
    df["bb_mid"]   = mid
    df["bb_lower"] = mid - k * std
    df["bb_width"]   = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]     = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["bb_breakout"] = df["close"] > df["bb_upper"]
    df["bb_squeeze"]  = df["bb_width"] < df["bb_width"].rolling(120).quantile(0.2)
    return df


# ─── KD 隨機指標 ──────────────────────────────────────────────────────────────

def _kd(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    low_n  = df["low"].rolling(period).min()
    high_n = df["high"].rolling(period).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100

    k_vals = [50.0]
    d_vals = [50.0]
    for i in range(1, len(df)):
        rsv_i = rsv.iloc[i]
        if pd.isna(rsv_i):
            k_vals.append(k_vals[-1])
            d_vals.append(d_vals[-1])
        else:
            k_new = k_vals[-1] * 2/3 + rsv_i * 1/3
            d_new = d_vals[-1] * 2/3 + k_new * 1/3
            k_vals.append(k_new)
            d_vals.append(d_new)

    df["kd_k"] = k_vals
    df["kd_d"] = d_vals
    df["kd_golden"] = (df["kd_k"] > df["kd_d"]) & (df["kd_k"].shift(1) <= df["kd_d"].shift(1))
    df["kd_death"]  = (df["kd_k"] < df["kd_d"]) & (df["kd_k"].shift(1) >= df["kd_d"].shift(1))
    df["kd_oversold"]   = df["kd_k"] < 20
    df["kd_overbought"] = df["kd_k"] > 80
    return df


# ─── MACD ──────────────────────────────────────────────────────────────────────

def _macd(df: pd.DataFrame,
          fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    hist     = macd - sig

    df["macd"]        = macd
    df["macd_signal"] = sig
    df["macd_hist"]   = hist
    df["macd_golden"]  = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    df["macd_death"]   = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    df["macd_positive"] = macd > 0
    df["macd_hist_up"]  = hist > hist.shift(1)
    return df


# ─── 成交量特徵 ────────────────────────────────────────────────────────────────

def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    df["vol_surge"]  = df["vol_ratio"] > 2.0   # 量暴增
    df["vol_shrink"] = df["vol_ratio"] < 0.5   # 量縮

    # 突破近 N 日高點
    for n in [10, 20, 60]:
        df[f"new_high_{n}"] = df["close"] >= df["high"].rolling(n).max().shift(1)

    # 價格相對強弱
    df["price_change"]   = df["close"].pct_change()
    df["price_change_5"] = df["close"].pct_change(5)
    return df


# ─── RSI ───────────────────────────────────────────────────────────────────────

def _rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_oversold"]   = df["rsi"] < 30
    df["rsi_overbought"] = df["rsi"] > 70
    return df


# ─── 法人籌碼附加 ─────────────────────────────────────────────────────────────

def merge_institutional(price_df: pd.DataFrame,
                        inst_df: pd.DataFrame) -> pd.DataFrame:
    """把三大法人資料 merge 進日K"""
    if inst_df.empty:
        for col in ["foreign_", "trust", "dealer", "inst_total",
                    "inst_consecutive_buy"]:
            price_df[col] = 0.0
        return price_df

    merged = price_df.merge(inst_df, on="date", how="left")
    for col in ["foreign_", "trust", "dealer"]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0)

    merged["inst_total"] = merged["foreign_"] + merged["trust"] + merged["dealer"]

    # 外資 + 投信同步買超
    merged["inst_sync_buy"] = (merged["foreign_"] > 0) & (merged["trust"] > 0)

    # 法人連續買超天數
    buy = (merged["inst_total"] > 0).astype(int)
    groups = (buy != buy.shift(1)).cumsum()
    merged["inst_consecutive_buy"] = buy.groupby(groups).cumsum() * buy
    return merged
