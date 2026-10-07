"""成績單用的統計檢定：「這個超額報酬是不是運氣？」「還要多少筆才能下結論？」

方法參考 mars-tw/anti-gambling-trader-tw（MIT）的 verdict/statistics，但改了一個關鍵處：
那邊假設每筆交易彼此獨立；我們的訊號常在同一週一次出好幾檔、大盤一跌一起跌，
直接當獨立樣本會高估顯著性。所以一律以「進場週」為群組（cluster）：
- 檢定與信賴區間都對「整週」重抽（cluster bootstrap），t 檢定用週平均當樣本
- 所需樣本量以週數估算，再換算成筆數

p 值語意：假設其實沒有優勢（平均超額 ≤ 0），純靠運氣出現至少這麼好成績的機率。
不是「優勢為真的機率」。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

Z_ALPHA_ONE_SIDED = 1.6449   # 單尾 α = 0.05
Z_POWER_80 = 0.8416          # 80% 檢定力


@dataclass
class Verdict:
    n: int                 # 事件數
    n_clusters: int        # 週數（有效獨立樣本）
    mean: float            # 平均超額報酬（%）
    ci_low: float          # 95% 信賴區間（cluster bootstrap）
    ci_high: float
    p_value: float         # 單尾 p（cluster bootstrap，shift method）
    p_t: float             # 單尾 p（週平均的 t 檢定，參考）
    need_n: int | None     # 以目前效果大小，80% 檢定力需要的總筆數（負期望 → None）
    top_share: float       # 最好的一筆佔總超額獲利的比例（集中度紅旗 > 50%）

    @property
    def label(self) -> str:
        if self.n < 2 or self.n_clusters < 3:
            return "樣本太少"
        if self.mean <= 0:
            return "❌ 沒有優勢"
        if self.p_value < 0.05:
            return "✅ 顯著"
        return "🟡 正但不顯著（可能是運氣）"


def _cluster_means(values: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"v": values, "c": clusters})
    g = df.groupby("c")["v"]
    return g.sum().to_numpy(), g.size().to_numpy()


def verdict(values, clusters, n_boot: int = 5000, seed: int = 1234) -> Verdict:
    """values：每筆超額報酬（%）；clusters：同長度的群組標籤（例：進場週）。"""
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    m = ~np.isnan(v)
    v, c = v[m], c[m]
    n = len(v)
    if n < 2:
        mean = float(v.mean()) if n else 0.0
        return Verdict(n, len(set(c)), mean, mean, mean, 1.0, 1.0, None, 0.0)
    mean = float(v.mean())
    sums, sizes = _cluster_means(v, c)
    k = len(sums)
    # cluster bootstrap：重抽週，均值 = 抽到的週總和 / 抽到的週筆數
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    # H0（平均 0）世界：每筆減去樣本平均後同樣重抽
    sums0 = sums - mean * sizes
    boot0 = sums0[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    p_boot = float((boot0 >= mean).mean())
    # 週平均的 t 檢定（參考）
    wk = sums / sizes
    if k >= 2 and wk.std(ddof=1) > 0:
        t = wk.mean() / (wk.std(ddof=1) / math.sqrt(k))
        p_t = float(stats.t.sf(t, k - 1))
    else:
        p_t = 1.0
    need = required_n(v, c)
    pos = v[v > 0]
    top_share = float(pos.max() / pos.sum()) if len(pos) else 0.0
    return Verdict(n, k, mean, float(ci_low), float(ci_high), p_boot, p_t, need, top_share)


def required_n(values, clusters, alpha_z: float = Z_ALPHA_ONE_SIDED,
               power_z: float = Z_POWER_80) -> int | None:
    """以目前的效果大小與週間波動，要多少「筆」才有 80% 機率在 α=0.05 下判定顯著。

    n_週 ≈ ((z_α + z_power) · sd_週 / mean)²，再乘上平均每週筆數。負期望 → None。
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    m = ~np.isnan(v)
    v, c = v[m], c[m]
    if len(v) < 2 or v.mean() <= 0:
        return None
    sums, sizes = _cluster_means(v, c)
    wk = sums / sizes
    if len(wk) < 2 or wk.std(ddof=1) == 0:
        return max(30, len(v))
    need_weeks = ((alpha_z + power_z) * wk.std(ddof=1) / v.mean()) ** 2
    return int(math.ceil(max(need_weeks, 3) * sizes.mean()))


def welch_compare(a, a_clusters, b, b_clusters) -> tuple[float, float]:
    """兩組平均超額是否不同（雙尾 Welch，以週平均為樣本）。回傳 (平均差, p)。"""
    def wk(vals, cl):
        v = np.asarray(vals, dtype=float)
        c = np.asarray(cl)
        m = ~np.isnan(v)
        s, z = _cluster_means(v[m], c[m])
        return s / z
    wa, wb = wk(a, a_clusters), wk(b, b_clusters)
    if len(wa) < 2 or len(wb) < 2:
        return float("nan"), 1.0
    t = stats.ttest_ind(wa, wb, equal_var=False)
    return float(np.nanmean(a) - np.nanmean(b)), float(t.pvalue)


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm 多重比較校正：同時評估多個工具時，誰在校正後仍顯著。

    7 個工具各做一次 α=0.05 的檢定、全部都沒有優勢的情況下，
    至少一個「看起來顯著」的機率約 30%——所以要校正。
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, still = {}, True
    for i, (name, p) in enumerate(items):
        still = still and p <= alpha / (m - i)
        out[name] = still
    return out


def losing_streak_prob(n_trades: int, streak: int, win_rate: float) -> float:
    """以這個勝率，未來 n 筆裡至少出現一次連虧 streak 次的機率（精確 DP）。"""
    q = 1.0 - win_rate
    if streak <= 0:
        return 1.0
    # state[j] = 目前連虧 j 次、尚未出現 streak 連虧 的機率
    state = np.zeros(streak)
    state[0] = 1.0
    hit = 0.0
    for _ in range(n_trades):
        new = np.zeros(streak)
        new[0] = state.sum() * (1 - q)
        new[1:] = state[:-1] * q
        hit += state[-1] * q
        state = new
    return float(hit)


def week_key(dates) -> np.ndarray:
    """進場日 → 週標籤（同一週的訊號視為一組）。"""
    d = pd.to_datetime(pd.Series(dates))
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d").to_numpy()


def fmt(v: Verdict) -> str:
    """一行白話摘要。"""
    if v.n < 2:
        return f"{v.n} 筆，樣本太少"
    need = (f"｜以目前效果需約 {v.need_n} 筆才能下結論（已 {v.n}）"
            if v.need_n else "")
    flag = f"｜⚠️ 最好一筆佔總超額 {v.top_share:.0%}" if v.top_share > 0.5 else ""
    head = f"{v.label}：平均超額 {v.mean:+.2f}%（95% 區間 {v.ci_low:+.2f}～{v.ci_high:+.2f}）"
    luck = (f"，假如沒有優勢、純靠運氣出現這麼好成績的機率 {v.p_value:.1%}" if v.mean > 0 else "")
    return f"{head}{luck}，{v.n} 筆／{v.n_clusters} 週{need}{flag}"
