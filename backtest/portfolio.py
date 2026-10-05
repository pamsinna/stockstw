"""投資組合層回測：資金有限、同時最多 N 檔、每檔 1/N 淨值、對照 0050 含息。

舊回測只算「每筆交易平均報酬」——不知道同時持有幾檔、錢夠不夠、訊號擠在同一天
要買哪幾檔，所以回答不了「這套系統值不值得用、該配多少錢」。

規則（與實盤一致）：
- 訊號 = backtest.pit_signals（point-in-time 基本面／散戶濾網）；進出場 = 回測引擎
  （T+1 開盤進、停損/停利/trailing、跳空用開盤價）
- 同時最多 max_positions 檔；新部位 = 前一日淨值 / max_positions（現金不夠就用剩下的）
- 同一天訊號太多 → 依訊號日 20 日動能由強到弱挑（同通知排序）
- 同一檔股票只持有一次（跨策略不重複）
- 成本：買手續費、賣手續費＋證交稅；個股不含股利（偏保守），0050 對照含息

用法：
  python -m backtest.portfolio                 # 全策略合併組合 + 各策略單獨，2019-01～最新
  python -m backtest.portfolio --positions 15
"""
from __future__ import annotations

import argparse
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.pit_signals import strategy_signals
from backtest.run_backtest import build_market_filter
from config import FEE_RATE_BUY, FEE_RATE_SELL, TAX_TWSE_OTC, TAX_EMERGING
from data.cache import load_prices
from technical.signals import STRATEGIES

logger = logging.getLogger(__name__)

LIVE_KEYS = ("long", "revenue", "growth", "accum")   # 實盤在跑的 S4～S7
TAG = {"long": "S4", "revenue": "S5", "growth": "S6", "accum": "S7"}
DATA_START_FOR_FILTER = "2018-01-01"
PERIODS = {"IS 2019-22": ("2019-01-01", "2022-12-31"),
           "OOS 2023-25": ("2023-01-01", "2025-12-31"),
           "2026 YTD": ("2026-01-01", "2026-12-31")}


# ─── 候選交易 ─────────────────────────────────────────────────────────────────

def collect_trades(stocks: list[str], start: str, end: str, keys=LIVE_KEYS,
                   market_map: dict[str, str] | None = None,
                   gate_overrides: dict[str, bool] | None = None) -> pd.DataFrame:
    """每個策略、每檔股票跑回測引擎 → 候選交易表（含訊號日動能，供同日排序）。"""
    mf = build_market_filter(DATA_START_FOR_FILTER, end)
    smf = build_market_filter(DATA_START_FOR_FILTER, end, strict=True)
    rows = []
    for key in keys:
        st = next(s for s in STRATEGIES if s["timeframe"] == key)
        gate = (gate_overrides or {}).get(key)
        sigs = strategy_signals(st, stocks, end, mf, smf, gate_fundamental=gate)
        col = st["signal_col"]
        for sid, df in sigs.items():
            if not df[col].any():
                continue
            mkt = (market_map or {}).get(sid, "TWSE")
            res = run_backtest(df, col, st["default_tp"], st["default_sl"], st["default_hold"],
                               start, end, sid, mkt, trail_trigger=st.get("trail_trigger"),
                               trail_pct=st.get("trail_pct", 0.15))
            if not res.trades:
                continue
            d = df.set_index("date")["close"]
            idx = {dt: i for i, dt in enumerate(d.index)}
            for t in res.trades:
                i = idx.get(t.entry_date)
                if i is None or i < 1:
                    continue
                sig_i = i - 1
                mom = d.iloc[sig_i] / d.iloc[sig_i - 20] - 1 if sig_i >= 20 else np.nan
                rows.append({"strategy": TAG[key], "stock_id": sid, "market": mkt,
                             "signal_date": d.index[sig_i], "entry_date": t.entry_date,
                             "entry_price": t.entry_price, "exit_date": t.exit_date,
                             "exit_price": t.exit_price, "exit_reason": t.exit_reason,
                             "pnl_pct": t.pnl_pct, "mom20": mom})
    return pd.DataFrame(rows)


# ─── 組合模擬 ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def _closes(sid: str) -> pd.Series:
    px = load_prices(sid, start="2018-01-01")
    return px.set_index("date")["close"] if not px.empty else pd.Series(dtype=float)


def simulate(trades: pd.DataFrame, start: str, end: str, max_positions: int = 10,
             initial: float = 1.0) -> tuple[pd.Series, pd.DataFrame]:
    """逐日模擬 → (淨值曲線, 實際成交的交易)。"""
    days = _closes("0050").loc[start:end].index
    tr = trades[(trades.entry_date >= pd.Timestamp(start)) & (trades.entry_date <= pd.Timestamp(end))]
    by_entry = {d: g.sort_values("mom20", ascending=False, na_position="last")
                for d, g in tr.groupby("entry_date")}
    cash, equity_prev = initial, initial
    pos: dict[str, dict] = {}
    taken, curve = [], []

    def close_positions(d):
        nonlocal cash
        for sid in [s for s, p in pos.items() if p["exit_date"] == d]:
            p = pos.pop(sid)
            tax = TAX_EMERGING if p["market"] == "Emerging" else TAX_TWSE_OTC
            cash += p["units"] * p["exit_price"] * (1 - FEE_RATE_SELL - tax)

    for d in days:
        close_positions(d)
        for _, t in by_entry.get(d, pd.DataFrame()).iterrows():
            if len(pos) >= max_positions or t.stock_id in pos:
                continue
            alloc = min(equity_prev / max_positions, cash)
            if alloc <= equity_prev / max_positions * 0.25:   # 現金太少就不硬塞
                continue
            cash -= alloc
            pos[t.stock_id] = {"units": alloc / (t.entry_price * (1 + FEE_RATE_BUY)),
                               "exit_date": t.exit_date, "exit_price": t.exit_price,
                               "market": t.market, "last": t.entry_price}
            taken.append(t)
        close_positions(d)   # 進場當天就觸發出場的
        mv = 0.0
        for sid, p in pos.items():
            c = _closes(sid).get(d)
            if c is not None and not pd.isna(c):
                p["last"] = c
            mv += p["units"] * p["last"]
        equity_prev = cash + mv
        curve.append((d, equity_prev, mv / equity_prev if equity_prev else 0.0))
    eq = pd.DataFrame(curve, columns=["date", "equity", "exposure"]).set_index("date")
    return eq, pd.DataFrame(taken)


# ─── 基準：0050（含息）────────────────────────────────────────────────────────

def benchmark_0050(start: str, end: str, total_return: bool = True) -> pd.Series:
    c = _closes("0050").loc[start:end]
    if not total_return:
        return c / c.iloc[0]
    from data.fetcher import _finmind
    div = _finmind("TaiwanStockDividendResult", "0050", "2018-01-01")
    factor = pd.Series(1.0, index=c.index)
    if div is not None and not div.empty:
        for _, r in div.iterrows():
            d = pd.Timestamp(r["date"])
            if d in factor.index and r["after_price"] > 0:
                factor.loc[d:] *= r["before_price"] / r["after_price"]
    tr = c * factor
    return tr / tr.iloc[0]


# ─── 指標 ─────────────────────────────────────────────────────────────────────

def perf(eq: pd.Series, bench: pd.Series | None = None) -> dict:
    eq = eq.dropna()
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    out = {"總報酬%": (eq.iloc[-1] / eq.iloc[0] - 1) * 100,
           "年化%": ((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(yrs, 1e-9)) - 1) * 100,
           "最大回撤%": (eq / eq.cummax() - 1).min() * 100,
           "Sharpe": r.mean() / (r.std() + 1e-12) * np.sqrt(250)}
    if bench is not None:
        b = bench.reindex(eq.index).pct_change().dropna()
        x = pd.concat([r, b], axis=1, join="inner").dropna()
        if len(x) > 20:
            beta = np.cov(x.iloc[:, 0], x.iloc[:, 1])[0, 1] / x.iloc[:, 1].var()
            alpha = (x.iloc[:, 0].mean() - beta * x.iloc[:, 1].mean()) * 250 * 100
            out.update({"Beta": beta, "Alpha年化%": alpha})
    return out


def report(eq: pd.DataFrame, bench: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    for name, (a, b) in PERIODS.items():
        e = eq["equity"].loc[a:b]
        if len(e) < 20:
            continue
        rows.append({"組合": label, "期間": name, **perf(e, bench.loc[a:b]),
                     "平均持股水位%": eq["exposure"].loc[a:b].mean() * 100})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=10)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    from data.universe import build_universe
    uni = build_universe()
    mm = dict(zip(uni.stock_id, uni.market))
    trades = collect_trades(uni.stock_id.tolist(), args.start, args.end, market_map=mm)
    bench = benchmark_0050(args.start, args.end)
    out = [report(pd.DataFrame({"equity": bench, "exposure": 1.0}), bench, "0050 含息")]
    eq, taken = simulate(trades, args.start, args.end, args.positions)
    out.append(report(eq, bench, f"S4～S7 合併（{args.positions} 檔）"))
    for tag in ("S4", "S5", "S6", "S7"):
        e, _ = simulate(trades[trades.strategy == tag], args.start, args.end, args.positions)
        out.append(report(e, bench, f"{tag} 單獨"))
    pd.set_option("display.width", 250)
    print(pd.concat(out).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
