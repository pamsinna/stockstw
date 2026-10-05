"""實盤成績單：凍結規則（RULES_VERSION）之後發出的訊號，照策略規則實際表現 vs 同期 0050 含息。

這是唯一「真正的樣本外」：2023-25 回測資料已被反覆拿來做決策，數字會偏樂觀。

  python scripts/live_scorecard.py            # 印成績單
  python scripts/live_scorecard.py --all      # 不限版本（含凍結前的舊訊號，僅供參考）

每筆：進場 = 訊號隔天開盤；已出場用出場價、未出場用最新收盤（同 notify.exit_monitor.rule_status）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.portfolio import benchmark_0050  # noqa: E402
from data.cache import load_open_signals  # noqa: E402
from notify.exit_monitor import rule_status  # noqa: E402
from technical.signals import RULES_VERSION, RULES_FROZEN_SINCE  # noqa: E402


def scorecard(include_all: bool = False) -> pd.DataFrame:
    log = load_open_signals()
    if log.empty:
        return pd.DataFrame()
    if not include_all:
        ver = log["rules_version"] if "rules_version" in log.columns else pd.Series("", index=log.index)
        log = log[(ver == RULES_VERSION) & (log["status"] != "not_notified")]
    if log.empty:
        return pd.DataFrame()
    bench = benchmark_0050(str(log["entry_date"].min()), pd.Timestamp.today().strftime("%Y-%m-%d"))
    rows = []
    for _, r in log.iterrows():
        st = rule_status(str(r["stock_id"]), str(r["strategy"]), str(r["entry_date"]))
        if not st or st["state"] == "pending":
            continue
        end = st.get("exit_date") or bench.index[-1]
        b = bench.loc[st["entry_date"]:end]
        rows.append({"strategy": r["strategy"], "stock_id": r["stock_id"], "name": r["name"],
                     "entry_date": st["entry_date"], "state": st["state"],
                     "ret%": st["pnl_pct"],
                     "0050%": (b.iloc[-1] / b.iloc[0] - 1) * 100 if len(b) > 1 else 0.0})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def f(g):
        return pd.Series({"筆數": len(g), "已出場": (g.state == "exit").sum(),
                          "平均%": g["ret%"].mean(), "中位數%": g["ret%"].median(),
                          "勝率%": (g["ret%"] > 0).mean() * 100,
                          "同期0050%": g["0050%"].mean(),
                          "贏0050比例%": (g["ret%"] > g["0050%"]).mean() * 100})
    out = df.groupby("strategy").apply(f)
    out.loc["合計"] = f(df)
    return out.round(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    df = scorecard(include_all=args.all)
    label = "全部（含凍結前）" if args.all else f"規則版本 {RULES_VERSION}（{RULES_FROZEN_SINCE} 起）"
    if df.empty:
        print(f"{label}：尚無可評估的訊號")
        return
    pd.set_option("display.width", 200)
    print(f"實盤成績單 — {label}")
    print(summarize(df).to_string())


if __name__ == "__main__":
    main()
