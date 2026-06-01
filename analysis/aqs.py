"""
AQS (Accumulation Quality Score) — 累積品質分

判斷一支股票最近的籌碼狀況是「真實累積」還是「派發給散戶」。
分數越高越像真實累積，越低越像派發。

五個維度各 20 分，dim4（法人 vs 股價同向性）可扣分至 -20：
  1. 量價同向性：上漲日量增 / 下跌日量縮（健康）
  2. 法人買進連續性：60 日內 > 60% 天數有買 = 持續累積
  3. 單日集中度：法人 60 日累計分散在多天（好）vs 集中在 1-2 個大買日（差）
  4. 法人 vs 股價同向：法人賣但股價漲 = 紅旗（散戶接刀）
  5. 回檔日量能：回檔時量縮（惜售好）vs 量增（賣壓）

Stage 標籤獨立於 AQS：
  - 早期累積（pos<40%, RSI<55）
  - 中期（pos<70%, RSI<65, BB%<85）
  - 中末段（pos 70-85% 或 BB% 85-100%）
  - 末段（pos>85% 或 RSI>=70 或 BB%>=100）

CLI 用法：
  python -m analysis.aqs 2327
  python -m analysis.aqs 2327 7769 3563 6214
"""
from __future__ import annotations
import sys
import logging

from data.cache import init_db, load_prices, load_institutional
from data.universe import build_universe
from technical.indicators import add_all

logger = logging.getLogger(__name__)


def compute_aqs(stock_id: str, lookback: int = 60) -> dict | None:
    """計算一支股票的 AQS + Stage。資料不足回 None。"""
    price = load_prices(stock_id, start="2025-01-01")
    inst = load_institutional(stock_id, start="2025-01-01")
    if price.empty or len(price) < lookback:
        return None
    df = add_all(price.sort_values("date"))
    df = df.merge(inst, on="date", how="left", suffixes=("", "_i"))
    df["inst_net"] = df["foreign_"].fillna(0) + df["trust"].fillna(0)
    df["price_chg"] = df["close"].diff()
    last_row = df.iloc[-1]
    win = df.tail(lookback).reset_index(drop=True)

    up = win[win["price_chg"] > 0]
    dn = win[win["price_chg"] < 0]
    up_vol = up["volume"].mean() if not up.empty else 0
    dn_vol = dn["volume"].mean() if not dn.empty else 1

    # 維度 1：量價同向（0-20）
    vp_ratio = up_vol / dn_vol if dn_vol > 0 else 1
    dim1 = max(0, min(20, (vp_ratio - 0.5) * 20))

    # 維度 2：法人買進連續性（0-20）
    buy_days = (win["inst_net"] > 0).sum()
    continuity = buy_days / len(win)
    dim2 = continuity * 20

    # 維度 3：單日集中度（0-20）
    inst_pos = win["inst_net"][win["inst_net"] > 0]
    if inst_pos.sum() > 0 and len(inst_pos) >= 5:
        conc = inst_pos.nlargest(5).sum() / inst_pos.sum()
        dim3 = max(0, min(20, (1.0 - conc) * 40))
    else:
        conc = 1.0
        dim3 = 5

    # 維度 4：法人 vs 股價同向（-20 ~ +20）
    inst_total = win["inst_net"].sum()
    price_chg = (win["close"].iloc[-1] / win["close"].iloc[0] - 1) * 100
    if inst_total > 0 and -20 < price_chg <= 5:
        dim4 = 20   # 法人在低接 = 最強訊號
    elif inst_total > 0 and 5 < price_chg <= 40:
        dim4 = 15   # 健康同向上漲
    elif inst_total > 0 and price_chg > 40:
        dim4 = 5    # 法人續買但股價已飆 → 末段建倉
    elif inst_total <= 0 and price_chg > 30:
        dim4 = -20  # 紅旗：法人賣 + 股價飆 = 散戶接刀
    elif inst_total <= 0 and price_chg > 0:
        dim4 = -10
    else:
        dim4 = 0

    # 維度 5：回檔日量能（0-20）
    avg_vol = win["volume"].mean()
    dn_vol_ratio = dn_vol / avg_vol if avg_vol > 0 else 1
    dim5 = max(0, min(20, (1.3 - dn_vol_ratio) * 40))

    score = max(0, min(100, dim1 + dim2 + dim3 + dim4 + dim5))

    # Stage（獨立於 AQS）
    high60 = win["close"].max()
    low60 = win["close"].min()
    pos = (win["close"].iloc[-1] - low60) / (high60 - low60) if high60 > low60 else 0.5
    rsi = float(last_row.get("rsi", 50))
    bb = float(last_row.get("bb_pct", 0.5))
    if pos < 0.4 and rsi < 55:
        stage = "🟢 早期累積"
    elif pos < 0.7 and rsi < 65 and bb < 0.85:
        stage = "🟡 中期"
    elif pos >= 0.85 or rsi >= 70 or bb >= 1.0:
        stage = "🔴 末段"
    else:
        stage = "🟠 中末段"

    # Verdict（綜合判斷）
    if score >= 70 and "早期" in stage:
        verdict = "✅ 進場時機好"
    elif score >= 70 and "中期" in stage:
        verdict = "✅ 仍可進，留意位置"
    elif score >= 70 and "末段" in stage:
        verdict = "⚠️ 法人在續買但末段，部位減半"
    elif 50 <= score < 70:
        verdict = "🟡 訊號中性，不主動"
    elif score < 50 and dim4 < 0:
        verdict = "🚫 派發 trap，避開"
    else:
        verdict = "🚫 訊號不健康"

    return {
        "stock_id": stock_id,
        "score": round(score, 1),
        "dim1_volprice": round(dim1, 1),
        "dim2_continuity": round(dim2, 1),
        "dim3_concentration": round(dim3, 1),
        "dim4_inst_price_align": round(dim4, 1),
        "dim5_pullback_vol": round(dim5, 1),
        "stage": stage,
        "verdict": verdict,
        "pos_pct": round(pos * 100, 0),
        "rsi": round(rsi, 0),
        "bb_pct": round(bb * 100, 0),
        "raw_up_dn_vol_ratio": round(vp_ratio, 2),
        "raw_buy_day_pct": round(continuity * 100, 1),
        "raw_top5_conc_pct": round(conc * 100, 1),
        "raw_inst_60d": int(inst_total),
        "raw_price_chg_60d_pct": round(price_chg, 1),
        "raw_dn_vol_ratio": round(dn_vol_ratio, 2),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.aqs <stock_id> [<stock_id> ...]")
        return 1
    init_db()
    universe = build_universe()
    name_map = dict(zip(universe["stock_id"], universe["stock_name"])) if not universe.empty else {}

    sids = sys.argv[1:]
    print(f"{'SID':<6}{'名稱':<14}{'AQS':>6}{'量價':>6}{'連續':>6}{'集中':>6}{'法:價':>7}{'回量':>6}  Stage          Verdict")
    print("-" * 120)
    for sid in sids:
        r = compute_aqs(sid)
        if r is None:
            print(f"{sid:<6}{name_map.get(sid, '?'):<14}  資料不足")
            continue
        print(f"{sid:<6}{name_map.get(sid, '?'):<14}"
              f"{r['score']:>6}{r['dim1_volprice']:>6}{r['dim2_continuity']:>6}"
              f"{r['dim3_concentration']:>6}{r['dim4_inst_price_align']:>+7}{r['dim5_pullback_vol']:>6}"
              f"  {r['stage']:<14} {r['verdict']}")
        print(f"  └ 位置 {r['pos_pct']:.0f}%  RSI {r['rsi']:.0f}  BB% {r['bb_pct']:.0f}  "
              f"法人60日 {r['raw_inst_60d']:+,}  股價60日 {r['raw_price_chg_60d_pct']:+.1f}%  "
              f"前5買日佔 {r['raw_top5_conc_pct']:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
