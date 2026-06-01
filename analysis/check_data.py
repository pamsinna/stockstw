"""
持股資料完整性檢查 — sync_db.sh 後跑一次，確認每支持股最近幾天的
price + institutional 都有抓到。漏資料就警告。

用法:
  python -m analysis.check_data           # 檢查 portfolio.yml 裡所有持股
  python -m analysis.check_data 8299 6414 # 檢查特定股票
"""
import sys
import yaml
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

from data.cache import init_db, load_prices, load_institutional
from data.universe import build_universe


def expected_trading_days(end_date: date, n: int = 5) -> list[date]:
    """回推 n 個交易日（粗略：跳過週末）"""
    days = []
    d = end_date
    while len(days) < n:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d -= timedelta(days=1)
    return list(reversed(days))


def check_one(sid: str, name: str, n_days: int = 5) -> dict:
    price = load_prices(sid, start="2026-05-15")
    inst = load_institutional(sid, start="2026-05-15")
    if price.empty:
        return {"stock_id": sid, "name": name, "error": "無價格資料"}

    last_price_date = pd.to_datetime(price["date"].max()).date()
    last_inst_date = pd.to_datetime(inst["date"].max()).date() if not inst.empty else None

    # 預期最近 5 個交易日（用本機 today 或 last_price_date 較新者推回去）
    today = date.today()
    ref = max(last_price_date, today)
    expected = expected_trading_days(ref, n_days)

    actual_price_dates = set(pd.to_datetime(price["date"]).dt.date)
    actual_inst_dates = set(pd.to_datetime(inst["date"]).dt.date) if not inst.empty else set()

    missing_price = [d for d in expected if d not in actual_price_dates]
    missing_inst = [d for d in expected if d not in actual_inst_dates]

    return {
        "stock_id": sid,
        "name": name,
        "last_price": last_price_date,
        "last_inst": last_inst_date,
        "expected": expected,
        "missing_price": missing_price,
        "missing_inst": missing_inst,
    }


def main():
    init_db()
    universe = build_universe()
    name_map = dict(zip(universe["stock_id"], universe["stock_name"])) if not universe.empty else {}

    # 從 CLI 或 portfolio.yml 拿股票列表
    if len(sys.argv) > 1:
        sids = sys.argv[1:]
    else:
        if not Path("portfolio.yml").exists():
            print("❌ 找不到 portfolio.yml，請傳入股票代號：python -m analysis.check_data 8299")
            return 1
        with open("portfolio.yml", encoding="utf-8") as f:
            p = yaml.safe_load(f)
        sids = [str(h["stock_id"]) for h in p.get("holdings", [])]

    print("\n📋 持股資料完整性檢查（檢查近 5 個交易日）")
    print("=" * 70)

    errors = 0
    for sid in sids:
        name = name_map.get(sid, "?")
        r = check_one(sid, name)
        label = f"{sid} {name}"[:18]
        if "error" in r:
            print(f"  ❌ {label:<18} {r['error']}")
            errors += 1
            continue
        if not r["missing_price"] and not r["missing_inst"]:
            print(f"  ✅ {label:<18} 最新 price={r['last_price']} inst={r['last_inst']}")
        else:
            errors += 1
            print(f"  ⚠️ {label:<18} 最新 price={r['last_price']} inst={r['last_inst']}")
            if r["missing_price"]:
                print(f"      漏價格: {', '.join(d.isoformat() for d in r['missing_price'])}")
            if r["missing_inst"]:
                print(f"      漏法人: {', '.join(d.isoformat() for d in r['missing_inst'])}")

    print(f"\n總結: 完整 {len(sids)-errors} 支 / 有缺 {errors} 支")
    if errors > 0:
        print("\n建議：重跑 daily screen 補資料（python main.py screen 或觸發 GitHub Actions）")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
