"""官方 bulk 抓取／寫入的單元測試。

兩個風險點：
1. 解析（fetcher）— TWSE 按欄名、TPEx 法人按位置(4/13/22)、千分位逗號、'--' 佔位符。
   全部 monkeypatch `_get`，不打網路。
2. 寫入（cache）— INSERT OR IGNORE 不覆蓋既有列、fetch_log 只前進不回退。
   用 tmp_path 暫存 SQLite。
"""
from __future__ import annotations

import pandas as pd
import pytest

from data import fetcher
from data import cache


# ─── fetcher 解析 ──────────────────────────────────────────────────────────────

def _fake_get(payload):
    """回傳一個忽略參數、固定吐 payload 的假 _get。"""
    return lambda *a, **k: payload


def test_num_cleans_commas_and_placeholders():
    assert fetcher._num("2,355.00") == 2355.0
    assert fetcher._num("30,228,535") == 30228535.0
    assert fetcher._num("--") is None
    assert fetcher._num("---") is None
    assert fetcher._num("") is None
    assert fetcher._num(None) is None
    assert fetcher._num("58.3") == 58.3


def test_twse_prices_by_date_parses_named_columns(monkeypatch):
    payload = {"stat": "OK", "tables": [{
        "fields": ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額",
                   "開盤價", "最高價", "最低價", "收盤價", "漲跌(+/-)"],
        "data": [
            ["2330", "台積電", "30,228,535", "96,307", "71,454,258,620",
             "2,360.00", "2,375.00", "2,345.00", "2,375.00", "+"],
            ["2453", "凌群", "530,189", "1,234", "30,000,000",
             "58.10", "58.60", "57.80", "58.30", "+"],
        ],
    }]}
    monkeypatch.setattr(fetcher, "_get", _fake_get(payload))
    df = fetcher.fetch_twse_prices_by_date("2026-06-17")
    assert list(df["stock_id"]) == ["2330", "2453"]
    r = df.set_index("stock_id").loc["2330"]
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == \
        (2360.0, 2375.0, 2345.0, 2375.0, 30228535.0)
    assert df["date"].unique().tolist() == ["2026-06-17"]


def test_twse_prices_drops_rows_without_close(monkeypatch):
    payload = {"stat": "OK", "tables": [{
        "fields": ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額",
                   "開盤價", "最高價", "最低價", "收盤價"],
        "data": [["9999", "停牌股", "0", "0", "0", "--", "--", "--", "--"]],
    }]}
    monkeypatch.setattr(fetcher, "_get", _fake_get(payload))
    assert fetcher.fetch_twse_prices_by_date("2026-06-17").empty


def test_twse_prices_non_trading_day_returns_empty(monkeypatch):
    monkeypatch.setattr(fetcher, "_get", _fake_get({"stat": "很抱歉，沒有符合條件的資料!"}))
    assert fetcher.fetch_twse_prices_by_date("2026-06-14").empty


def test_tpex_prices_by_date_parses_named_columns(monkeypatch):
    payload = {"stat": "ok", "tables": [{
        "fields": ["代號", "名稱", "收盤", "漲跌", "開盤", "最高", "最低",
                   "均價", "成交股數"],
        "data": [["6104", "創惟", "100.00", "+1", "102.50", "109.00",
                  "100.00", "104.0", "7,671,311"]],
    }]}
    monkeypatch.setattr(fetcher, "_get", _fake_get(payload))
    df = fetcher.fetch_tpex_prices_by_date("2026-06-10")
    r = df.set_index("stock_id").loc["6104"]
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == \
        (102.5, 109.0, 100.0, 100.0, 7671311.0)


def test_twse_inst_by_date_maps_by_field_name(monkeypatch):
    payload = {"stat": "OK", "fields": [
        "證券代號", "證券名稱",
        "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)",
        "外陸資買賣超股數(不含外資自營商)",
        "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
        "投信買進股數", "投信賣出股數", "投信買賣超股數",
        "自營商買賣超股數"],
        "data": [["2330", "台積電", "1", "2", "-15,008,392",
                  "0", "0", "0", "1", "2", "2,660,087", "-1,253,628"]]}
    monkeypatch.setattr(fetcher, "_get", _fake_get(payload))
    df = fetcher.fetch_twse_inst_by_date("2026-06-09")
    r = df.set_index("stock_id").loc["2330"]
    assert (r["foreign_"], r["trust"], r["dealer"]) == (-15008392.0, 2660087.0, -1253628.0)


def test_tpex_inst_by_date_maps_by_position(monkeypatch):
    # 24 欄；外陸資不含外資自營=idx4、投信=idx13、自營商合計=idx22（已比對 DB 驗證）
    row = ["6104", "創惟",
           "1,237,858", "404,082", "833,776",     # 2-4  外陸資(不含外資自營)
           "0", "0", "0",                          # 5-7  外資自營
           "1,237,858", "404,082", "833,776",      # 8-10 外陸資合計
           "1,000", "0", "1,000",                  # 11-13 投信
           "0", "0", "0",                          # 14-16 自營(自行)
           "118,621", "26,880", "91,741",          # 17-19 自營(避險)
           "118,621", "26,880", "91,741",          # 20-22 自營合計
           "926,517"]                              # 23 三大法人合計
    payload = {"stat": "ok", "tables": [{"fields": ["x"] * 24, "data": [row]}]}
    monkeypatch.setattr(fetcher, "_get", _fake_get(payload))
    df = fetcher.fetch_tpex_inst_by_date("2026-06-09")
    r = df.set_index("stock_id").loc["6104"]
    assert (r["foreign_"], r["trust"], r["dealer"]) == (833776.0, 1000.0, 91741.0)


# ─── cache 寫入 ────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "cache.db"
    monkeypatch.setattr(cache, "DB_PATH", db)
    cache.init_db()
    return db


def test_save_prices_bulk_writes_rows_and_fetch_log(temp_db):
    df = pd.DataFrame([
        {"stock_id": "2330", "date": "2026-06-16", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
        {"stock_id": "2330", "date": "2026-06-17", "open": 2, "high": 3, "low": 2, "close": 3, "volume": 20},
        {"stock_id": "2453", "date": "2026-06-17", "open": 5, "high": 6, "low": 5, "close": 6, "volume": 30},
    ])
    cache.save_prices_bulk(df)
    assert len(cache.load_prices("2330", start="2026-01-01")) == 2
    assert cache.last_price_date("2330") == "2026-06-17"
    assert cache.last_price_date("2453") == "2026-06-17"


def test_save_prices_bulk_insert_or_ignore_keeps_existing(temp_db):
    cache.save_prices_bulk(pd.DataFrame([
        {"stock_id": "2330", "date": "2026-06-17", "open": 2, "high": 3, "low": 2, "close": 3, "volume": 20},
    ]))
    # 同 (stock,date) 但收盤不同 → 應被忽略，不覆蓋
    cache.save_prices_bulk(pd.DataFrame([
        {"stock_id": "2330", "date": "2026-06-17", "open": 9, "high": 9, "low": 9, "close": 999, "volume": 99},
    ]))
    p = cache.load_prices("2330", start="2026-01-01")
    assert len(p) == 1
    assert p.iloc[0]["close"] == 3.0


def test_save_prices_bulk_fetch_log_only_advances(temp_db):
    cache.save_prices_bulk(pd.DataFrame([
        {"stock_id": "2330", "date": "2026-06-17", "open": 2, "high": 3, "low": 2, "close": 3, "volume": 20},
    ]))
    # 補一筆更舊的日期（回補歷史）→ fetch_log 不可退回到舊日期
    cache.save_prices_bulk(pd.DataFrame([
        {"stock_id": "2330", "date": "2026-06-10", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 5},
    ]))
    assert cache.last_price_date("2330") == "2026-06-17"
    assert len(cache.load_prices("2330", start="2026-01-01")) == 2


def test_save_institutional_bulk_and_foreign_column(temp_db):
    cache.save_institutional_bulk(pd.DataFrame([
        {"stock_id": "2330", "date": "2026-06-17", "foreign_": -100.0, "trust": 50.0, "dealer": -3.0},
    ]))
    inst = cache.load_institutional("2330", start="2026-01-01")
    assert inst.iloc[0]["foreign_"] == -100.0
    assert cache.last_institutional_date("2330") == "2026-06-17"


def test_parse_revenue_opendata_unit_and_report_month(monkeypatch):
    # 資料年月 11505 = 民國115年5月營收 → 申報月 +1 → date 2026-06-01；仟元 ×1000
    data = [{"資料年月": "11505", "公司代號": "2330",
             "營業收入-當月營收": "416975163", "營業收入-去年同月增減(%)": "30.09"}]
    df = fetcher._parse_revenue_opendata(data)
    r = df.iloc[0]
    assert r["date"] == "2026-06-01"
    assert r["revenue"] == 416975163000.0
    assert abs(r["revenue_yoy"] - 30.09) < 1e-6
    # 12 月營收 → 跨年申報隔年 1 月
    dec = [{"資料年月": "11412", "公司代號": "1101",
            "營業收入-當月營收": "1000", "營業收入-去年同月增減(%)": "5"}]
    assert fetcher._parse_revenue_opendata(dec).iloc[0]["date"] == "2026-01-01"
    # 非 list / 空 → 空 DataFrame
    assert fetcher._parse_revenue_opendata(None).empty


def test_save_monthly_revenue_bulk(temp_db):
    cache.save_monthly_revenue_bulk(pd.DataFrame([
        {"stock_id": "2330", "date": "2026-06-01", "revenue": 4.16e11, "revenue_yoy": 30.0},
    ]))
    r = cache.load_monthly_revenue("2330")
    assert len(r) == 1 and r.iloc[0]["revenue"] == 4.16e11
    assert cache.last_revenue_date("2330") == "2026-06-01"


def test_screen_today_aborts_on_stale_proxy(monkeypatch):
    """絕對日曆 gate：0050 落後現實太多 → 中止選股、回傳空訊號 + stale meta。"""
    import screener.daily_run as dr
    monkeypatch.setattr(dr, "build_market_filter", lambda **k: pd.Series(dtype=bool))
    monkeypatch.setattr(dr, "load_shareholding_latest", lambda: pd.DataFrame())
    stale = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                          "close": [100.0, 101.0]})
    monkeypatch.setattr(dr, "load_prices", lambda *a, **k: stale)
    uni = pd.DataFrame({"stock_id": ["2330"], "market": ["TWSE"]})
    out = dr.screen_today(uni, use_fundamental_filter=False)
    assert all(out[k].empty for k in ("long", "revenue", "growth", "accum", "combo_47"))
    assert out["_meta"].iloc[0]["data_stale_days"] > dr.MAX_PROXY_STALE_DAYS


def test_earliest_last_date_since(temp_db):
    cache.save_prices_bulk(pd.DataFrame([
        {"stock_id": "A", "date": "2026-06-09", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"stock_id": "B", "date": "2026-06-17", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"stock_id": "C", "date": "2026-04-30", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ]))
    # cutoff 2026-05-18：C(04-30) 不算入；活躍中最舊 = A(06-09)
    assert cache.earliest_last_date_since("price", "2026-05-18") == "2026-06-09"
    # cutoff 更早：C 也算進來
    assert cache.earliest_last_date_since("price", "2026-01-01") == "2026-04-30"


# ─── 2026-10 漏洞修補：MOPS 月營收／PER bulk／財報滾動刷新 ──────────────────────

_MOPS_HTML = """<html><body>
<table><tr><th>產業別：電腦及週邊設備業</th></tr></table>
<table>
<tr><th rowspan=2>公司 代號</th><th rowspan=2>公司名稱</th><th colspan=5>營業收入</th>
    <th colspan=3>累計營業收入</th><th rowspan=2>備註</th></tr>
<tr><th>當月營收</th><th>上月營收</th><th>去年當月營收</th><th>上月比較 增減(%)</th>
    <th>去年同月 增減(%)</th><th>當月累計營收</th><th>去年累計營收</th><th>前期比較 增減(%)</th></tr>
<tr><td>6933</td><td>AMAX-KY</td><td>585,033</td><td>665,135</td><td>338,937</td>
    <td>-12.04</td><td>72.6</td><td>3,070,638</td><td>2,736,011</td><td>12.23</td><td>-</td></tr>
<tr><td>合計</td><td></td><td>585,033</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
</table></body></html>"""


def test_parse_mops_revenue_html_units_and_total_row():
    df = fetcher._parse_mops_revenue_html(_MOPS_HTML, "2026-06-01")
    assert df["stock_id"].tolist() == ["6933"]  # 「合計」列被排除
    r = df.iloc[0]
    assert r["date"] == "2026-06-01"
    assert r["revenue"] == 585033000.0           # 仟元 × 1000
    assert abs(r["revenue_yoy"] - 72.6) < 1e-9


def test_parse_mops_revenue_html_empty_page():
    # 外國公司頁在月初常常還沒有任何申報 → 無 <table>，不可拋例外
    assert fetcher._parse_mops_revenue_html("<html><body>無資料</body></html>", "2026-10-01").empty


def test_twse_per_loss_maps_to_zero(monkeypatch):
    payload = {"stat": "OK",
               "fields": ["證券代號", "證券名稱", "收盤價", "殖利率(%)", "股利年度",
                          "本益比", "股價淨值比", "財報年/季"],
               "data": [["6933", "AMAX-KY", "371.00", "0.67", 114, "46.03", "6.26", "115/2"],
                        ["9999", "虧損股", "10.00", "0.00", 114, "-", "0.80", "115/2"]]}
    monkeypatch.setattr(fetcher, "_get", _fake_get(payload))
    df = fetcher.fetch_twse_per_by_date("2026-09-30").set_index("stock_id")
    assert df.loc["6933", "per"] == 46.03 and df.loc["6933", "pbr"] == 6.26
    # 虧損不可是 NaN：訊號端 NaN = 無資料放行，會誤放虧損股
    assert df.loc["9999", "per"] == 0.0


def test_tpex_per_na_maps_to_zero(monkeypatch):
    payload = {"stat": "ok", "tables": [{
        "fields": ["股票代號", "公司名稱", "本益比", "每股股利", "股利年度",
                   "殖利率(%)", "股價淨值比", "財報年/季"],
        "data": [["1240", "茂生農經", "10.09", "0.5", 114, "0.92", "1.60", "115Q2"],
                 ["8888", "虧損櫃", "N/A", "0", 114, "0.00", "1.10", "115Q2"]]}]}
    monkeypatch.setattr(fetcher, "_get", _fake_get(payload))
    df = fetcher.fetch_tpex_per_by_date("2026-09-30").set_index("stock_id")
    assert df.loc["1240", "per"] == 10.09
    assert df.loc["8888", "per"] == 0.0


def test_save_per_bulk_and_last_full_market_date(temp_db):
    full = pd.DataFrame([{"stock_id": f"{i:04d}", "date": "2026-05-05",
                          "per": 10.0, "pbr": 1.0, "div_yield": 2.0} for i in range(3)])
    stray = pd.DataFrame([{"stock_id": "0000", "date": "2026-06-22",
                           "per": 11.0, "pbr": 1.0, "div_yield": 2.0}])
    cache.save_per_bulk(full)
    cache.save_per_bulk(stray)
    # 零星一檔的新日期不能讓全市場看起來是新的
    assert cache.last_full_market_date("daily_per", min_rows=3) == "2026-05-05"
    assert cache.last_per_date("0000") == "2026-06-22"


def test_save_monthly_revenue_bulk_backfill_keeps_fetched_null(temp_db):
    cache.save_monthly_revenue_bulk(pd.DataFrame([
        {"stock_id": "6933", "date": "2026-06-01", "revenue": 5.85e8, "revenue_yoy": 72.6},
    ]), fetched_date=None)
    r = cache.load_monthly_revenue("6933")
    assert pd.isna(r.iloc[0]["fetched_date"])
    assert cache.revenue_month_counts("2026-01-01") == {"2026-06-01": 1}


def test_save_financial_bulk_and_coverage(temp_db):
    q = "2026-06-30"
    cache.save_financial_bulk(pd.DataFrame([
        {"stock_id": "A", "date": "2026-03-31", "type": "EPS", "value": 1.0},
        {"stock_id": "B", "date": "2026-03-31", "type": "EPS", "value": 1.0},
        {"stock_id": "A", "date": q, "type": "EPS", "value": 2.0},
    ]))
    # 不覆蓋既有（INSERT OR IGNORE）
    cache.save_financial_bulk(pd.DataFrame([{"stock_id": "A", "date": q, "type": "EPS", "value": 9.0}]))
    fa = cache.load_financial("A")
    assert fa[fa["date"] == pd.Timestamp(q)]["value"].iloc[0] == 2.0
    # 分母只算曾有財報的追蹤股（C 沒財報、像 ETF，不算）
    assert cache.financial_coverage(q, {"A", "B", "C"}) == 0.5


@pytest.mark.parametrize("day,expected", [
    ("2026-10-02", "2026-06-30"),
    ("2026-08-18", "2026-03-31"),
    ("2026-08-19", "2026-06-30"),
    ("2026-04-04", "2025-09-30"),
    ("2026-04-05", "2025-12-31"),
    ("2026-12-01", "2026-09-30"),
])
def test_latest_due_quarter(day, expected):
    from datetime import date
    from screener.daily_run import _latest_due_quarter
    assert _latest_due_quarter(date.fromisoformat(day)) == expected


# ─── 股本變動還原（2026-10：0050 一拆四沒還原 → S4 大盤濾網關 3 個月）─────────

def _px(closes, opens=None, start="2025-06-02"):
    dates = pd.bdate_range(start, periods=len(closes))
    opens = opens or closes
    return pd.DataFrame({"stock_id": "X", "date": dates, "open": opens, "high": closes,
                         "low": closes, "close": closes, "volume": [1000.0] * len(closes)})


def test_split_snaps_to_clean_ratio_and_scales_volume():
    df = _px([100, 101, 102, 101, 100, 100, 26.0, 26.5], opens=[100] * 6 + [25.5, 26.0])
    out = cache.adjust_corporate_actions(df)
    assert cache.corporate_action_events(df)[0][1] == 0.25          # 25.5/100 → 取整 1/4
    assert out["close"].iloc[5] == 25.0 and out["close"].iloc[6] == 26.0
    assert out["volume"].iloc[0] == 4000.0 and out["volume"].iloc[-1] == 1000.0


def test_capital_reduction_uses_open_ratio():
    df = _px([10, 10, 10, 10, 10, 10, 17.0], opens=[10] * 6 + [17.0])
    out = cache.adjust_corporate_actions(df)
    assert abs(out["close"].iloc[0] - 17.0) < 1e-9                 # 減資 1.7 倍（非整數）照開盤比例


def test_limit_moves_and_first_five_days_not_adjusted():
    # 漲停 +10% 不是股本變動；上市頭 5 天無漲跌幅限制的大漲也不算
    df = _px([50, 90, 95, 99, 104, 110, 121.0])
    assert cache.corporate_action_events(df) == []


def test_zero_close_rows_dropped():
    df = _px([10, 0, 10.5, 0, 11, 11, 11])
    assert (cache.adjust_corporate_actions(df)["close"] > 0).all()
    assert len(cache.adjust_corporate_actions(df)) == 5


def test_load_prices_adjusts_and_respects_start(temp_db):
    df = _px([100, 101, 102, 101, 100, 100, 26.0, 26.5], opens=[100] * 6 + [25.5, 26.0])
    cache.save_prices_bulk(df.drop(columns=[]).assign(stock_id="0050"))
    start = df["date"].iloc[4].strftime("%Y-%m-%d")
    adj = cache.load_prices("0050", start=start)
    raw = cache.load_prices("0050", start=start, adjust=False)
    assert len(adj) == len(raw) == 4
    assert adj["close"].iloc[0] == 25.0 and raw["close"].iloc[0] == 100.0
    assert cache.price_adjust_factor("0050", df["date"].iloc[2].strftime("%Y-%m-%d")) == 0.25
    assert cache.price_adjust_factor("0050", df["date"].iloc[7].strftime("%Y-%m-%d")) == 1.0


# ─── 2026-10 移除 FinMind：MOPS 財報 / 期交所 / 季別排程 ─────────────────────────

_FIN_HTML = """<table><tr><th>公司 代號</th><th>公司名稱</th><th>營業收入</th><th>營業毛利（毛損）</th>
<th>營業利益（損失）</th><th>本期淨利（淨損）</th><th>基本每股盈餘（元）</th></tr>
<tr><td>2330</td><td>台積電</td><td>2,404,483,690</td><td>1,611,606,116</td><td>1,500,000,000</td>
<td>1,279,582,227</td><td>49.34</td></tr>
<tr><td>合計</td><td></td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr></table>"""


def test_parse_mops_fin_html_maps_columns():
    df = fetcher._parse_mops_fin_html(_FIN_HTML, "t163sb04").set_index("stock_id")
    assert list(df.index) == ["2330"]
    assert df.loc["2330", "Revenue"] == 2404483690 and df.loc["2330", "EPS"] == 49.34


def test_mops_financials_decumulates_income_and_scales(monkeypatch):
    def fake(ep, year, q):
        if ep == "t163sb04":
            v = {1: (100.0, 2.0), 2: (250.0, 5.0)}[q]
            return pd.DataFrame([{"stock_id": "2330", "Revenue": v[0], "EPS": v[1]}])
        if ep == "t163sb05":
            return pd.DataFrame([{"stock_id": "2330", "Equity": 1000.0}])
        return pd.DataFrame([{"stock_id": "2330", "CashFlowsFromOperatingActivities": 80.0}])
    monkeypatch.setattr(fetcher, "fetch_mops_statement", fake)
    df = fetcher.fetch_mops_financials(2026, 2).set_index("type")["value"]
    assert df["Revenue"] == 150_000.0          # (250-100) 仟元 → 元
    assert df["EPS"] == 3.0                     # EPS 也去累計、不乘 1000
    assert df["Equity"] == 1_000_000.0          # 時點值
    assert df["CashFlowsFromOperatingActivities"] == 80_000.0   # 現金流維持年初累計


def test_taifex_futures_inst_parses_csv(monkeypatch):
    csv = ("日期,商品名稱,身份別,多方交易口數,多方交易契約金額(千元),空方交易口數,空方交易契約金額(千元),"
           "多空交易口數淨額,多空交易契約金額淨額(千元),多方未平倉口數,多方未平倉契約金額(千元),"
           "空方未平倉口數,空方未平倉契約金額(千元),多空未平倉口數淨額,多空未平倉契約金額淨額(千元)\n"
           "2026/09/29,臺股期貨,外資及陸資,1,1,1,1,0,0,8788,1,87817,1,-79029,1\n")

    class R:
        text = csv
        encoding = "big5"
    monkeypatch.setattr(fetcher._session, "post", lambda *a, **k: R())
    df = fetcher.fetch_taifex_futures_inst("TX", "2026-09-29", "2026-09-29")
    r = df.iloc[0]
    assert r["institution"] == "外資" and r["net_oi"] == -79029 and r["long_oi"] == 8788


@pytest.mark.parametrize("day,expected", [
    ("2026-10-05", [(2026, 3), (2026, 2)]),     # Q3 申報中（每天抓）＋ 已過期限的 Q2
    ("2026-11-20", [(2026, 3)]),               # Q3 期限+5 內
    ("2026-12-01", [(2026, 3)]),               # 只剩已過期的 Q3
    ("2026-02-10", [(2025, 4), (2025, 3)]),    # 年報申報中
])
def test_quarters_to_refresh(day, expected):
    from datetime import date
    from screener.daily_run import _quarters_to_refresh
    assert _quarters_to_refresh(date.fromisoformat(day)) == expected


def test_mops_statement_retries_after_refused(monkeypatch):
    calls = {"n": 0}

    class R:
        encoding = "utf-8"

        def __init__(self, text):
            self.text = text

    def post(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("Connection refused")
        return R(_FIN_HTML)
    monkeypatch.setattr(fetcher._session, "post", post)
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)
    df = fetcher.fetch_mops_statement("t163sb04", 2026, 2)
    assert "2330" in set(df["stock_id"]) and calls["n"] == 3   # sii 失敗一次＋重試成功，otc 一次


def test_radar_logs_are_separate_per_variant(temp_db):
    row = pd.DataFrame([{"date": "2026-10-06", "stock_id": "2379", "close": 760.0, "since_start_pct": 1.0,
                         "ex20_pct": 0.0, "dist52_pct": 15.0, "rev_yoy": 28.0, "retail_wchg": -0.1,
                         "trust_days": 4, "trust_20d": 1e6}])
    cache.save_radar_log(row)
    cache.save_radar_log(row.drop(columns=["trust_days", "trust_20d"]).assign(foreign_ratio=0.05,
                                                                              foreign_20d=2e6), variant="foreign")
    t, f = cache.load_radar_log(), cache.load_radar_log("foreign")
    assert len(t) == 1 and len(f) == 1                       # 同一檔同一天兩版都記得到
    assert t.iloc[0]["rules_version"].startswith("R1") and f.iloc[0]["rules_version"].startswith("F1")
    assert f.iloc[0]["foreign_ratio"] == 0.05
