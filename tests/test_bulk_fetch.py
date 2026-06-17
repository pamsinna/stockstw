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
