"""每日通知格式（2026-10 改版：照動作分三層、每檔一列、規則另發置頂）。

name map / 動能 / 實盤 log 都 monkeypatch 掉 → 不碰 DB、不碰網路。
"""
from __future__ import annotations

import pandas as pd
import pytest

import notify.telegram_bot as tg


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    monkeypatch.setattr(tg, "_name_map", lambda: {"1513": "中興電", "2451": "創見", "6933": "AMAX-KY",
                                                  "2305": "全友", "3653": "健策"})
    monkeypatch.setattr(tg, "_mom20", lambda sid, date: {"1513": 0.02, "2451": -0.02}.get(sid, 0.05))
    monkeypatch.setattr(tg, "_load_recent_log", lambda n=20: pd.DataFrame())
    monkeypatch.setattr(tg, "_append_signal_log", lambda df, date: None)


def _row(sid, **kw):
    return {"stock_id": sid, "close": 100.0, "f_60d": 2_000_000.0, "t_60d": 0.0, **kw}


def test_candidates_merge_one_row_per_stock_with_tags():
    sig = {"long": pd.DataFrame([_row("2451"), _row("1513")]),
           "accum": pd.DataFrame([_row("1513", aqs_score=75, aqs_stage="中期")])}
    c = tg._merge_candidates(sig, drop=set())
    assert sorted(c["stock_id"]) == ["1513", "2451"]
    assert c.set_index("stock_id").loc["1513", "tags"] == ["S4", "S7"]
    # S7 的 AQS 補進 S4 先建立的那列
    assert c.set_index("stock_id").loc["1513", "aqs_score"] == 75


def test_combo_adds_recent_tag_with_star():
    sig = {"accum": pd.DataFrame([_row("1513")]),
           "combo_47": pd.DataFrame([_row("1513", s4_today=False, s7_today=True)])}
    c = tg._merge_candidates(sig, drop=set())
    assert c.iloc[0]["tags"] == ["S4*", "S7"]


def test_exit_stocks_dropped_from_candidates():
    sig = {"long": pd.DataFrame([_row("2451"), _row("1513")])}
    c = tg._merge_candidates(sig, drop={"1513"})
    assert c["stock_id"].tolist() == ["2451"]


def test_format_three_tiers_in_order_and_single_message():
    sig = {
        "_meta": pd.DataFrame([{"regime_label": "🔥 多頭", "regime_60d_return": 0.08,
                                "credit_stress": "🌡 HY OAS 2.9%"}]),
        "exits": pd.DataFrame([{"level": "🚨 出場", "stock_id": "2451", "name": "創見", "strategy": "S4",
                                "entry_date": "2026-09-12", "entry_price": 58.3, "close": 61.0,
                                "pnl_pct": 4.6, "reason": "外資連6日賣"}]),
        "long": pd.DataFrame([_row("2451"), _row("1513")]),
        "accum": pd.DataFrame([_row("1513")]),
        "watch": pd.DataFrame([
            {"stock_id": "6933", "industry": "電腦及週邊設備業", "close": 371.0, "since_pct": 7.0,
             "is_new": False, "burst_yoy": 191.0, "burst_g3": 108.0, "f_60d": 1_655_000.0},
            {"stock_id": "2305", "industry": "電腦及週邊設備業", "close": 68.7, "since_pct": 0.0,
             "is_new": True, "burst_yoy": 240.0, "burst_g3": 57.0, "f_60d": -175_000.0},
            {"stock_id": "3653", "industry": "電子零組件業", "close": 6800.0, "since_pct": -2.0,
             "is_new": False, "burst_yoy": 91.0, "burst_g3": 34.0, "f_60d": -1_265_000.0},
        ]),
    }
    msgs = tg.format_signals(sig, "2026-10-01")
    assert len(msgs) == 1
    m = msgs[0]
    i_exit, i_cand, i_watch, i_ref = (m.index("要處理"), m.index("新進場候選"),
                                      m.index("營收爆發觀察"), m.index("參考"))
    assert i_exit < i_cand < i_watch < i_ref
    cand = m[i_cand:i_watch]
    assert "2451" not in cand                      # 🚨 出場的不再列為進場候選
    assert "[S4+S7⭐]" in cand and cand.count("1513") == 1
    watch = m[i_watch:i_ref]
    assert watch.index("電腦及週邊設備業") < watch.index("電子零組件業")   # 檔數多的族群在前
    assert watch.index("🆕2305") < watch.index("6933")                    # 族群內 🆕 優先
    assert "+1,655張" in watch and "觸發後+7%" in watch
    assert "停損" not in m                         # 規則不再每天重印


def test_empty_day_is_one_line():
    msgs = tg.format_signals({"_meta": pd.DataFrame([{"regime_label": "🟡 中性", "regime_60d_return": 0.0}])},
                             "2026-10-01")
    assert len(msgs) == 1 and "今日無新訊號" in msgs[0]


def test_pack_splits_long_blocks_on_line_boundaries():
    block = "\n".join(f"line {i} " + "x" * 90 for i in range(100))
    msgs = tg._pack(["head", block])
    assert len(msgs) > 1 and all(len(m) <= tg.TG_MAX_LEN for m in msgs)
    assert "\n".join(msgs).replace("\n\n", "\n").count("line ") == 100


def test_rules_message_reads_numbers_from_strategies():
    from technical.signals import STRATEGIES
    txt = tg.rules_message()
    s7 = next(s for s in STRATEGIES if s["timeframe"] == "accum")
    s5 = next(s for s in STRATEGIES if s["timeframe"] == "revenue")
    assert f"停損 −{s7['default_sl']:.0%}" in txt
    assert f"最長 {s5['default_hold']} 天" in txt                  # 無 trailing → 天數上限生效
    assert f"最長 {s7['default_hold']} 天" not in txt and "無天數上限" in txt   # trailing → 引擎不檢查天數
    for tag in ("[S4]", "[S5]", "[S6]", "[S7]"):
        assert tag in txt


# ─── 2026-10-04：候選名單＝追蹤名單、價格精度、出場合併、報告日 ─────────────────

def test_finalize_drops_low_aqs_caps_s5_and_total(monkeypatch):
    import screener.candidates as cand
    monkeypatch.setattr(cand, "_mom20", lambda sid, date: -int(sid) / 1000)  # 代號小 = 動能強
    rev = pd.DataFrame([_row(f"{i:04d}", aqs_score=60) for i in range(1, 31)])     # S5 一次 30 檔
    acc = pd.DataFrame([_row("0001", aqs_score=40), _row("0100", aqs_score=80)])
    out = cand.finalize_candidates({"revenue": rev, "accum": acc}, "2026-10-01")
    assert out["revenue"]["stock_id"].tolist() == [f"{i:04d}" for i in range(1, 11)]   # S5 單日上限 10
    assert "0001" not in set(out["accum"]["stock_id"])                                 # AQS 40 剔除
    ids = set(out["revenue"]["stock_id"]) | set(out["accum"]["stock_id"])
    assert len(ids) <= cand.MAX_CANDIDATES


def test_finalize_keeps_rows_without_aqs(monkeypatch):
    import screener.candidates as cand
    monkeypatch.setattr(cand, "_mom20", lambda sid, date: 0.0)
    out = cand.finalize_candidates({"long": pd.DataFrame([_row("2451")])}, "2026-10-01")
    assert out["long"]["stock_id"].tolist() == ["2451"]


@pytest.mark.parametrize("v,expected", [(24.85, "24.85"), (24.9, "24.90"), (132.5, "132.5"),
                                        (3440, "3,440"), (float("nan"), "—")])
def test_px_tick_aware(v, expected):
    assert tg._px(v) == expected


def test_exits_merged_per_stock():
    ex = pd.DataFrame([
        {"level": "⚠️ 注意", "stock_id": "3017", "name": "奇鋐", "strategy": "S6", "entry_date": "2026-08-06",
         "entry_price": 2940.0, "close": 3440.0, "pnl_pct": 17.0, "reason": "近5日法人轉賣 1,225 張"},
        {"level": "⚠️ 注意", "stock_id": "3017", "name": "奇鋐", "strategy": "S7", "entry_date": "2026-09-15",
         "entry_price": 3115.0, "close": 3440.0, "pnl_pct": 10.4, "reason": "近5日法人轉賣 1,225 張"},
    ])
    m = tg.format_signals({"exits": ex}, "2026-10-02")[0]
    assert m.count("3017") == 1 and "[S6+S7]" in m and "要處理</b>（1）" in m
    assert m.count("近5日法人轉賣") == 1


def test_report_date_uses_trade_date():
    sig = {"_meta": pd.DataFrame([{"trade_date": "2026-10-02"}])}
    assert tg.report_date(sig) == "2026-10-02"
    assert "10/02（五）" in tg.format_signals(sig, tg.report_date(sig))[0]


def test_radar_section_rendered_between_candidates_and_watch():
    radar = pd.DataFrame([{"stock_id": "2379", "industry": "半導體業", "close": 760.0, "trust_days": 4,
                           "trust_20d": 2_000_000.0, "trust_20d_pct_shares": 0.55, "since_start_pct": 0.66,
                           "ex20_pct": -0.05, "dist52_pct": 15.6, "rev_yoy": 28.1, "retail_wchg": -0.25}])
    watch = pd.DataFrame([{"stock_id": "6933", "industry": "電腦及週邊設備業", "close": 371.0, "since_pct": 7.0,
                           "is_new": False, "burst_yoy": 191.0, "burst_g3": 108.0, "f_60d": 1_655_000.0}])
    m = tg.format_signals({"long": pd.DataFrame([_row("2451")]), "radar": radar, "watch": watch},
                          "2026-10-02")[0]
    assert m.index("新進場候選") < m.index("法人佈局雷達") < m.index("營收爆發觀察")
    assert "投信連買4天" in m and "0.55%股本" in m and "作帳" in m      # 10 月 → 作帳期提醒


def test_radar_scorecard_dedups_repeat_listings():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "rs", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "radar_scorecard.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    days = pd.bdate_range("2026-10-06", periods=40)
    log = pd.DataFrame({"date": [days[0], days[1], days[5], days[25]], "stock_id": ["A", "A", "B", "A"]})
    ep = rs.episodes(log, days)
    assert list(zip(ep["stock_id"], ep["date"])) == [("A", days[0]), ("B", days[5]), ("A", days[25])]
