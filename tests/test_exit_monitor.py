"""出場監控 classify 純函式測試（資金「持續」撤離才出場）。"""
from notify.exit_monitor import classify


def aqs(score=70, dim4=10, dim1=14, stage="🟡 中期"):
    return {"score": score, "dim4_inst_price_align": dim4,
            "dim1_volprice": dim1, "stage": stage}


def test_sustained_distribution_to_retail_exits():
    # 大幅賣超 + 賣超天數足(8) + 散戶接手 → 🚨
    lvl, r = classify(aqs(), foreign_10d=-3_000_000, foreign_selldays=8,
                      inst_5d=0, retail_rising=True)
    assert lvl == "🚨 出場" and "持續撤離" in r[0]


def test_sustained_distribution_redflag_dim4_exits():
    lvl, _ = classify(aqs(dim4=-20), foreign_10d=-2_000_000, foreign_selldays=7,
                      inst_5d=0, retail_rising=False)
    assert lvl == "🚨 出場"


def test_single_day_sell_not_sustained_holds():
    # 2301 情境：10日加總大賣，但是單一爆量日(賣超天數僅 2) → 不算持續撤離
    lvl, _ = classify(aqs(), foreign_10d=-12_000_000, foreign_selldays=2,
                      inst_5d=0, retail_rising=True)
    assert lvl != "🚨 出場"


def test_sustained_sell_without_retail_or_redflag_holds():
    # 持續賣超但散戶沒接手、dim4 正常 → 不算倒貨
    lvl, _ = classify(aqs(dim4=10), foreign_10d=-3_000_000, foreign_selldays=8,
                      inst_5d=0, retail_rising=False)
    assert lvl != "🚨 出場"


def test_distribution_stage_exits():
    lvl, _ = classify(aqs(stage="⚫ 派發中段"), 0, 0, 0, False)
    assert lvl == "🚨 出場"


def test_trap_low_score_negative_dim4_exits():
    lvl, _ = classify(aqs(score=45, dim4=-10, stage="🔴 末段"), 0, 0, 0, False)
    assert lvl == "🚨 出場"


def test_weak_buying_power_warns():
    lvl, r = classify(aqs(dim1=5), 0, 0, 0, False)
    assert lvl == "⚠️ 注意" and "買力" in r[0]


def test_inst_5d_sell_threshold():
    assert classify(aqs(), 0, 0, -1_000_000, False)[0] == "⚠️ 注意"   # 大賣超 → ⚠️
    assert classify(aqs(), 0, 0, -100_000, False)[0] == "✅ 持有"      # 零星 → 不觸發


def test_late_stage_alone_does_not_warn():
    lvl, _ = classify(aqs(stage="🔴 末段", dim1=14), 0, 0, 0, False)
    assert lvl == "✅ 持有"


def test_healthy_holds():
    lvl, r = classify(aqs(78, 15, 14, "🟢 早期累積"), 800_000, 0, 300_000, False)
    assert lvl == "✅ 持有" and r == []


def test_none_aqs_does_not_crash():
    assert classify(None, None, None, None, False)[0] == "✅ 持有"


def test_exit_dedup_removes_stock_from_entry_section():
    """同一檔出現在出場區時，不該又出現在進場區（修義隆「又買又賣」bug）。"""
    import pandas as pd
    from notify.telegram_bot import format_signals
    long_df = pd.DataFrame([{"stock_id": "2458", "close": 178.5, "f_60d": 1_000_000,
                             "t_60d": 0, "market": "TWSE", "aqs_score": 81,
                             "aqs_stage": "🔴 末段"}])
    exits = pd.DataFrame([{"level": "🚨 出場", "stock_id": "2458", "name": "義隆",
                           "strategy": "S7", "entry_date": "2026-06-10",
                           "entry_price": 150.0, "close": 178.5, "pnl_pct": 19.0,
                           "reason": "派發"}])
    meta = pd.DataFrame([{"regime_label": "多頭", "regime_60d_return": 0.1}])
    msgs = format_signals({"long": long_df, "exits": exits, "_meta": meta}, "2026-06-22")
    m = "\n".join(msgs)
    assert "新進場候選" not in m                          # 唯一候選被 🚨 拿掉 → 無進場區
    assert "2458" in m[m.index("要處理"):]               # 出場區有 2458


def test_warn_does_not_suppress_entry():
    """⚠️ 注意 是 heads-up，不把股票從進場區拿掉（避免太嚴格、誤殺健康新訊號）。"""
    import pandas as pd
    from notify.telegram_bot import format_signals
    long_df = pd.DataFrame([{"stock_id": "2458", "close": 178.5, "f_60d": 1_000_000,
                             "t_60d": 0, "market": "TWSE", "aqs_score": 81,
                             "aqs_stage": "🟡 中期"}])
    exits = pd.DataFrame([{"level": "⚠️ 注意", "stock_id": "2458", "name": "義隆",
                           "strategy": "S7", "entry_date": "2026-06-10",
                           "entry_price": 150.0, "close": 178.5, "pnl_pct": 19.0,
                           "reason": "買力轉弱"}])
    meta = pd.DataFrame([{"regime_label": "多頭", "regime_60d_return": 0.1}])
    msgs = format_signals({"long": long_df, "exits": exits, "_meta": meta}, "2026-06-22")
    m = "\n".join(msgs)
    assert "2458" in m[m.index("新進場候選"):]     # ⚠️ 不 suppress，進場區仍有


def test_large_cap_small_ratio_not_sustained():
    """大型股絕對賣超過 floor，但佔量比很小 → 不算持續撤離（修絕對門檻誤殺大型股）。"""
    lvl, _ = classify(aqs(), foreign_10d=-5_000_000, foreign_selldays=7,
                      inst_5d=0, retail_rising=True, foreign_10d_ratio=-0.008)
    assert lvl != "🚨 出場"


def test_sustained_with_heavy_ratio_exits():
    """同樣絕對賣超，但佔量比夠大（≤ -5%）→ 真持續撤離。"""
    lvl, _ = classify(aqs(), foreign_10d=-5_000_000, foreign_selldays=7,
                      inst_5d=0, retail_rising=True, foreign_10d_ratio=-0.09)
    assert lvl == "🚨 出場"


def test_retail_rising_recent_filters_noise():
    """散戶比例近月 +0.17pp 是噪音不算接手；+1.2pp 才算；資料不足回 False。"""
    import pandas as pd
    from notify.exit_monitor import retail_rising_recent
    d = pd.date_range("2026-05-01", periods=5, freq="W")
    noise = pd.DataFrame({"date": d, "retail_pct": [8.26, 8.28, 8.25, 8.45, 8.43]})
    real = pd.DataFrame({"date": d, "retail_pct": [8.0, 8.3, 8.6, 8.9, 9.2]})
    assert retail_rising_recent(noise) is False
    assert retail_rising_recent(real) is True
    assert retail_rising_recent(pd.DataFrame()) is False


def test_prune_untracked_keeps_top_momentum(monkeypatch):
    import pandas as pd
    import notify.exit_monitor as em
    import notify.telegram_bot as tg
    rows = [{"entry_date": "2026-10-01", "stock_id": f"{i:04d}", "name": "", "strategy": "S5",
             "entry_price": 10.0, "status": "open", "alert_level": "none", "exit_date": "",
             "exit_reason": "", "pnl_pct": ""} for i in range(1, 13)]
    rows.append({**rows[0], "stock_id": "9999", "strategy": "S7"})
    state = {"log": pd.DataFrame(rows)}
    monkeypatch.setattr(em, "_load", lambda: state["log"].copy())
    monkeypatch.setattr(em, "_save", lambda df: state.update(log=df))
    monkeypatch.setattr(tg, "_mom20", lambda sid, date: -int(sid))   # 代號小 = 動能強
    assert em.prune_untracked() == 2
    log = state["log"]
    dropped = set(log[log.status == "not_notified"]["stock_id"])
    assert dropped == {"0011", "0012"}
    assert em.prune_untracked() == 0                                   # 冪等


# ─── 出場＝策略原規則（2026-10 改版，用回測引擎判斷）──────────────────────────

def _bars(closes, start="2026-06-01", highs=None, lows=None):
    import pandas as pd
    d = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": d, "open": closes, "high": highs or closes,
                         "low": lows or closes, "close": closes, "volume": 1e6})


def test_rule_status_stop_loss_today_detected():
    from notify.exit_monitor import rule_status
    # S4：停損 10%。訊號第 15 根，隔天 100 進場，最後一天（今天）跌到 85 → 今天就要報
    closes = [100.0] * 15 + [100.0, 99.0, 98.0, 85.0]
    px = _bars(closes)
    r = rule_status("X", "S4", px["date"].iloc[14].strftime("%Y-%m-%d"), px=px)
    assert r["state"] == "exit" and r["reason"] == "stop_loss"
    assert abs(r["exit_price"] - r["entry_price"] * 0.9) < 1e-6
    assert r["exit_date"] == px["date"].iloc[-1]


def test_rule_status_trailing_activation_and_open():
    from notify.exit_monitor import rule_status
    closes = [100.0] * 15 + [100.0, 110.0, 125.0, 124.0]   # +25% ≥ S4 trail_trigger 20%
    px = _bars(closes)
    r = rule_status("X", "S4", px["date"].iloc[14].strftime("%Y-%m-%d"), px=px)
    assert r["state"] == "open" and r["trail_active"]
    assert abs(r["trail_level"] - 125.0 * 0.85) < 1e-6


def test_rule_status_signal_today_is_pending():
    from notify.exit_monitor import rule_status
    px = _bars([100.0] * 20)
    assert rule_status("X", "S4", px["date"].iloc[-1].strftime("%Y-%m-%d"), px=px)["state"] == "pending"


def test_rule_status_max_hold_exits_next_open():
    # S5 沒有 trailing → 最長持有天數生效（S4/S6/S7 的 trailing 分支在引擎裡跳過了天數檢查）
    from notify.exit_monitor import rule_status, _strategy_cfg
    hold = _strategy_cfg("S5")["default_hold"]
    n = hold // 7 * 5 + 30                                   # 足夠多交易日超過日曆天上限
    px = _bars([100.0] * 15 + [100.5] * n)
    r = rule_status("X", "S5", px["date"].iloc[14].strftime("%Y-%m-%d"), px=px)
    assert r["state"] == "exit" and r["reason"] == "max_hold"


def test_evaluate_reports_recent_exit_and_silences_old(monkeypatch):
    import pandas as pd
    import notify.exit_monitor as em
    px = _bars([100.0] * 15 + [100.0, 85.0] + [86.0] * 10)        # 第 17 根就停損，之後橫盤 10 天
    sig_day = px["date"].iloc[14].strftime("%Y-%m-%d")
    row = {"entry_date": sig_day, "stock_id": "X", "name": "", "strategy": "S4", "entry_price": 100.0,
           "status": "open", "alert_level": "none", "exit_date": "", "exit_reason": "", "pnl_pct": ""}
    state = {"log": pd.DataFrame([row, {**row, "stock_id": "Y"}])}
    monkeypatch.setattr(em, "_load", lambda: state["log"].astype(object))   # 同正式 _load
    monkeypatch.setattr(em, "_save", lambda df: state.update(log=df))
    monkeypatch.setattr(em, "load_prices", lambda sid, start="": px)
    monkeypatch.setattr(em, "_chip_note", lambda sid: "")
    today = px["date"].iloc[-1].strftime("%Y-%m-%d")
    out = em.evaluate(today)                                       # 停損在 10 個交易日前 → 靜默
    assert out.empty and (state["log"].status == "exited").all()
    state["log"] = pd.DataFrame([row])
    out = em.evaluate(px["date"].iloc[16].strftime("%Y-%m-%d"))   # 停損當天評估 → 要報
    assert len(out) == 1 and out.iloc[0]["level"] == "🚨 出場" and "停損" in out.iloc[0]["reason"]
