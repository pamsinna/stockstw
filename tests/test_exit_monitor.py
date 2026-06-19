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
