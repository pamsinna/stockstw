"""基本面濾網：OCF 現金轉換率（TTM）。

FinMind 現金流量表是「年初至今累計」、損益表淨利是「單季」→ 兩者都要轉 TTM
才能相除。舊版欄名對不上，ocf_ratio 恆為 NaN（2026-10 修正）。
"""
import numpy as np
import pandas as pd

from fundamental.quality_filter import _calc_ocf_ratio, _ttm_from_ytd


def _ytd(pairs):
    return pd.Series({pd.Timestamp(d): v for d, v in pairs})


def test_ttm_from_ytd_mid_year():
    s = _ytd([("2025-06-30", 50), ("2025-12-31", 120), ("2026-06-30", 70)])
    assert _ttm_from_ytd(s) == 70 + 120 - 50


def test_ttm_from_ytd_year_end_is_annual():
    assert _ttm_from_ytd(_ytd([("2025-09-30", 80), ("2025-12-31", 120)])) == 120


def test_ttm_from_ytd_missing_history_is_nan():
    assert np.isnan(_ttm_from_ytd(_ytd([("2026-06-30", 70)])))


def _fin(ocf_pairs, ni_quarters):
    rows = [{"date": d, "type": "CashFlowsFromOperatingActivities", "value": v} for d, v in ocf_pairs]
    rows += [{"date": d, "type": "IncomeAfterTaxes", "value": v} for d, v in ni_quarters]
    return pd.DataFrame(rows)


NI4 = [("2025-09-30", 10), ("2025-12-31", 10), ("2026-03-31", 10), ("2026-06-30", 10)]


def test_ocf_ratio_uses_finmind_names_and_ttm():
    fin = _fin([("2025-06-30", 20), ("2025-12-31", 40), ("2026-06-30", 28)], NI4)
    # TTM OCF = 28 + 40 − 20 = 48；TTM NI = 40
    assert abs(_calc_ocf_ratio(fin)["ocf_ratio"] - 1.2) < 1e-9


def test_ocf_ratio_negative_cash_burn():
    # AMAX-KY 型：獲利但營業現金流連續為負 → 比率 < 0 → 硬門檻剔除
    fin = _fin([("2025-06-30", -900), ("2025-12-31", -1000), ("2026-06-30", -380)], NI4)
    assert _calc_ocf_ratio(fin)["ocf_ratio"] < 0


def test_ocf_ratio_loss_and_burn_marked_negative():
    fin = _fin([("2025-12-31", -5)], [(d, -1) for d, _ in NI4])
    assert _calc_ocf_ratio(fin)["ocf_ratio"] == -1.0
