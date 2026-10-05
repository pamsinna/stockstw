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


# ─── point-in-time：回測只能用當時已公告的財報（2026-10 修正前視偏差）──────────

import pytest  # noqa: E402

from fundamental.quality_filter import calc_fundamentals, financial_available_date, pit_mask  # noqa: E402


@pytest.mark.parametrize("q,avail", [("2026-03-31", "2026-05-15"), ("2026-06-30", "2026-08-14"),
                                     ("2026-09-30", "2026-11-14"), ("2025-12-31", "2026-03-31")])
def test_financial_available_date(q, avail):
    assert financial_available_date(pd.Timestamp(q)) == pd.Timestamp(avail)


def test_pit_mask_uses_last_known_and_false_before_first():
    tl = pd.Series({pd.Timestamp("2026-05-15"): True, pd.Timestamp("2026-08-14"): False})
    dates = pd.Series(pd.to_datetime(["2026-05-14", "2026-05-15", "2026-08-13", "2026-08-14"]))
    assert pit_mask(dates, tl).tolist() == [False, True, True, False]


def test_calc_fundamentals_asof_ignores_unpublished_quarters():
    q = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"]
    rows = [{"date": d, "type": "EPS", "value": v} for d, v in zip(q, [1, 1, 1, 1, 9])]
    fin = pd.DataFrame(rows)
    empty = pd.DataFrame(columns=["date", "revenue", "revenue_yoy"])
    before = calc_fundamentals("X", asof="2026-05-14", fin=fin, rev=empty)   # Q1 2026 尚未公告
    after = calc_fundamentals("X", asof="2026-05-15", fin=fin, rev=empty)
    assert before["eps_ttm"] == 4 and after["eps_ttm"] == 12
