"""成績單統計檢定（analysis/significance）。"""
import numpy as np
import pytest

from analysis import significance as sg


def test_strong_positive_edge_is_significant():
    rng = np.random.default_rng(0)
    v = rng.normal(2.0, 5.0, 400)
    c = np.repeat(np.arange(80), 5)
    r = sg.verdict(v, c)
    assert r.p_value < 0.05 and r.ci_low > 0 and r.label.startswith("✅")


def test_zero_edge_rarely_significant():
    """沒有優勢時，被判「顯著」的比例應約 ≤ 5%（型一錯誤）。"""
    hits = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        r = sg.verdict(rng.normal(0.0, 5.0, 200), np.repeat(np.arange(40), 5), n_boot=1000)
        hits += r.p_value < 0.05
    assert hits / 200 <= 0.09


def test_clustering_is_more_conservative_than_iid():
    """同一週一起漲跌（共同衝擊）時，以週為群組的 p 值要比當獨立樣本保守。"""
    rng = np.random.default_rng(2)
    weeks = np.repeat(np.arange(30), 10)
    shock = rng.normal(0, 6, 30)[weeks]
    v = 1.0 + shock + rng.normal(0, 2, 300)
    clustered = sg.verdict(v, weeks).p_value
    iid = sg.verdict(v, np.arange(300)).p_value
    assert clustered > iid


def test_required_n_shrinks_with_bigger_edge():
    rng = np.random.default_rng(3)
    c = np.repeat(np.arange(40), 5)
    small = sg.required_n(rng.normal(0.5, 5, 200), c)
    big = sg.required_n(rng.normal(3.0, 5, 200), c)
    assert big < small
    assert sg.required_n(rng.normal(-1, 5, 200), c) is None     # 負期望再多樣本也沒用


def test_holm_correction():
    out = sg.holm({"A": 0.001, "B": 0.02, "C": 0.04})
    assert out == {"A": True, "B": True, "C": True}            # 0.001≤.0167, .02≤.025, .04≤.05
    out = sg.holm({"A": 0.01, "B": 0.03, "C": 0.04})
    assert out == {"A": True, "B": False, "C": False}          # .03 > .025 → 之後全部不顯著


def test_losing_streak_prob_matches_simulation():
    rng = np.random.default_rng(4)
    sims = rng.random((20000, 50)) > 0.35                          # True = 虧
    def has(row, k=8):
        run = 0
        for x in row:
            run = run + 1 if x else 0
            if run >= k:
                return True
        return False
    mc = np.mean([has(r) for r in sims[:4000]])
    assert sg.losing_streak_prob(50, 8, 0.35) == pytest.approx(mc, abs=0.03)


def test_welch_and_week_key():
    d, p = sg.welch_compare([3, 4, 5, 7], ["a", "a", "b", "b"], [0, 1, 0, 2], ["c", "c", "d", "d"])
    assert d > 0
    assert list(sg.week_key(["2026-10-07", "2026-10-09", "2026-10-12"])) == ["2026-10-05", "2026-10-05", "2026-10-12"]


def test_fmt_wording():
    rng = np.random.default_rng(5)
    c = np.repeat(np.arange(40), 5)
    pos = sg.fmt(sg.verdict(rng.normal(2, 5, 200), c))
    neg = sg.fmt(sg.verdict(rng.normal(-2, 5, 200), c))
    assert "純靠運氣" in pos and "需約" in pos
    assert "純靠運氣" not in neg and neg.startswith("❌")
