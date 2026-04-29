"""
基本面品質篩選：量化「定價權」和「業界前三」
所有邏輯基於財報數字，不依賴人工標籤
"""
import logging
import pandas as pd
import numpy as np
from data.cache import load_financial, load_monthly_revenue
from config import FUNDAMENTAL_FILTERS as F

logger = logging.getLogger(__name__)


# ─── 單股基本面計算 ───────────────────────────────────────────────────────────

def calc_fundamentals(stock_id: str) -> dict:
    """
    回傳該股票最新的基本面評分字典。
    鍵值：eps_ttm, roe, gross_margin, op_margin, ocf_ratio,
           eps_growth_q, revenue_growth_m, quality_score, passes_filter
    """
    fin = load_financial(stock_id)
    rev = load_monthly_revenue(stock_id)

    result = {
        "stock_id": stock_id,
        "eps_ttm": np.nan,
        "roe": np.nan,
        "gross_margin": np.nan,
        "op_margin": np.nan,
        "ocf_ratio": np.nan,
        "eps_growth_q": 0,
        "revenue_growth_m": 0,
        "quality_score": 0.0,
        "passes_filter": False,
    }

    if fin.empty:
        result["passes_filter"] = True  # 無財報資料時不過濾，讓技術面決定
        return result

    try:
        result.update(_calc_eps_roe(fin))
        result.update(_calc_margins(fin))
        result.update(_calc_ocf_ratio(fin))
        result.update(_calc_eps_growth(fin))
    except Exception as e:
        logger.debug(f"{stock_id} financial calc error: {e}")

    if not rev.empty:
        try:
            result.update(_calc_revenue_growth(rev))
        except Exception as e:
            logger.debug(f"{stock_id} revenue calc error: {e}")

    result["quality_score"] = _score(result)
    result["passes_filter"] = _passes(result)
    return result


def _calc_eps_roe(fin: pd.DataFrame) -> dict:
    """近四季 EPS 合計（TTM）和最新 ROE"""
    eps_df = fin[fin["type"] == "EPS"].copy()
    roe_df = fin[fin["type"] == "ROE"].copy()

    eps_ttm = np.nan
    if not eps_df.empty:
        # 取最新 4 筆季度 EPS
        eps_df = eps_df.sort_values("date")
        recent = eps_df.tail(4)
        eps_ttm = recent["value"].sum()

    roe = np.nan
    if not roe_df.empty:
        roe = roe_df.sort_values("date").iloc[-1]["value"]

    return {"eps_ttm": eps_ttm, "roe": roe}


def _calc_margins(fin: pd.DataFrame) -> dict:
    """毛利率（定價權代理）和營業利益率"""
    gp_df  = fin[fin["type"].isin(["GrossProfit", "毛利"])].copy()
    rev_df = fin[fin["type"].isin(["Revenue", "營業收入"])].copy()
    op_df  = fin[fin["type"].isin(["OperatingIncome", "營業利益"])].copy()

    gross_margin = np.nan
    op_margin    = np.nan

    if not gp_df.empty and not rev_df.empty:
        gp  = gp_df.sort_values("date").iloc[-1]["value"]
        rev = rev_df.sort_values("date").iloc[-1]["value"]
        if rev and rev != 0:
            gross_margin = gp / rev * 100

    if not op_df.empty and not rev_df.empty:
        op  = op_df.sort_values("date").iloc[-1]["value"]
        rev = rev_df.sort_values("date").iloc[-1]["value"]
        if rev and rev != 0:
            op_margin = op / rev * 100

    return {"gross_margin": gross_margin, "op_margin": op_margin}


def _calc_ocf_ratio(fin: pd.DataFrame) -> dict:
    """現金轉換率 = OCF / Net Income（> 0.6 代表盈餘品質佳）"""
    ocf_df = fin[fin["type"].isin(["OperatingCashFlow", "營業活動現金流量"])].copy()
    ni_df  = fin[fin["type"].isin(["NetIncome", "本期淨利"])].copy()

    ocf_ratio = np.nan
    if not ocf_df.empty and not ni_df.empty:
        ocf = ocf_df.sort_values("date").iloc[-1]["value"]
        ni  = ni_df.sort_values("date").iloc[-1]["value"]
        if ni and ni != 0:
            ocf_ratio = ocf / ni

    return {"ocf_ratio": ocf_ratio}


def _calc_eps_growth(fin: pd.DataFrame) -> dict:
    """EPS 連續成長季數（與去年同期比）"""
    eps_df = fin[fin["type"] == "EPS"].sort_values("date").copy()
    if len(eps_df) < 5:
        return {"eps_growth_q": 0}

    eps_df["yoy"] = eps_df["value"].diff(4)  # 與4季前比
    recent = eps_df.tail(4)
    growth_q = int((recent["yoy"] > 0).sum())
    return {"eps_growth_q": growth_q}


def _calc_revenue_growth(rev: pd.DataFrame) -> dict:
    """月營收連續年增正成長月數"""
    rev = rev.sort_values("date").copy()
    if "revenue_yoy" not in rev.columns or rev.empty:
        return {"revenue_growth_m": 0}

    recent = rev.tail(6)
    growth_m = int((recent["revenue_yoy"] > 0).sum())
    return {"revenue_growth_m": growth_m}


# ─── 評分 ─────────────────────────────────────────────────────────────────────

def _score(r: dict) -> float:
    """0～100 分的基本面品質分數"""
    score = 0.0

    # EPS > 1 (+15)
    if _ok(r["eps_ttm"]) and r["eps_ttm"] >= F["min_eps"]:
        score += 15
    # ROE > 12% (+20)
    if _ok(r["roe"]) and r["roe"] >= F["min_roe"]:
        score += 20
    # 毛利率 > 25%（定價權）(+25)
    if _ok(r["gross_margin"]) and r["gross_margin"] >= F["pricing_power_margin"]:
        score += 25
    elif _ok(r["gross_margin"]) and r["gross_margin"] >= F["min_gross_margin"]:
        score += 10
    # OCF 品質 (+15)
    if _ok(r["ocf_ratio"]) and r["ocf_ratio"] >= F["min_ocf_ratio"]:
        score += 15
    # EPS 連續成長 (最高 +15)
    score += min(r.get("eps_growth_q", 0) / 4 * 15, 15)
    # 月營收連續成長 (最高 +10)
    score += min(r.get("revenue_growth_m", 0) / 6 * 10, 10)

    return round(score, 1)


def _passes(r: dict) -> bool:
    """硬性門檻：必須全部通過才進選股池"""
    if _ok(r["eps_ttm"]) and r["eps_ttm"] < F["min_eps"]:
        return False
    if _ok(r["gross_margin"]) and r["gross_margin"] < F["min_gross_margin"]:
        return False
    if _ok(r["ocf_ratio"]) and r["ocf_ratio"] < 0:
        return False
    return True


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


# ─── 業界前 N 名（市值排名代理）────────────────────────────────────────────────

def rank_by_industry(universe: pd.DataFrame,
                     market_cap: pd.DataFrame,
                     top_n: int | None = None) -> pd.DataFrame:
    """
    universe: 有 stock_id, industry 欄位
    market_cap: 有 stock_id, mkt_cap 欄位
    回傳附加 industry_rank 欄位的 DataFrame
    """
    top_n = top_n or F["top_n_industry"]
    df = universe.merge(market_cap, on="stock_id", how="left")
    df["industry_rank"] = (
        df.groupby("industry")["mkt_cap"]
          .rank(ascending=False, method="min")
    )
    df["is_top_n_industry"] = df["industry_rank"] <= top_n
    return df


def batch_fundamentals(stock_ids: list[str]) -> pd.DataFrame:
    """批次計算所有股票的基本面，回傳 DataFrame"""
    rows = []
    for sid in stock_ids:
        rows.append(calc_fundamentals(sid))
    return pd.DataFrame(rows)
