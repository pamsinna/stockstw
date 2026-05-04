"""
資料抓取層：整合 TWSE、TPEx 官方 API（免費無限制）+ FinMind（需 token）
TWSE/TPEx 處理日K和法人籌碼，FinMind 處理財報、月營收、興櫃特有資料
"""
import os
import time
import logging
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
_RATE_LIMIT_SEC = 6.0  # FinMind free tier: 600 req/hr → 6 s/req

TWSE_BASE = "https://www.twse.com.tw/exchangeReport"
TPEX_BASE = "https://www.tpex.org.tw/web/stock"

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (research bot)"})


_PERM_SKIP = object()  # sentinel: 402/403 — permanent skip, no retry, no sleep


def _get(url: str, params: dict, retries: int = 3, delay: float = 1.0):
    for i in range(retries):
        try:
            r = _session.get(url, params=params, timeout=15)
            # 402/403 = 付費限制或已下市，永久跳過（不重試、不睡覺）
            if r.status_code in (402, 403):
                return _PERM_SKIP
            # 429 = rate limit，等久一點再試
            if r.status_code == 429:
                time.sleep(60)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"GET failed ({i+1}/{retries}): {url} — {e}")
            time.sleep(delay * (i + 1))
    return None


# ─── TWSE 官方 API ────────────────────────────────────────────────────────────

def _parse_isin_page(mode: str, market: str) -> pd.DataFrame:
    """
    TWSE ISIN 頁面解析器（上市 strMode=2 / 上櫃 strMode=4）。
    頁面用 colspan=7 的行當區塊標題（股票、ETF、受益憑證⋯），
    只保留「股票」區塊，過濾掉 ETF 和其他產品。
    """
    url = "https://isin.twse.com.tw/isin/C_public.jsp"
    resp = _session.get(url, params={"strMode": mode}, timeout=15)
    resp.encoding = "big5"
    tables = pd.read_html(resp.text)
    df = tables[0].copy()
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)

    col0 = df.columns[0]
    other_cols = df.columns[1:]

    # section 標頭列：其他欄全部 NaN（colspan=7 的行）
    is_header = df[other_cols].isna().all(axis=1)

    # 對每一列標記它屬於哪個區塊
    section = ""
    sections = []
    for idx in df.index:
        if is_header[idx]:
            section = str(df.loc[idx, col0]).strip()
        sections.append(section)
    df["_section"] = sections

    # 只留「股票」區塊的非標頭列
    df = df[~is_header & df["_section"].str.contains("股票", na=False)].copy()

    df[["stock_id", "stock_name"]] = df[col0].str.split("　", n=1, expand=True)
    df = df[df["stock_id"].str.match(r"^\d{4}$")].copy()
    df["market"] = market
    industry_col = df.columns[4] if len(df.columns) > 4 else None
    df["industry"] = df[industry_col] if industry_col else ""
    return df[["stock_id", "stock_name", "market", "industry"]].reset_index(drop=True)


def fetch_twse_stock_list() -> pd.DataFrame:
    """取得所有上市普通股清單（過濾 ETF、受益憑證等）"""
    try:
        return _parse_isin_page("2", "TWSE")
    except Exception as e:
        logger.error(f"fetch_twse_stock_list failed: {e}")
        return pd.DataFrame()


def fetch_tpex_stock_list() -> pd.DataFrame:
    """取得所有上櫃普通股清單（過濾 ETF、受益憑證等）"""
    try:
        return _parse_isin_page("4", "TPEx")
    except Exception as e:
        logger.error(f"fetch_tpex_stock_list failed: {e}")
        return pd.DataFrame()


# ─── FinMind API ──────────────────────────────────────────────────────────────

def _finmind(dataset: str, stock_id: str, start: str, end: str = "") -> pd.DataFrame | None:
    """
    回傳 DataFrame（有或沒有資料）或 None。
    None = 402/403 永久跳過，呼叫方應寫入 fetch_log 避免下次重試。
    """
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start,
        "token": FINMIND_TOKEN,
    }
    if end:
        params["end_date"] = end
    data = _get(FINMIND_URL, params)
    time.sleep(_RATE_LIMIT_SEC)  # 402/403 也要睡：請求已打出去，需遵守 rate limit
    if data is _PERM_SKIP:
        logger.debug(f"FinMind {dataset} {stock_id}: 402/403 permanent skip")
        return None
    if not data or data.get("status") != 200:
        logger.warning(f"FinMind {dataset} {stock_id}: {data.get('msg') if data else 'no response'}")
        return pd.DataFrame()
    df = pd.DataFrame(data.get("data", []))
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_price(stock_id: str, start: str, end: str = "") -> pd.DataFrame | None:
    """日K OHLCV。回傳 None 代表 402/403 永久跳過；空 DataFrame 代表暫無資料。"""
    df = _finmind("TaiwanStockPrice", stock_id, start, end)
    if df is None:
        return None
    if df.empty:
        return df
    cols = {"open": "open", "max": "high", "min": "low", "close": "close",
            "Trading_Volume": "volume", "date": "date"}
    df = df.rename(columns={k: v for k, v in cols.items() if k in df.columns})
    needed = ["date", "open", "high", "low", "close", "volume"]
    existing = [c for c in needed if c in df.columns]
    return df[existing].sort_values("date").reset_index(drop=True)


def fetch_institutional(stock_id: str, start: str, end: str = "") -> pd.DataFrame | None:
    """三大法人買賣超。回傳 None 代表 402/403 永久跳過。"""
    df = _finmind("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start, end)
    if df is None:
        return None
    if df.empty:
        return df

    # FinMind 回傳 buy / sell 分開欄位，買賣超 = buy - sell
    df["buy_sell"] = pd.to_numeric(df["buy"], errors="coerce") - pd.to_numeric(df["sell"], errors="coerce")

    # 把各法人類別 pivot 成欄位
    pivot = df.pivot_table(index="date", columns="name", values="buy_sell", aggfunc="sum")
    pivot.columns.name = None
    pivot = pivot.reset_index()

    # 外資 = Foreign_Investor，投信 = Investment_Trust，自營 = Dealer_self + Dealer_Hedging
    foreign = pivot.get("Foreign_Investor", 0)
    trust   = pivot.get("Investment_Trust", 0)
    dealer  = pivot.get("Dealer_self", 0) + pivot.get("Dealer_Hedging", 0)

    result = pd.DataFrame({
        "date":    pivot["date"],
        "foreign": foreign,
        "trust":   trust,
        "dealer":  dealer,
    })
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values("date").reset_index(drop=True)


def fetch_financial_statement(stock_id: str, start: str) -> pd.DataFrame:
    """綜合損益表（含毛利率、營業利益）"""
    return _finmind("TaiwanStockFinancialStatements", stock_id, start)


def fetch_balance_sheet(stock_id: str, start: str) -> pd.DataFrame:
    return _finmind("TaiwanStockBalanceSheet", stock_id, start)


def fetch_cash_flow(stock_id: str, start: str) -> pd.DataFrame:
    return _finmind("TaiwanStockCashFlowsStatement", stock_id, start)


def fetch_monthly_revenue(stock_id: str, start: str) -> pd.DataFrame:
    """月營收（用來判斷連續成長）"""
    return _finmind("TaiwanStockMonthRevenue", stock_id, start)


def fetch_stock_list_finmind() -> pd.DataFrame:
    """FinMind 股票清單（含興櫃）"""
    params = {"dataset": "TaiwanStockInfo", "token": FINMIND_TOKEN}
    data = _get(FINMIND_URL, params)
    if not data or data.get("status") != 200:
        return pd.DataFrame()
    df = pd.DataFrame(data.get("data", []))
    return df


def fetch_emerging_stock_list() -> pd.DataFrame:
    """興櫃股票清單（透過 FinMind TaiwanStockInfo 過濾）"""
    df = fetch_stock_list_finmind()
    if df.empty or "type" not in df.columns:
        return pd.DataFrame()
    emerging = df[df["type"].str.contains("興櫃", na=False)].copy()
    emerging["market"] = "Emerging"
    rename = {"stock_id": "stock_id", "stock_name": "stock_name", "industry_category": "industry"}
    emerging = emerging.rename(columns={k: v for k, v in rename.items() if k in emerging.columns})
    needed = ["stock_id", "stock_name", "market", "industry"]
    existing = [c for c in needed if c in emerging.columns]
    return emerging[existing].reset_index(drop=True)


def rate_limit_sleep(n_stocks: int, req_per_stock: int = 3) -> None:
    """估算需要 sleep 多少秒以不超過 600 req/hr"""
    total = n_stocks * req_per_stock
    if total > 500:
        per_req = 3600 / 600  # 6 秒/請求
        logger.info(f"Throttling: {total} requests estimated, sleeping {per_req:.1f}s/req")
