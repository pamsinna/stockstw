"""
資料抓取層：整合 TWSE、TPEx 官方 API（免費無限制）+ FinMind（需 token）
TWSE/TPEx 處理日K和法人籌碼，FinMind 處理財報、月營收、興櫃特有資料
"""
import io
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
TWSE_FUND = "https://www.twse.com.tw/fund"          # 三大法人 T86
TPEX_WWW  = "https://www.tpex.org.tw/www/zh-tw"     # 上櫃新版 JSON API

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (research bot)"})


_PERM_SKIP = object()  # sentinel: 403 — permanent skip, no retry, no sleep


def _get(url: str, params: dict, retries: int = 3, delay: float = 1.0, timeout: int = 15):
    for i in range(retries):
        try:
            r = _session.get(url, params=params, timeout=timeout)
            # 403 = 已下市或無權限，永久跳過
            if r.status_code == 403:
                return _PERM_SKIP
            # 402 = 可能是暫時限流，回傳空讓下次重試（不寫 9999）
            if r.status_code == 402:
                return None
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
    頁面 colspan=7 的 section header 被 pandas 展開成所有格填同一值；
    用 col0 == col1 偵測，只保留「股票」區塊。
    """
    url = "https://isin.twse.com.tw/isin/C_public.jsp"
    resp = _session.get(url, params={"strMode": mode}, timeout=15)
    resp.encoding = "big5"
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0].copy()
    # 第 0 列是欄位名稱
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)

    col0 = df.columns[0]  # 有價證券代號及名稱
    col1 = df.columns[1]  # 國際證券辨識號碼(ISIN Code)
    col_industry = df.columns[4] if len(df.columns) > 4 else None

    # section header: pandas 把 colspan=7 的格展開，col0 == col1
    is_header = df[col0].astype(str) == df[col1].astype(str)

    section = ""
    sections = []
    for idx in df.index:
        if is_header.loc[idx]:
            section = str(df.loc[idx, col0]).strip()
        sections.append(section)
    df["_section"] = sections

    # 只留「股票」區塊的非標頭列
    df = df[~is_header & df["_section"].str.contains("股票", na=False)].copy()

    df[["stock_id", "stock_name"]] = df[col0].str.split("　", n=1, expand=True)
    df = df[df["stock_id"].str.match(r"^\d{4}$")].copy()
    df["market"] = market
    df["industry"] = df[col_industry].fillna("") if col_industry else ""
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
    None = 403 永久跳過，呼叫方應寫入 fetch_log 避免下次重試。
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
        logger.debug(f"FinMind {dataset} {stock_id}: 403 permanent skip")
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


def fetch_per(stock_id: str, start: str) -> pd.DataFrame | None:
    """每日本益比、股價淨值比、殖利率。回傳 None 代表 402/403 永久跳過。"""
    df = _finmind("TaiwanStockPER", stock_id, start)
    if df is None:
        return None
    if df.empty:
        return df
    df = df.rename(columns={"PER": "per", "PBR": "pbr", "dividend_yield": "div_yield"})
    needed = ["date", "per", "pbr", "div_yield"]
    existing = [c for c in needed if c in df.columns]
    return df[existing].sort_values("date").reset_index(drop=True)


# ─── 官方 bulk：單一請求回傳全市場單日資料（免 token、無 600/hr 限流）─────────────
# FinMind 一檔一請求，1145 檔 × 6s 撐不過免費額度 → 每天只刷新 ~287 檔。
# TWSE/TPEx 官方一次回傳整個市場，每日只需個位數請求。值經比對與 FinMind 完全一致。

def _num(s) -> float | None:
    """清掉千分位逗號、處理 '--'/'---'/空白等佔位符。"""
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
    if s in ("", "--", "---", "X", "x", "N/A", "尚無成交價"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_twse_prices_by_date(date_iso: str) -> pd.DataFrame:
    """全上市個股單日 OHLCV（MI_INDEX）。非交易日／無資料回傳空 DataFrame。"""
    data = _get(f"{TWSE_BASE}/MI_INDEX",
                {"response": "json", "date": date_iso.replace("-", ""), "type": "ALLBUT0999"}, timeout=30)
    if not data or data is _PERM_SKIP or data.get("stat") != "OK":
        return pd.DataFrame()
    table = next((t for t in data.get("tables", [])
                  if "證券代號" in (t.get("fields") or []) and "收盤價" in t["fields"]), None)
    if not table or not table.get("data"):
        return pd.DataFrame()
    idx = {name: i for i, name in enumerate(table["fields"])}
    rows = [{
        "stock_id": r[idx["證券代號"]].strip(),
        "date": date_iso,
        "open": _num(r[idx["開盤價"]]), "high": _num(r[idx["最高價"]]),
        "low": _num(r[idx["最低價"]]), "close": _num(r[idx["收盤價"]]),
        "volume": _num(r[idx["成交股數"]]),
    } for r in table["data"]]
    return pd.DataFrame(rows).dropna(subset=["close"])


def fetch_tpex_prices_by_date(date_iso: str) -> pd.DataFrame:
    """全上櫃個股單日 OHLCV（新版 dailyQuotes）。非交易日回傳空 DataFrame。"""
    y, m, d = date_iso.split("-")
    data = _get(f"{TPEX_WWW}/afterTrading/dailyQuotes",
                {"date": f"{y}/{m}/{d}", "type": "EW", "response": "json"}, timeout=30)
    if not data or data is _PERM_SKIP or str(data.get("stat", "")).lower() != "ok":
        return pd.DataFrame()
    tables = data.get("tables") or []
    if not tables or not tables[0].get("data"):
        return pd.DataFrame()
    idx = {name: i for i, name in enumerate(tables[0]["fields"])}
    rows = [{
        "stock_id": r[idx["代號"]].strip(),
        "date": date_iso,
        "open": _num(r[idx["開盤"]]), "high": _num(r[idx["最高"]]),
        "low": _num(r[idx["最低"]]), "close": _num(r[idx["收盤"]]),
        "volume": _num(r[idx["成交股數"]]),
    } for r in tables[0]["data"]]
    return pd.DataFrame(rows).dropna(subset=["close"])


def fetch_twse_inst_by_date(date_iso: str) -> pd.DataFrame:
    """全上市三大法人買賣超（T86）。欄位語意與 FinMind 比對一致。"""
    data = _get(f"{TWSE_FUND}/T86",
                {"response": "json", "date": date_iso.replace("-", ""), "selectType": "ALLBUT0999"}, timeout=30)
    if not data or data is _PERM_SKIP or data.get("stat") != "OK":
        return pd.DataFrame()
    fields = data.get("fields") or []
    if "證券代號" not in fields:
        return pd.DataFrame()
    idx = {name: i for i, name in enumerate(fields)}
    fcol, tcol, dcol = ("外陸資買賣超股數(不含外資自營商)", "投信買賣超股數", "自營商買賣超股數")
    rows = [{
        "stock_id": r[idx["證券代號"]].strip(),
        "date": date_iso,
        "foreign_": _num(r[idx[fcol]]), "trust": _num(r[idx[tcol]]), "dealer": _num(r[idx[dcol]]),
    } for r in data.get("data", [])]
    return pd.DataFrame(rows)


def fetch_tpex_inst_by_date(date_iso: str) -> pd.DataFrame:
    """全上櫃三大法人買賣超（dailyTrade）。欄位以位置對應（已比對 DB 驗證）：
    外陸資(不含外資自營)=4、投信=13、自營商合計=22。"""
    y, m, d = date_iso.split("-")
    data = _get(f"{TPEX_WWW}/insti/dailyTrade",
                {"type": "Daily", "sect": "EW", "date": f"{y}/{m}/{d}", "response": "json"}, timeout=30)
    if not data or data is _PERM_SKIP or str(data.get("stat", "")).lower() != "ok":
        return pd.DataFrame()
    tables = data.get("tables") or []
    if not tables or not tables[0].get("data"):
        return pd.DataFrame()
    rows = [{
        "stock_id": r[0].strip(), "date": date_iso,
        "foreign_": _num(r[4]), "trust": _num(r[13]), "dealer": _num(r[22]),
    } for r in tables[0]["data"] if len(r) > 22]
    return pd.DataFrame(rows)


def fetch_all_prices_by_date(date_iso: str) -> pd.DataFrame:
    """TWSE + TPEx 單日全市場 OHLCV。"""
    parts = [fetch_twse_prices_by_date(date_iso), fetch_tpex_prices_by_date(date_iso)]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fetch_all_inst_by_date(date_iso: str) -> pd.DataFrame:
    """TWSE + TPEx 單日全市場三大法人。"""
    parts = [fetch_twse_inst_by_date(date_iso), fetch_tpex_inst_by_date(date_iso)]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ─── 月營收 bulk（MOPS opendata：上市 t187ap05_L + 上櫃 mopsfin_t187ap05_O）──────
TWSE_OPENAPI = "https://openapi.twse.com.tw/v1"
TPEX_OPENAPI = "https://www.tpex.org.tw/openapi/v1"


def _parse_revenue_opendata(data) -> pd.DataFrame:
    """把 MOPS opendata（list of dict）轉成 stock_id/date/revenue/revenue_yoy。

    對齊既有 DB（FinMind）慣例：
    - date 用「申報月」標記 = 營收月 + 1 個月（FinMind: 5月營收 → date 2026-06-01）。
      實測 opendata 資料年月 11505 的當月營收 == FinMind date 2026-06-01 完全相等。
    - 單位：當月營收仟元 → 元（×1000）。
    - YoY：直接取「去年同月增減(%)」（單月 bulk 無法自算 pct_change(12)）。
    """
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    rows = []
    for r in data:
        ym = str(r.get("資料年月", "")).strip()      # 民國 YYYMM, e.g. "11505"（115年5月）
        code = str(r.get("公司代號", "")).strip()
        if len(ym) != 5 or not code.isdigit():
            continue
        y, m = int(ym[:3]) + 1911, int(ym[3:])
        ry, rm = (y + 1, 1) if m == 12 else (y, m + 1)   # 申報月 = 營收月 + 1
        rev = _num(r.get("營業收入-當月營收"))         # 仟元
        rows.append({
            "stock_id": code,
            "date": f"{ry:04d}-{rm:02d}-01",
            "revenue": rev * 1000 if rev is not None else None,
            "revenue_yoy": _num(r.get("營業收入-去年同月增減(%)")),
        })
    return pd.DataFrame(rows)


def fetch_all_monthly_revenue() -> pd.DataFrame:
    """全市場最新月營收（上市+上櫃 MOPS opendata）。免 token、各一次請求。"""
    parts = []
    for url in (f"{TWSE_OPENAPI}/opendata/t187ap05_L",
                f"{TPEX_OPENAPI}/mopsfin_t187ap05_O"):
        df = _parse_revenue_opendata(_get(url, {}, timeout=30))
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ─── 美國信用壓力溫度（HY OAS，FRED 免費）──────────────────────────────────────
# 固收領先股市：信用利差「擴大」是 risk-off 的領先溫度。當提醒看、非機械避險。

def fetch_hy_oas() -> pd.Series:
    """ICE BofA US 高收益債利差 OAS（%）— FRED 免費 CSV。失敗回空 Series。"""
    try:
        r = _session.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": "BAMLH0A0HYM2"}, timeout=20)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text), na_values=["."])
        d.columns = ["date", "v"]
        d["date"] = pd.to_datetime(d["date"])
        return d.dropna().set_index("date")["v"]
    except Exception as e:
        logger.warning(f"fetch_hy_oas failed: {e}")
        return pd.Series(dtype=float)


def us_credit_stress_summary() -> str:
    """一行「美國信用壓力」溫度（含絕對分級 + 月內擴大警訊）。"""
    s = fetch_hy_oas()
    if s.empty:
        return ""
    cur = float(s.iloc[-1])
    if cur < 3.5:
        band = "🟢 偏低"
    elif cur < 5.0:
        band = "🟡 中性"
    elif cur < 7.0:
        band = "🟠 偏高，留意"
    else:
        band = "🔴 警戒，建議縮手"
    mo = s[s.index <= s.index[-1] - pd.Timedelta(days=30)]
    trend = ""
    if not mo.empty:
        chg = cur - float(mo.iloc[-1])
        if chg >= 0.5:
            trend = f"，月內擴大 +{chg:.1f}（⚠️ 警訊）"
        elif chg <= -0.5:
            trend = f"，月內收斂 {chg:.1f}"
    return f"🌡 美國信用壓力(HY OAS)：{cur:.2f}%  {band}{trend}"


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


def fetch_futures_inst(futures_id: str, start: str, end: str = "") -> pd.DataFrame | None:
    """期貨三大法人未平倉。回傳含三家法人（外資/投信/自營商）的長表。"""
    df = _finmind("TaiwanFuturesInstitutionalInvestors", futures_id, start, end)
    if df is None:
        return None
    if df.empty:
        return df

    long_oi = pd.to_numeric(df["long_open_interest_balance_volume"], errors="coerce")
    short_oi = pd.to_numeric(df["short_open_interest_balance_volume"], errors="coerce")
    out = pd.DataFrame({
        "date": df["date"],
        "institution": df["institutional_investors"],
        "long_oi": long_oi,
        "short_oi": short_oi,
        "net_oi": long_oi - short_oi,
    })
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "institution"]).reset_index(drop=True)


def rate_limit_sleep(n_stocks: int, req_per_stock: int = 3) -> None:
    """估算需要 sleep 多少秒以不超過 600 req/hr"""
    total = n_stocks * req_per_stock
    if total > 500:
        per_req = 3600 / 600  # 6 秒/請求
        logger.info(f"Throttling: {total} requests estimated, sleeping {per_req:.1f}s/req")


# ─── TDCC 集保結算所：千張大戶週報 ─────────────────────────────────────────────

TDCC_OPENDATA_URL = "https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5"
_TDCC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Referer": "https://www.tdcc.com.tw/portal/zh/smWeb/qryStock",
}


def fetch_tdcc_shareholding() -> pd.DataFrame:
    """下載 TDCC 最新一週集保戶股權分散表（全市場 bulk CSV）。

    持股分級 levels:
      1=1-999股、2-3=1k-10k、4-8=10k-50k、9-10=50k-200k、
      11=200k-400k、12-14=400k-1M、15=>1M(千張大戶)、16=備註、17=合計

    回傳每股一行：
      stock_id, date, large_holder_pct (15-16),
      mid_holder_pct (11-14), retail_pct (1-8), total_shares (level 17)
    """
    import io
    r = _session.get(TDCC_OPENDATA_URL, headers=_TDCC_HEADERS, timeout=60)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text), dtype={"證券代號": str})
    raw.columns = ["date", "stock_id", "level", "holders", "shares", "pct"]
    raw["stock_id"] = raw["stock_id"].str.strip()
    raw["date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")

    # 只保留 4 碼純數字（過濾 ETF/權證/特別股的 6 碼代號）
    raw = raw[raw["stock_id"].str.match(r"^\d{4}$")].copy()
    if raw.empty:
        logger.warning("TDCC shareholding: no 4-digit stocks in response")
        return pd.DataFrame()

    # 用 pivot 把每支股票的各 level pct 攤平
    pct_pivot = raw.pivot_table(index=["stock_id", "date"], columns="level",
                                values="pct", aggfunc="sum", fill_value=0)
    # total_shares 從 level 17 取（合計）
    total = raw[raw["level"] == 17].set_index(["stock_id", "date"])["shares"]

    def _pct_sum(levels: list[int]) -> pd.Series:
        cols = [lv for lv in levels if lv in pct_pivot.columns]
        return pct_pivot[cols].sum(axis=1) if cols else pd.Series(0.0, index=pct_pivot.index)

    out = pd.DataFrame({
        "large_holder_pct": _pct_sum([15, 16]),
        "mid_holder_pct":   _pct_sum([11, 12, 13, 14]),
        "retail_pct":       _pct_sum([1, 2, 3, 4, 5, 6, 7, 8]),
        "total_shares":     total.reindex(pct_pivot.index).fillna(0),
    }).reset_index()
    return out
