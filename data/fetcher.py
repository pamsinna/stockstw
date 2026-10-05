"""
資料抓取層：只用官方免費來源（2026-10 起完全移除 FinMind）
- 證交所 TWSE／櫃買 TPEx：日 K、三大法人、本益比、上市櫃清單、個股歷史、除權息
- 公開資訊觀測站 MOPS：月營收、財報（損益／資產負債／現金流量彙總表）
- 期交所 TAIFEX：期貨三大法人未平倉；集保 TDCC：股權分散；FRED：美國信用利差
"""
import io
import time
import logging
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


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
    resp.encoding = "cp950"  # 不能用 big5：碁/堃 等擴充字會變亂碼（宏碁 → 宏��）
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
    """全市場最新月營收（上市+上櫃 MOPS opendata）。免 token、各一次請求。

    ⚠️ opendata 約每月 17 日才換成新月份（實測 8 月營收出表日 0917），比法定
    10 日公布晚一週以上 → 每日選股改用 fetch_mops_monthly_revenue（即時）。
    """
    parts = []
    for url in (f"{TWSE_OPENAPI}/opendata/t187ap05_L",
                f"{TPEX_OPENAPI}/mopsfin_t187ap05_O"):
        df = _parse_revenue_opendata(_get(url, {}, timeout=30))
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# MOPS 月營收彙總靜態頁：公司申報後即時更新，可指定任意歷史月份。
# sii=上市、otc=上櫃；_0=國內公司、_1=外國公司（KY 股在 _1，漏抓就沒有 KY）。
MOPS_REVENUE_URL = "https://mopsov.twse.com.tw/nas/t21/{mk}/t21sc03_{roc}_{m}_{k}.html"


def _parse_mops_revenue_html(html: str, label_date: str) -> pd.DataFrame:
    """解析 t21sc03 HTML（每個產業一張 11 欄表）→ stock_id/date/revenue/revenue_yoy。"""
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except (ValueError, ImportError):  # 無任何 <table>（該月尚無人申報／頁面不存在）
        return pd.DataFrame()
    rows = []
    for t in tables:
        if t.shape[1] != 11:
            continue
        for r in t.itertuples(index=False):
            code = str(r[0]).strip()
            if not code.isdigit():  # 「合計」列
                continue
            rev = _num(r[2])  # 仟元
            rows.append({
                "stock_id": code,
                "date": label_date,
                "revenue": rev * 1000 if rev is not None else None,
                "revenue_yoy": _num(r[6]),
            })
    return pd.DataFrame(rows).drop_duplicates("stock_id") if rows else pd.DataFrame()


def fetch_mops_monthly_revenue(year: int, month: int) -> pd.DataFrame:
    """全市場（上市+上櫃，含 KY）指定營收月份的月營收。4 個請求、免 token。

    date 沿用 DB 慣例「申報月」= 營收月 + 1（3 月營收 → date 04-01）。
    """
    ry, rm = (year + 1, 1) if month == 12 else (year, month + 1)
    label = f"{ry:04d}-{rm:02d}-01"
    parts = []
    for mk in ("sii", "otc"):
        for k in (0, 1):
            url = MOPS_REVENUE_URL.format(mk=mk, roc=year - 1911, m=month, k=k)
            try:
                r = _session.get(url, timeout=30)
                if r.status_code != 200:
                    continue
                r.encoding = "cp950"  # big5 解不出碁等擴充字
                df = _parse_mops_revenue_html(r.text, label)
            except Exception as e:
                logger.warning(f"MOPS revenue {url} failed: {e}")
                continue
            if not df.empty:
                parts.append(df)
            time.sleep(0.5)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates("stock_id")


# ─── 財報 bulk（MOPS 彙總報表：損益 t163sb04、資產負債 t163sb05、現金流量 t163sb20）──
# 每季上市(sii)／上櫃(otc)各一個請求就涵蓋全市場（含 KY）。取代 FinMind 逐檔抓。
# 損益表與現金流量表是「年初至今累計」；資產負債表是季底時點值。單位仟元（EPS 元）。
MOPS_FIN_URL = "https://mopsov.twse.com.tw/mops/web/ajax_{ep}"
# DB financial.type ← MOPS 欄名（不同產業表欄名略有不同，依序找第一個存在的）
_MOPS_FIN_COLS = {
    "t163sb04": {
        "Revenue": ("營業收入", "收入"),
        "GrossProfit": ("營業毛利（毛損）",),
        "OperatingIncome": ("營業利益（損失）",),
        "IncomeAfterTaxes": ("本期淨利（淨損）",),
        "EPS": ("基本每股盈餘（元）", "基本每股盈餘"),
    },
    "t163sb05": {"Equity": ("權益總計", "權益總額")},
    "t163sb20": {"CashFlowsFromOperatingActivities": ("營業活動之淨現金流入（流出）",)},
}


def _parse_mops_fin_html(html: str, ep: str) -> pd.DataFrame:
    """MOPS 彙總報表 HTML → 寬表 stock_id + DB type 欄（原始單位，未換算）。"""
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except (ValueError, ImportError):
        return pd.DataFrame()
    parts = []
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "公司 代號" not in cols and "公司代號" not in cols:
            continue
        t.columns = cols
        code_col = "公司 代號" if "公司 代號" in cols else "公司代號"
        out = pd.DataFrame({"stock_id": t[code_col].astype(str).str.strip()})
        for typ, names in _MOPS_FIN_COLS[ep].items():
            col = next((n for n in names if n in cols), None)
            if col is not None:
                out[typ] = [_num(v) for v in t[col]]
        parts.append(out)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    return df[df["stock_id"].str.fullmatch(r"\d{4}")].drop_duplicates("stock_id")


def fetch_mops_statement(ep: str, year: int, quarter: int) -> pd.DataFrame:
    """單一報表、單季、上市＋上櫃全市場（寬表，原始單位）。"""
    parts = []
    for typek in ("sii", "otc"):
        try:
            r = _session.post(MOPS_FIN_URL.format(ep=ep), timeout=60, data={
                "encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "isQuery": "Y",
                "TYPEK": typek, "year": str(year - 1911), "season": f"{quarter:02d}"})
            r.encoding = "utf-8"
            df = _parse_mops_fin_html(r.text, ep)
        except Exception as e:
            logger.warning(f"MOPS {ep} {typek} {year}Q{quarter} failed: {e}")
            continue
        if not df.empty:
            parts.append(df)
        time.sleep(1.0)   # MOPS 對頻繁請求敏感
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


_FLOW_TYPES = ("Revenue", "GrossProfit", "OperatingIncome", "IncomeAfterTaxes", "EPS")


def fetch_mops_financials(year: int, quarter: int) -> pd.DataFrame:
    """全市場單季財報 → DB 長表（stock_id, date, type, value），格式同既有 financial 表：
    - 損益類（營收／毛利／營益／稅後淨利／EPS）轉成「單季」：本季累計 − 上季累計（Q1 不用減）
    - 現金流量維持「年初至今累計」（quality_filter 自己轉 TTM）
    - 權益為季底時點值；金額 仟元 → 元
    """
    q_end = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}[quarter]
    date = f"{year}-{q_end}"
    inc = fetch_mops_statement("t163sb04", year, quarter)
    bal = fetch_mops_statement("t163sb05", year, quarter)
    cf = fetch_mops_statement("t163sb20", year, quarter)
    if inc.empty and bal.empty and cf.empty:
        return pd.DataFrame()
    if quarter > 1 and not inc.empty:
        prev = fetch_mops_statement("t163sb04", year, quarter - 1).set_index("stock_id")
        cur = inc.set_index("stock_id")
        common = cur.index.intersection(prev.index)
        flows = [c for c in _FLOW_TYPES if c in cur.columns and c in prev.columns]
        cur = cur.loc[common, flows] - prev.loc[common, flows]   # 缺上季累計的公司無法換算 → 不收
        inc = cur.reset_index()
    rows = []
    for df in (inc, bal, cf):
        if df.empty:
            continue
        long = df.melt(id_vars="stock_id", var_name="type", value_name="value").dropna(subset=["value"])
        long.loc[long["type"] != "EPS", "value"] *= 1000   # 仟元 → 元
        rows.append(long)
    out = pd.concat(rows, ignore_index=True)
    out["date"] = date
    return out[["stock_id", "date", "type", "value"]]


# ─── 期交所：期貨三大法人未平倉（取代 FinMind TaiwanFuturesInstitutionalInvestors）──
TAIFEX_INST_URL = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
_TAIFEX_COMMODITY = {"TX": "TXF"}
_TAIFEX_INST_NAME = {"外資及陸資": "外資", "投信": "投信", "自營商": "自營商"}


def fetch_taifex_futures_inst(futures_id: str, start: str, end: str = "") -> pd.DataFrame:
    """期貨三大法人未平倉（口數）。欄位同舊版：date, institution, long_oi, short_oi, net_oi。"""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        r = _session.post(TAIFEX_INST_URL, timeout=30, data={
            "queryStartDate": pd.Timestamp(start).strftime("%Y/%m/%d"),
            "queryEndDate": pd.Timestamp(end).strftime("%Y/%m/%d"),
            "commodityId": _TAIFEX_COMMODITY.get(futures_id, futures_id)})
        r.encoding = "big5"
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        logger.warning(f"TAIFEX futures inst {futures_id} failed: {e}")
        return pd.DataFrame()
    if df.empty or "身份別" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "date": pd.to_datetime(df["日期"]),
        "institution": df["身份別"].map(lambda x: _TAIFEX_INST_NAME.get(str(x).strip(), str(x).strip())),
        "long_oi": pd.to_numeric(df["多方未平倉口數"], errors="coerce"),
        "short_oi": pd.to_numeric(df["空方未平倉口數"], errors="coerce"),
    })
    out["net_oi"] = out["long_oi"] - out["short_oi"]
    return out.sort_values(["date", "institution"]).reset_index(drop=True)


# ─── 證交所：除權息計算結果（TWT49U，取代 FinMind TaiwanStockDividendResult）─────

def fetch_twse_ex_rights(stock_id: str, start: str, end: str = "") -> pd.DataFrame:
    """上市股票／ETF 除權息前收盤價與參考價（逐年查，回傳 date, before_price, after_price）。"""
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
    rows = []
    for y in range(pd.Timestamp(start).year, end_ts.year + 1):
        a = max(pd.Timestamp(start), pd.Timestamp(y, 1, 1))
        b = min(end_ts, pd.Timestamp(y, 12, 31))
        data = _get("https://www.twse.com.tw/rwd/zh/exRight/TWT49U",
                    {"response": "json", "startDate": a.strftime("%Y%m%d"), "endDate": b.strftime("%Y%m%d")},
                    timeout=30)
        if not data or data is _PERM_SKIP or data.get("stat") != "OK":
            continue
        idx = {n: i for i, n in enumerate(data.get("fields") or [])}
        for r in data.get("data", []):
            if r[idx["股票代號"]].strip() != stock_id:
                continue
            ymd = r[idx["資料日期"]].replace("年", "-").replace("月", "-").replace("日", "")
            yy, mm, dd = ymd.split("-")
            rows.append({"date": pd.Timestamp(int(yy) + 1911, int(mm), int(dd)),
                         "before_price": _num(r[idx["除權息前收盤價"]]),
                         "after_price": _num(r[idx["除權息參考價"]])})
        time.sleep(0.5)
    return pd.DataFrame(rows)


# ─── 個股歷史日 K（TWSE STOCK_DAY / TPEx tradingStock，取代 FinMind 逐檔深歷史）──

def _roc_date(s: str) -> str:
    y, m, d = s.strip().split("/")
    return f"{int(y) + 1911:04d}-{int(m):02d}-{int(d):02d}"


def fetch_stock_history_month(stock_id: str, market: str, year: int, month: int) -> pd.DataFrame:
    """單檔單月日 K。market: TWSE / TPEx。成交量單位：股。"""
    if market == "TPEx":
        data = _get(f"{TPEX_WWW}/afterTrading/tradingStock",
                    {"code": stock_id, "date": f"{year}/{month:02d}/01", "response": "json"}, timeout=30)
        if not data or data is _PERM_SKIP or str(data.get("stat", "")).lower() != "ok":
            return pd.DataFrame()
        t = (data.get("tables") or [{}])[0]
        idx = {n.replace(" ", ""): i for i, n in enumerate(t.get("fields") or [])}
        rows = [{"date": _roc_date(r[idx["日期"]]), "open": _num(r[idx["開盤"]]), "high": _num(r[idx["最高"]]),
                 "low": _num(r[idx["最低"]]), "close": _num(r[idx["收盤"]]),
                 "volume": (_num(r[idx["成交張數"]]) or 0) * 1000} for r in t.get("data") or []]
    else:
        data = _get(f"{TWSE_BASE}/STOCK_DAY",
                    {"response": "json", "date": f"{year}{month:02d}01", "stockNo": stock_id}, timeout=30)
        if not data or data is _PERM_SKIP or data.get("stat") != "OK":
            return pd.DataFrame()
        idx = {n: i for i, n in enumerate(data.get("fields") or [])}
        rows = [{"date": _roc_date(r[idx["日期"]]), "open": _num(r[idx["開盤價"]]), "high": _num(r[idx["最高價"]]),
                 "low": _num(r[idx["最低價"]]), "close": _num(r[idx["收盤價"]]),
                 "volume": _num(r[idx["成交股數"]])} for r in data.get("data") or []]
    df = pd.DataFrame(rows)
    return df.dropna(subset=["close"]) if not df.empty else df


def fetch_stock_history(stock_id: str, market: str, start: str, end: str = "") -> pd.DataFrame:
    """單檔 start～end 日 K（逐月查官方個股頁）。"""
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
    parts = []
    for m in pd.period_range(pd.Timestamp(start), end_ts, freq="M"):
        df = fetch_stock_history_month(stock_id, market, m.year, m.month)
        if not df.empty:
            parts.append(df)
        time.sleep(0.5)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df[(df["date"] >= pd.Timestamp(start).strftime("%Y-%m-%d")) & (df["date"] <= end_ts.strftime("%Y-%m-%d"))]
    return df.drop_duplicates("date").reset_index(drop=True)


# ─── 本益比 bulk（TWSE BWIBBU_d + TPEx peQryDate，可指定歷史日期）─────────────

def _per_num(s) -> float:
    """本益比：官方以 '-' / 'N/A' 表示虧損（無意義）→ 0，對齊 FinMind 慣例。
    不能留 NaN：訊號端把 NaN 視為「無資料放行」，虧損股會被誤放。"""
    v = _num(s)
    return v if v is not None else 0.0


def fetch_twse_per_by_date(date_iso: str) -> pd.DataFrame:
    """全上市單日本益比／股價淨值比／殖利率。非交易日回傳空 DataFrame。"""
    data = _get(f"{TWSE_BASE}/BWIBBU_d",
                {"response": "json", "date": date_iso.replace("-", ""), "selectType": "ALL"}, timeout=30)
    if not data or data is _PERM_SKIP or data.get("stat") != "OK":
        return pd.DataFrame()
    fields = data.get("fields") or []
    if "證券代號" not in fields:
        return pd.DataFrame()
    idx = {name: i for i, name in enumerate(fields)}
    rows = [{
        "stock_id": r[idx["證券代號"]].strip(), "date": date_iso,
        "per": _per_num(r[idx["本益比"]]), "pbr": _num(r[idx["股價淨值比"]]),
        "div_yield": _num(r[idx["殖利率(%)"]]),
    } for r in data.get("data", [])]
    return pd.DataFrame(rows)


def fetch_tpex_per_by_date(date_iso: str) -> pd.DataFrame:
    """全上櫃單日本益比／股價淨值比／殖利率。非交易日回傳空 DataFrame。"""
    y, m, d = date_iso.split("-")
    data = _get(f"{TPEX_WWW}/afterTrading/peQryDate",
                {"date": f"{y}/{m}/{d}", "response": "json"}, timeout=30)
    if not data or data is _PERM_SKIP or str(data.get("stat", "")).lower() != "ok":
        return pd.DataFrame()
    tables = data.get("tables") or []
    if not tables or not tables[0].get("data"):
        return pd.DataFrame()
    idx = {name: i for i, name in enumerate(tables[0]["fields"])}
    rows = [{
        "stock_id": r[idx["股票代號"]].strip(), "date": date_iso,
        "per": _per_num(r[idx["本益比"]]), "pbr": _num(r[idx["股價淨值比"]]),
        "div_yield": _num(r[idx["殖利率(%)"]]),
    } for r in tables[0]["data"]]
    return pd.DataFrame(rows)


def fetch_all_per_by_date(date_iso: str) -> pd.DataFrame:
    """TWSE + TPEx 單日全市場本益比。"""
    parts = [fetch_twse_per_by_date(date_iso), fetch_tpex_per_by_date(date_iso)]
    parts = [p for p in parts if not p.empty]
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
