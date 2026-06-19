"""
本地 SQLite 快取：一次性全量下載，之後每天只增量更新。
所有資料落地後從這裡讀，不重複打 API。
"""
import sqlite3
import logging
from datetime import date as _date
import pandas as pd
from config import DB_PATH

logger = logging.getLogger(__name__)

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _insert_or_ignore(table, conn, keys, data_iter):
    """pandas to_sql 自定義 method：對應 SQLite INSERT OR IGNORE"""
    data = list(data_iter)
    if not data:
        return
    placeholders = ", ".join(["?" for _ in keys])
    cols = ", ".join(keys)
    stmt = f"INSERT OR IGNORE INTO {table.name} ({cols}) VALUES ({placeholders})"
    conn.executemany(stmt, data)


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """建立所有資料表（幂等）"""
    ddl = """
    CREATE TABLE IF NOT EXISTS stock_universe (
        stock_id   TEXT NOT NULL,
        stock_name TEXT,
        market     TEXT,
        industry   TEXT,
        updated_at TEXT,
        PRIMARY KEY (stock_id)
    );

    CREATE TABLE IF NOT EXISTS daily_price (
        stock_id TEXT NOT NULL,
        date     TEXT NOT NULL,
        open     REAL,
        high     REAL,
        low      REAL,
        close    REAL,
        volume   REAL,
        PRIMARY KEY (stock_id, date)
    );

    CREATE TABLE IF NOT EXISTS institutional (
        stock_id TEXT NOT NULL,
        date     TEXT NOT NULL,
        foreign_ REAL,
        trust    REAL,
        dealer   REAL,
        PRIMARY KEY (stock_id, date)
    );

    CREATE TABLE IF NOT EXISTS financial (
        stock_id TEXT NOT NULL,
        date     TEXT NOT NULL,
        type     TEXT,
        value    REAL,
        PRIMARY KEY (stock_id, date, type)
    );

    CREATE TABLE IF NOT EXISTS monthly_revenue (
        stock_id     TEXT NOT NULL,
        date         TEXT NOT NULL,
        revenue      REAL,
        revenue_yoy  REAL,
        fetched_date TEXT,
        PRIMARY KEY (stock_id, date)
    );

    CREATE TABLE IF NOT EXISTS daily_per (
        stock_id  TEXT NOT NULL,
        date      TEXT NOT NULL,
        per       REAL,
        pbr       REAL,
        div_yield REAL,
        PRIMARY KEY (stock_id, date)
    );

    CREATE TABLE IF NOT EXISTS fetch_log (
        stock_id   TEXT NOT NULL,
        dataset    TEXT NOT NULL,
        last_date  TEXT,
        PRIMARY KEY (stock_id, dataset)
    );

    CREATE TABLE IF NOT EXISTS shareholding (
        stock_id          TEXT NOT NULL,
        date              TEXT NOT NULL,  -- 週報日期 YYYY-MM-DD
        large_holder_pct  REAL,           -- 千張大戶比例 (持股 ≥1000張 = level 15-16)
        mid_holder_pct    REAL,           -- 中戶比例 (持股 200-1000張 = level 11-14)
        retail_pct        REAL,           -- 散戶比例 (持股 <50張 = level 1-8)
        total_shares      REAL,           -- 集保庫存總股數
        PRIMARY KEY (stock_id, date)
    );

    CREATE TABLE IF NOT EXISTS futures_inst (
        futures_id  TEXT NOT NULL,        -- 商品代號 (TX = 台指期)
        date        TEXT NOT NULL,
        institution TEXT NOT NULL,        -- 外資/投信/自營商
        long_oi     REAL,                 -- 多方未平倉口數
        short_oi    REAL,                 -- 空方未平倉口數
        net_oi      REAL,                 -- 淨多單 = long - short（負數=淨空）
        PRIMARY KEY (futures_id, date, institution)
    );
    """
    with _conn() as con:
        con.executescript(ddl)
        # migration: add fetched_date to existing DBs (idempotent)
        try:
            con.execute("ALTER TABLE monthly_revenue ADD COLUMN fetched_date TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
    logger.info(f"DB initialised at {DB_PATH}")
    _cleanup_mislabeled_skip()


def _cleanup_mislabeled_skip() -> None:
    """每次啟動時清理被誤標 9999-12-31 的 fetch_log 記錄。

    歷史背景：2026-04-30 ~ 05-01 期間 FinMind 可能系統異常單日大量回 403，
    舊的 mark_fetch_skip 把 1063 支股票（含台積電、南亞科、樺漢、聯發科）
    永久標記為跳過，導致它們的 5 月-6 月資料完全沒抓到，所有訊號與健診失準。

    清理規則：
    - 若資料表（daily_price 等）裡仍有該股票歷史資料 → 把 fetch_log 改回
      最後一筆實際資料日，讓 incremental_update 從那天起續抓
    - 若資料表完全沒資料 → 刪除 fetch_log entry，讓系統當作沒抓過
    """
    table_map = {
        "price": "daily_price",
        "institutional": "institutional",
        "revenue": "monthly_revenue",
        "per": "daily_per",
        # 財報三表共用同一張 financial 表；之前漏修，導致 1563 支股票財報
        # 永久被標 9999 → passes_filter=False → S4/S6/S7 silently skip
        "fin_stmt": "financial",
        "fin_bs":   "financial",
        "fin_cf":   "financial",
    }
    with _conn() as con:
        total_fixed = 0
        total_deleted = 0
        for dataset, table in table_map.items():
            cur = con.execute(
                f"UPDATE fetch_log SET last_date = "
                f"(SELECT MAX(date) FROM {table} WHERE {table}.stock_id=fetch_log.stock_id) "
                f"WHERE dataset=? AND last_date='9999-12-31' "
                f"AND EXISTS (SELECT 1 FROM {table} WHERE {table}.stock_id=fetch_log.stock_id)",
                (dataset,)
            )
            total_fixed += cur.rowcount
            cur = con.execute(
                "DELETE FROM fetch_log WHERE dataset=? AND last_date='9999-12-31'",
                (dataset,)
            )
            total_deleted += cur.rowcount
        if total_fixed or total_deleted:
            logger.warning(
                f"Cleaned mislabeled 9999 fetch_log: {total_fixed} reset to last actual date, "
                f"{total_deleted} entries deleted (will refetch from scratch)"
            )


# ─── stock_universe ───────────────────────────────────────────────────────────

def save_universe(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["updated_at"] = pd.Timestamp.now().isoformat()
    with _conn() as con:
        df.to_sql("stock_universe", con, if_exists="replace", index=False)


def load_universe(markets: list[str] | None = None) -> pd.DataFrame:
    q = "SELECT * FROM stock_universe"
    if markets:
        placeholders = ",".join("?" * len(markets))
        q += f" WHERE market IN ({placeholders})"
    with _conn() as con:
        df = pd.read_sql(q, con, params=markets or [])
    return df


# ─── daily_price ──────────────────────────────────────────────────────────────

def save_prices(stock_id: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["stock_id"] = stock_id
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    with _conn() as con:
        df.to_sql("daily_price", con, if_exists="append", index=False,
                  method=_insert_or_ignore)
        con.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?)",
            (stock_id, "price", df["date"].max())
        )


def save_prices_bulk(df: pd.DataFrame) -> None:
    """批次寫入多檔／多日價量（官方 bulk 來源）。

    df 欄位：stock_id, date, open, high, low, close, volume。
    用 INSERT OR IGNORE 避免覆蓋既有列；fetch_log 只在新日期較新時前進，
    不會把已較新的個股回退。
    """
    if df.empty:
        return
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    cols = ["stock_id", "date", "open", "high", "low", "close", "volume"]
    df = df[cols]
    maxd = df.groupby("stock_id")["date"].max()
    with _conn() as con:
        df.to_sql("daily_price", con, if_exists="append", index=False,
                  method=_insert_or_ignore)
        con.executemany(
            "INSERT INTO fetch_log(stock_id, dataset, last_date) VALUES (?, 'price', ?) "
            "ON CONFLICT(stock_id, dataset) DO UPDATE SET last_date=excluded.last_date "
            "WHERE excluded.last_date > fetch_log.last_date",
            list(maxd.items())
        )


def load_prices(stock_id: str, start: str = "2018-01-01", end: str = "") -> pd.DataFrame:
    q = "SELECT * FROM daily_price WHERE stock_id=? AND date>=?"
    params: list = [stock_id, start]
    if end:
        q += " AND date<=?"
        params.append(end)
    q += " ORDER BY date"
    with _conn() as con:
        df = pd.read_sql(q, con, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def last_price_date(stock_id: str) -> str | None:
    with _conn() as con:
        row = con.execute(
            "SELECT last_date FROM fetch_log WHERE stock_id=? AND dataset='price'",
            (stock_id,)
        ).fetchone()
    return row[0] if row else None


def earliest_last_date_since(dataset: str, cutoff: str) -> str | None:
    """最近還活躍（last_date >= cutoff）的個股中，最舊的 last_date。

    用來決定 bulk 補資料的起點：正常每日跑時 = 昨天（只補 1～2 日），
    積壓時 = 最落後個股的日期（一次補齊缺口）。完全落後（< cutoff）的
    個股不算進來，交給 FinMind 深歷史 fallback。
    """
    with _conn() as con:
        row = con.execute(
            "SELECT MIN(last_date) FROM fetch_log WHERE dataset=? AND last_date>=?",
            (dataset, cutoff)
        ).fetchone()
    return row[0] if row and row[0] else None


# ─── institutional ────────────────────────────────────────────────────────────

def save_institutional(stock_id: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["stock_id"] = stock_id
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    # 欄位名稱對應 DB schema (foreign_ 避免關鍵字衝突)
    if "foreign" in df.columns:
        df = df.rename(columns={"foreign": "foreign_"})
    with _conn() as con:
        df.to_sql("institutional", con, if_exists="append", index=False,
                  method=_insert_or_ignore)
        con.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?)",
            (stock_id, "institutional", df["date"].max())
        )


def save_institutional_bulk(df: pd.DataFrame) -> None:
    """批次寫入多檔／多日三大法人買賣超（官方 bulk 來源）。

    df 欄位：stock_id, date, foreign_, trust, dealer。語意同 save_prices_bulk。
    """
    if df.empty:
        return
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    cols = ["stock_id", "date", "foreign_", "trust", "dealer"]
    df = df[cols]
    maxd = df.groupby("stock_id")["date"].max()
    with _conn() as con:
        df.to_sql("institutional", con, if_exists="append", index=False,
                  method=_insert_or_ignore)
        con.executemany(
            "INSERT INTO fetch_log(stock_id, dataset, last_date) VALUES (?, 'institutional', ?) "
            "ON CONFLICT(stock_id, dataset) DO UPDATE SET last_date=excluded.last_date "
            "WHERE excluded.last_date > fetch_log.last_date",
            list(maxd.items())
        )


def last_institutional_date(stock_id: str) -> str | None:
    with _conn() as con:
        row = con.execute(
            "SELECT last_date FROM fetch_log WHERE stock_id=? AND dataset='institutional'",
            (stock_id,)
        ).fetchone()
    return row[0] if row else None


def load_institutional(stock_id: str, start: str = "2018-01-01") -> pd.DataFrame:
    with _conn() as con:
        df = pd.read_sql(
            "SELECT * FROM institutional WHERE stock_id=? AND date>=? ORDER BY date",
            con, params=[stock_id, start]
        )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ─── financial ────────────────────────────────────────────────────────────────

def save_financial(stock_id: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["stock_id"] = stock_id
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    with _conn() as con:
        df[["stock_id", "date", "type", "value"]].to_sql(
            "financial", con, if_exists="append", index=False, method=_insert_or_ignore
        )


def load_financial(stock_id: str, type_filter: list[str] | None = None) -> pd.DataFrame:
    q = "SELECT * FROM financial WHERE stock_id=?"
    params: list = [stock_id]
    if type_filter:
        placeholders = ",".join("?" * len(type_filter))
        q += f" AND type IN ({placeholders})"
        params.extend(type_filter)
    q += " ORDER BY date"
    with _conn() as con:
        df = pd.read_sql(q, con, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ─── monthly_revenue ──────────────────────────────────────────────────────────

def save_monthly_revenue(stock_id: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["stock_id"] = stock_id
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    # record today as the first-fetch date (INSERT OR IGNORE keeps original)
    df["fetched_date"] = str(_date.today())
    needed = ["stock_id", "date", "revenue", "revenue_yoy", "fetched_date"]
    existing = [c for c in needed if c in df.columns]
    with _conn() as con:
        df[existing].to_sql("monthly_revenue", con, if_exists="append",
                            index=False, method=_insert_or_ignore)
        con.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?)",
            (stock_id, "revenue", df["date"].max())
        )


def save_monthly_revenue_bulk(df: pd.DataFrame) -> None:
    """批次寫入多檔最新月營收（官方 MOPS bulk）。語意同 save_prices_bulk。

    df 欄位：stock_id, date, revenue, revenue_yoy。
    """
    if df.empty:
        return
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["fetched_date"] = str(_date.today())
    cols = ["stock_id", "date", "revenue", "revenue_yoy", "fetched_date"]
    df = df[cols]
    maxd = df.groupby("stock_id")["date"].max()
    with _conn() as con:
        df.to_sql("monthly_revenue", con, if_exists="append", index=False,
                  method=_insert_or_ignore)
        con.executemany(
            "INSERT INTO fetch_log(stock_id, dataset, last_date) VALUES (?, 'revenue', ?) "
            "ON CONFLICT(stock_id, dataset) DO UPDATE SET last_date=excluded.last_date "
            "WHERE excluded.last_date > fetch_log.last_date",
            list(maxd.items())
        )


_PROTECTED_STOCKS = {"0050"}  # 大盤代理：永遠不可標記為永久跳過


def mark_fetch_skip(stock_id: str, dataset: str) -> None:
    """記錄此股票／資料集為永久跳過（403）。

    安全防護：
    1. 0050 是市場過濾錨點，永不可跳過
    2. 若該股票最近 30 天內有過實際資料（不是真的下市），
       拒絕永久標記—很可能是 FinMind 偶發 403，不是真下市

    這個防護是 2026/04 - 05 期間誤標 1063 支股票（包含台積電、南亞科、
    樺漢等）為永久跳過後加上的：當時可能 FinMind 系統異常單日大量回 403，
    舊代碼直接全標 9999-12-31 導致這些股票被永久排除、不再更新。
    """
    if stock_id in _PROTECTED_STOCKS:
        logger.warning(f"Refusing to mark_fetch_skip protected stock {stock_id}/{dataset}")
        return

    # 檢查該股票最近是否仍有實際資料 — 若是，視為偶發 403，拒絕永久標記
    table_map = {"price": "daily_price", "institutional": "institutional",
                 "revenue": "monthly_revenue", "per": "daily_per",
                 "fin_stmt": "financial", "fin_bs": "financial", "fin_cf": "financial"}
    table = table_map.get(dataset)
    if table:
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=30)).isoformat()
        with _conn() as con:
            row = con.execute(
                f"SELECT MAX(date) FROM {table} WHERE stock_id=?",
                (stock_id,)
            ).fetchone()
        if row and row[0] and row[0] >= cutoff:
            logger.warning(
                f"Refusing mark_fetch_skip {stock_id}/{dataset}: "
                f"recent activity until {row[0]}, suspect transient 403 not delisting"
            )
            return

    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?)",
            (stock_id, dataset, "9999-12-31")
        )


def last_revenue_date(stock_id: str) -> str | None:
    with _conn() as con:
        row = con.execute(
            "SELECT last_date FROM fetch_log WHERE stock_id=? AND dataset='revenue'",
            (stock_id,)
        ).fetchone()
    return row[0] if row else None


def load_monthly_revenue(stock_id: str) -> pd.DataFrame:
    with _conn() as con:
        df = pd.read_sql(
            "SELECT * FROM monthly_revenue WHERE stock_id=? ORDER BY date",
            con, params=[stock_id]
        )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ─── daily_per ────────────────────────────────────────────────────────────────

def save_per(stock_id: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["stock_id"] = stock_id
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    with _conn() as con:
        df.to_sql("daily_per", con, if_exists="append", index=False,
                  method=_insert_or_ignore)
        con.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?)",
            (stock_id, "per", df["date"].max())
        )


def load_per(stock_id: str, start: str = "2018-01-01", end: str = "") -> pd.DataFrame:
    q = "SELECT * FROM daily_per WHERE stock_id=? AND date>=?"
    params: list = [stock_id, start]
    if end:
        q += " AND date<=?"
        params.append(end)
    q += " ORDER BY date"
    with _conn() as con:
        df = pd.read_sql(q, con, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def last_per_date(stock_id: str) -> str | None:
    with _conn() as con:
        row = con.execute(
            "SELECT last_date FROM fetch_log WHERE stock_id=? AND dataset='per'",
            (stock_id,)
        ).fetchone()
    return row[0] if row else None


# ─── shareholding (TDCC 週報) ─────────────────────────────────────────────────

def save_shareholding(df: pd.DataFrame) -> int:
    """bulk save：一次寫入整週全市場資料。
    df 欄位：stock_id, date, large_holder_pct, mid_holder_pct, retail_pct, total_shares
    """
    if df.empty:
        return 0
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    with _conn() as con:
        df.to_sql("shareholding", con, if_exists="append", index=False,
                  method=_insert_or_ignore)
        # fetch_log 用 stock_id="_GLOBAL" 記錄全市場 snapshot 日
        con.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?)",
            ("_GLOBAL", "shareholding", df["date"].max())
        )
    return len(df)


def load_shareholding_latest() -> pd.DataFrame:
    """讀取每支股票最新一筆 shareholding 資料"""
    q = """
    SELECT s.* FROM shareholding s
    JOIN (
        SELECT stock_id, MAX(date) AS max_date
        FROM shareholding GROUP BY stock_id
    ) m ON s.stock_id = m.stock_id AND s.date = m.max_date
    """
    with _conn() as con:
        df = pd.read_sql(q, con)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_shareholding(stock_id: str, start: str = "2018-01-01") -> pd.DataFrame:
    """讀取單股 shareholding 歷史"""
    with _conn() as con:
        df = pd.read_sql(
            "SELECT * FROM shareholding WHERE stock_id=? AND date>=? ORDER BY date",
            con, params=[stock_id, start]
        )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ─── 訊號出場監控狀態（存 DB 才能跨 CI run 持久化）──────────────────────────────

def load_open_signals() -> pd.DataFrame:
    """讀取出場監控的追蹤狀態（表不存在則回空）。"""
    with _conn() as con:
        try:
            df = pd.read_sql("SELECT * FROM open_signals", con)
        except Exception:
            return pd.DataFrame()
    if not df.empty and "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str)
    return df


def save_open_signals(df: pd.DataFrame) -> None:
    """整表覆寫追蹤狀態。"""
    with _conn() as con:
        df.to_sql("open_signals", con, if_exists="replace", index=False)


def last_shareholding_date() -> str | None:
    with _conn() as con:
        row = con.execute(
            "SELECT last_date FROM fetch_log WHERE stock_id='_GLOBAL' AND dataset='shareholding'"
        ).fetchone()
    return row[0] if row else None


# ─── futures_inst ─────────────────────────────────────────────────────────────

def save_futures_inst(futures_id: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["futures_id"] = futures_id
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    with _conn() as con:
        df.to_sql("futures_inst", con, if_exists="append", index=False,
                  method=_insert_or_ignore)
        con.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?)",
            (futures_id, "futures_inst", df["date"].max())
        )


def load_futures_inst(futures_id: str, institution: str = "外資",
                      start: str = "2018-01-01") -> pd.DataFrame:
    with _conn() as con:
        df = pd.read_sql(
            "SELECT date, long_oi, short_oi, net_oi FROM futures_inst "
            "WHERE futures_id=? AND institution=? AND date>=? ORDER BY date",
            con, params=[futures_id, institution, start]
        )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def last_futures_inst_date(futures_id: str) -> str | None:
    with _conn() as con:
        row = con.execute(
            "SELECT last_date FROM fetch_log WHERE stock_id=? AND dataset='futures_inst'",
            (futures_id,)
        ).fetchone()
    return row[0] if row and row[0] < "9999-01-01" else None
