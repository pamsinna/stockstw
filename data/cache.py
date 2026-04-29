"""
本地 SQLite 快取：一次性全量下載，之後每天只增量更新。
所有資料落地後從這裡讀，不重複打 API。
"""
import sqlite3
import logging
import pandas as pd
from pathlib import Path
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
        PRIMARY KEY (stock_id, date)
    );

    CREATE TABLE IF NOT EXISTS fetch_log (
        stock_id   TEXT NOT NULL,
        dataset    TEXT NOT NULL,
        last_date  TEXT,
        PRIMARY KEY (stock_id, dataset)
    );
    """
    with _conn() as con:
        con.executescript(ddl)
    logger.info(f"DB initialised at {DB_PATH}")


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
    df["date"] = df["date"].astype(str)
    with _conn() as con:
        df.to_sql("daily_price", con, if_exists="append", index=False,
                  method=_insert_or_ignore)
        con.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?)",
            (stock_id, "price", df["date"].max())
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


# ─── institutional ────────────────────────────────────────────────────────────

def save_institutional(stock_id: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["stock_id"] = stock_id
    df["date"] = df["date"].astype(str)
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
    df["date"] = df["date"].astype(str)
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
    df["date"] = df["date"].astype(str)
    needed = ["stock_id", "date", "revenue", "revenue_yoy"]
    existing = [c for c in needed if c in df.columns]
    with _conn() as con:
        df[existing].to_sql("monthly_revenue", con, if_exists="append",
                            index=False, method=_insert_or_ignore)
        con.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?)",
            (stock_id, "revenue", df["date"].max())
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
