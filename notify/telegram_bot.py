"""
Telegram 通知：用同步 requests 發訊息，不需要 async（GitHub Actions 環境簡單用）
訊號格式：每個時間框架一則訊息，清楚列出股票代號、市場、關鍵指標
"""
import os
import logging
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
API_URL = f"https://api.telegram.org/bot{TOKEN}"

MARKET_EMOJI = {"TWSE": "🔵", "TPEx": "🟢", "Emerging": "🟡"}
TF_LABEL = {
    "short": "⚡ 短線（1-5天）",
    "swing": "📈 波段（1-4週）",
    "long":  "🏔 中長線（1-3月）",
}


def send_message(text: str) -> bool:
    if not TOKEN or not CHAT_ID:
        logger.warning("Telegram not configured (TOKEN or CHAT_ID missing)")
        return False
    try:
        r = requests.post(
            f"{API_URL}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


POSITION_SIZE_PCT  = 5    # 每筆建議部位：總資金 5%
MAX_POSITIONS      = 10   # 最多同時持有 10 檔
WIN_RATE_PAUSE_THR = 0.28 # 近 20 筆勝率低於此值 → 發出暫停警告

SIGNAL_LOG = "reports/signal_log.csv"  # 每日訊號歷史紀錄


def _load_recent_log(n: int = 20) -> pd.DataFrame:
    """讀取最近 n 筆歷史訊號（用於監控實際勝率）"""
    try:
        df = pd.read_csv(SIGNAL_LOG)
        df = df[df["result"].notna()]  # 只看已有結果的
        return df.tail(n)
    except Exception:
        return pd.DataFrame()


def _append_signal_log(long_df: pd.DataFrame, date: str) -> None:
    """把今日訊號寫入歷史紀錄（result 待日後人工填寫或自動追蹤）"""
    if long_df.empty:
        return
    rows = []
    for _, row in long_df.iterrows():
        rows.append({
            "date":     date,
            "stock_id": row["stock_id"],
            "close":    row.get("close", ""),
            "result":   "",   # 出場後填寫：win / loss / hold
            "pnl_pct":  "",
        })
    new_df = pd.DataFrame(rows)
    try:
        existing = pd.read_csv(SIGNAL_LOG)
        pd.concat([existing, new_df], ignore_index=True).to_csv(SIGNAL_LOG, index=False)
    except FileNotFoundError:
        new_df.to_csv(SIGNAL_LOG, index=False)


def format_signals(signals: dict[str, pd.DataFrame], date: str) -> list[str]:
    """
    只主推策略四（中長線）。
    訊號超過 10 支時按成交量排序（流動性優先）。
    附近期績效監控與執行紀律提醒。
    """
    import os
    messages = []

    long_df    = signals.get("long",    pd.DataFrame())
    short_df   = signals.get("short",   pd.DataFrame())
    swing_df   = signals.get("swing",   pd.DataFrame())
    revenue_df = signals.get("revenue", pd.DataFrame())

    # 超過 10 支時按市值代理排序（收盤價 × 成交量 ≈ 當日成交金額）
    # vol_ratio 是策略一的條件，對策略四的 alpha 來源無關；
    # 成交金額最大的流動性最穩，避免大部位卡在小型股
    if not long_df.empty and len(long_df) > MAX_POSITIONS:
        if "volume" in long_df.columns and "close" in long_df.columns:
            long_df = long_df.assign(
                _turnover=long_df["close"] * long_df["volume"]
            ).sort_values("_turnover", ascending=False).drop(columns="_turnover")
        else:
            long_df = long_df.sample(frac=1, random_state=42)  # 隨機（統計最乾淨）

    # ── Header ────────────────────────────────────────────────────────────
    header = (
        f"📊 <b>台股選股報告 {date}</b>\n"
        f"主力訊號（中長線）：{len(long_df)} 支\n"
        f"建議部位：每筆 {POSITION_SIZE_PCT}%，最多同時持 {MAX_POSITIONS} 檔"
    )
    messages.append(header)

    # ── 近期績效監控 ──────────────────────────────────────────────────────
    recent = _load_recent_log(20)
    if len(recent) >= 10:
        wr = (recent["result"] == "win").mean()
        status = "🟢 正常" if wr >= WIN_RATE_PAUSE_THR else "🔴 警告：近期勝率偏低"
        perf_msg = (
            f"📉 <b>近 {len(recent)} 筆實盤勝率：{wr*100:.0f}%</b>  {status}\n"
            f"{'⚠️ 建議暫停並檢視策略是否失效' if wr < WIN_RATE_PAUSE_THR else ''}"
        )
        messages.append(perf_msg)

    # ── 主力策略：中長線 ──────────────────────────────────────────────────
    if long_df.empty:
        messages.append("🏔 <b>中長線（主力）</b>\n今日無訊號，持股不動")
    else:
        lines = [
            f"🏔 <b>中長線（主力）</b>  共 {len(long_df)} 支",
            f"停利 +30%  停損 -10%  最長 90 天",
            f"<i>超過 10 支時取成交金額最高者（流動性優先）</i>\n",
        ]
        for _, row in long_df.head(MAX_POSITIONS).iterrows():
            emoji = MARKET_EMOJI.get(row.get("market", "TWSE"), "⚪")
            sid   = row["stock_id"]
            close = row.get("close", "—")
            kd    = row.get("kd_k", 0)
            rsi   = row.get("rsi", 0)
            bb    = row.get("bb_pct", float("nan"))
            inst  = row.get("inst_total", 0)
            inst_str = f"+{int(inst//1000)}K" if inst > 0 else f"{int(inst//1000)}K"
            bb_str = f"{bb*100:.0f}%" if not pd.isna(bb) else "—"
            lines.append(
                f"{emoji} <b>{sid}</b>  ${close}  "
                f"BB{bb_str}  KD{kd:.0f}  RSI{rsi:.0f}  法人{inst_str}"
            )
        messages.append("\n".join(lines))
        _append_signal_log(long_df, date)

    # ── 策略五：月營收動能（每月 10 日後才有，其他日子不顯示）──────────────
    if not revenue_df.empty:
        rev_lines = [
            f"📊 <b>策略五：月營收動能</b>  共 {len(revenue_df)} 支",
            f"停利 +40%  停損 -12%  最長 120 天",
            f"<i>今為公布後第一交易日，基本面轉折訊號</i>\n",
        ]
        for _, row in revenue_df.head(MAX_POSITIONS).iterrows():
            emoji = MARKET_EMOJI.get(row.get("market", "TWSE"), "⚪")
            sid   = row["stock_id"]
            close = row.get("close", "—")
            rsi   = row.get("rsi", 0)
            inst  = row.get("inst_total", 0)
            inst_str = f"+{int(inst//1000)}K" if inst > 0 else f"{int(inst//1000)}K"
            rev_lines.append(
                f"{emoji} <b>{sid}</b>  ${close}  RSI{rsi:.0f}  法人{inst_str}"
            )
        messages.append("\n".join(rev_lines))

    # ── 執行紀律提醒 ──────────────────────────────────────────────────────
    messages.append(
        "📌 <i>紀律提醒：連續虧損時不可修改參數。"
        "停損是策略的一部分，不是失敗。</i>"
    )

    # ── 參考訊號 ──────────────────────────────────────────────────────────
    ref_parts = []
    if not short_df.empty:
        sids = " ".join(short_df["stock_id"].head(5).tolist())
        ref_parts.append(f"⚡ 短線參考：{sids}")
    if not swing_df.empty:
        sids = " ".join(swing_df["stock_id"].head(5).tolist())
        ref_parts.append(f"📈 波段參考：{sids}")
    if ref_parts:
        messages.append(
            "⚠️ <i>以下策略驗證期負期望值，僅供觀察勿重倉</i>\n"
            + "\n".join(ref_parts)
        )

    return messages


def notify(signals: dict[str, pd.DataFrame]) -> None:
    from datetime import datetime
    date = datetime.today().strftime("%Y-%m-%d")
    msgs = format_signals(signals, date)
    for msg in msgs:
        send_message(msg)
        import time; time.sleep(0.3)  # Telegram rate limit
