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


POSITION_SIZE_PCT = 5   # 每筆建議部位：總資金 5%
MAX_POSITIONS     = 10  # 最多同時持有 10 檔

def format_signals(signals: dict[str, pd.DataFrame], date: str) -> list[str]:
    """
    只主推策略四（中長線），其他策略降級為「參考訊號」。
    每筆附上部位建議（固定 5% 資金）。
    """
    messages = []

    long_df  = signals.get("long",  pd.DataFrame())
    short_df = signals.get("short", pd.DataFrame())
    swing_df = signals.get("swing", pd.DataFrame())

    total_main = len(long_df)
    header = (
        f"📊 <b>台股選股報告 {date}</b>\n"
        f"主力訊號（中長線）：{total_main} 支\n"
        f"建議部位：每筆 {POSITION_SIZE_PCT}%，最多同時持 {MAX_POSITIONS} 檔"
    )
    messages.append(header)

    # ── 主力策略：中長線 ──────────────────────────────────────────────────
    if long_df.empty:
        messages.append("🏔 <b>中長線（主力）</b>\n今日無訊號，持股不動")
    else:
        lines = [f"🏔 <b>中長線（主力）</b>  共 {len(long_df)} 支\n"
                 f"停利 +30%  停損 -10%  持股最長 90 天\n"]
        for _, row in long_df.head(10).iterrows():
            emoji = MARKET_EMOJI.get(row.get("market", "TWSE"), "⚪")
            sid   = row["stock_id"]
            close = row.get("close", "—")
            kd    = row.get("kd_k", 0)
            rsi   = row.get("rsi", 0)
            inst  = row.get("inst_total", 0)
            inst_str = f"+{int(inst//1000)}K" if inst > 0 else f"{int(inst//1000)}K"
            lines.append(
                f"{emoji} <b>{sid}</b>  ${close}  "
                f"KD{kd:.0f}  RSI{rsi:.0f}  法人{inst_str}"
            )
        messages.append("\n".join(lines))

    # ── 參考訊號（策略尚未通過驗證，僅供觀察）────────────────────────────
    ref_parts = []
    if not short_df.empty:
        sids = " ".join(short_df["stock_id"].head(5).tolist())
        ref_parts.append(f"⚡ 短線參考：{sids}")
    if not swing_df.empty:
        sids = " ".join(swing_df["stock_id"].head(5).tolist())
        ref_parts.append(f"📈 波段參考：{sids}")
    if ref_parts:
        messages.append("⚠️ <i>以下策略驗證期負期望值，僅供觀察勿重倉</i>\n" + "\n".join(ref_parts))

    return messages


def notify(signals: dict[str, pd.DataFrame]) -> None:
    from datetime import datetime
    date = datetime.today().strftime("%Y-%m-%d")
    msgs = format_signals(signals, date)
    for msg in msgs:
        send_message(msg)
        import time; time.sleep(0.3)  # Telegram rate limit
