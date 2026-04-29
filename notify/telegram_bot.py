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


def format_signals(signals: dict[str, pd.DataFrame], date: str) -> list[str]:
    """把三個時間框架的訊號格式化成 Telegram 訊息列表"""
    messages = []

    header = f"📊 <b>台股選股報告 {date}</b>\n"
    total = sum(len(df) for df in signals.values())
    header += f"本日合計 {total} 個訊號\n"
    messages.append(header)

    for tf in ["short", "swing", "long"]:
        df = signals.get(tf, pd.DataFrame())
        label = TF_LABEL.get(tf, tf)

        if df.empty:
            messages.append(f"{label}\n（今日無訊號）")
            continue

        lines = [f"{label}  共 {len(df)} 支\n"]
        for _, row in df.head(15).iterrows():  # 最多顯示 15 支
            emoji = MARKET_EMOJI.get(row.get("market", "TWSE"), "⚪")
            sid   = row["stock_id"]
            close = row.get("close", "—")
            vr    = row.get("vol_ratio", 0)
            kd    = row.get("kd_k", "—")
            inst  = row.get("inst_total", 0)
            ma    = "✓" if row.get("ma_aligned") else "✗"

            inst_str = f"+{int(inst//1000)}K" if inst > 0 else f"{int(inst//1000)}K"
            lines.append(
                f"{emoji} <b>{sid}</b>  ${close}  "
                f"量{vr:.1f}x  KD{kd:.0f}  法人{inst_str}  均{ma}"
            )

        messages.append("\n".join(lines))

    return messages


def notify(signals: dict[str, pd.DataFrame]) -> None:
    from datetime import datetime
    date = datetime.today().strftime("%Y-%m-%d")
    msgs = format_signals(signals, date)
    for msg in msgs:
        send_message(msg)
        import time; time.sleep(0.3)  # Telegram rate limit
