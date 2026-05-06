"""
Discord 通知：透過 Webhook URL 發訊息（最簡單，不需要 bot token / OAuth）
與 telegram_bot 共用 format_signals，僅將 HTML 標籤轉為 Discord Markdown。
"""
import os
import re
import time
import logging
import requests
import pandas as pd
from dotenv import load_dotenv

from notify.telegram_bot import format_signals

load_dotenv(override=True)
logger = logging.getLogger(__name__)

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

DISCORD_MAX_LEN = 2000  # Discord 單則訊息上限


def _html_to_markdown(text: str) -> str:
    """把 telegram 格式的 <b>/<i> 轉成 Discord Markdown。其餘 HTML 標籤剝除。"""
    text = re.sub(r"</?b>", "**", text)
    text = re.sub(r"</?i>", "*", text)
    text = re.sub(r"<[^>]+>", "", text)  # 剝除其他 HTML
    return text


def _split_for_discord(text: str, limit: int = DISCORD_MAX_LEN) -> list[str]:
    """超過 2000 字時依行切成多則，避免 webhook 拒收。"""
    if len(text) <= limit:
        return [text]
    chunks, buf = [], ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > limit:
            if buf:
                chunks.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        chunks.append(buf)
    return chunks


def send_message(text: str) -> bool:
    if not WEBHOOK_URL:
        logger.warning("Discord not configured (DISCORD_WEBHOOK_URL missing)")
        return False
    md = _html_to_markdown(text)
    ok = True
    for chunk in _split_for_discord(md):
        try:
            r = requests.post(
                WEBHOOK_URL,
                json={"content": chunk},
                timeout=10,
            )
            r.raise_for_status()
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            ok = False
        time.sleep(0.3)  # webhook rate limit ~5 req/2s，保守一點
    return ok


def notify(signals: dict[str, pd.DataFrame]) -> None:
    from datetime import datetime
    date = datetime.today().strftime("%Y-%m-%d")
    msgs = format_signals(signals, date)
    for msg in msgs:
        send_message(msg)
        time.sleep(0.3)
