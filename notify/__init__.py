"""
Dispatcher：依環境變數自動選擇要推播的平台。
- TELEGRAM_TOKEN + TELEGRAM_CHAT_ID 設定時推 Telegram
- DISCORD_WEBHOOK_URL 設定時推 Discord
- 兩個都設則同時推；都沒設則 no-op（搭配 logger 警告）
"""
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def notify(signals: dict[str, pd.DataFrame]) -> None:
    sent_any = False

    if os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        from notify.telegram_bot import notify as tg_notify
        tg_notify(signals)
        sent_any = True

    if os.getenv("DISCORD_WEBHOOK_URL"):
        from notify.discord_bot import notify as dc_notify
        dc_notify(signals)
        sent_any = True

    if not sent_any:
        logger.warning(
            "No notification platform configured "
            "(set TELEGRAM_TOKEN+TELEGRAM_CHAT_ID and/or DISCORD_WEBHOOK_URL)"
        )
