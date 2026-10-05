"""
主入口：
  python main.py screen     # 每日選股（GitHub Actions 用）
  python main.py rules      # 發送策略規則對照並置頂 Telegram
  python main.py backtest   # 跑回測 + 輸出報告
  python main.py bootstrap [2019-01-01]  # 從零重建資料庫（全部官方來源）
"""
import sys
import subprocess
import logging
from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"

    if mode == "screen":
        from screener.daily_run import run_daily
        from notify import notify
        run_daily(notify_fn=notify)

    elif mode == "rules":
        # 發送策略規則對照並置頂（改規則後重發一次即可）
        from notify.telegram_bot import send_rules
        send_rules(pin=True)

    elif mode == "backtest":
        subprocess.run([
            sys.executable, "-m", "backtest.run_backtest",
            "--mode", "strategy",
        ])

    elif mode == "bootstrap":
        # 從零重建資料庫（全部官方來源）：python main.py bootstrap [2019-01-01]
        import logging
        from data.cache import init_db
        from data.universe import build_universe
        from screener.daily_run import bootstrap_official
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        init_db()
        universe = build_universe(force_refresh=True)
        if universe.empty:
            logging.error("Universe is empty — TWSE/TPEx ISIN 頁面無法取得")
            sys.exit(1)
        bootstrap_official(universe, sys.argv[2] if len(sys.argv) > 2 else "2019-01-01")

    elif mode == "optimize":
        subprocess.run([
            sys.executable, "-m", "backtest.run_backtest",
            "--mode", "optimize",
            "--strategy", sys.argv[2] if len(sys.argv) > 2 else "0",
        ])

    else:
        print("Usage: python main.py [screen|rules|backtest|bootstrap|optimize]")
        sys.exit(1)


if __name__ == "__main__":
    main()
