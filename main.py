"""
主入口：
  python main.py screen     # 每日選股（GitHub Actions 用）
  python main.py backtest   # 跑回測 + 輸出報告
  python main.py download   # 只下載資料不選股
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
        from notify.telegram_bot import notify
        run_daily(notify_fn=notify)

    elif mode == "backtest":
        subprocess.run([
            sys.executable, "-m", "backtest.run_backtest",
            "--mode", "strategy",
        ])

    elif mode == "download":
        from data.cache import init_db
        from data.universe import build_universe
        from backtest.run_backtest import download_all
        init_db()
        universe = build_universe(force_refresh=True)
        download_all(universe)

    elif mode == "optimize":
        subprocess.run([
            sys.executable, "-m", "backtest.run_backtest",
            "--mode", "optimize",
            "--strategy", sys.argv[2] if len(sys.argv) > 2 else "0",
        ])

    else:
        print("Usage: python main.py [screen|backtest|download|optimize]")
        sys.exit(1)


if __name__ == "__main__":
    main()
