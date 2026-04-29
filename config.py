from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "cache.db"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# FinMind
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = ""  # set in .env

# Telegram
TELEGRAM_TOKEN = ""   # set in .env
TELEGRAM_CHAT_ID = "" # set in .env

# 手續費 / 稅率
FEE_RATE_BUY = 0.001425          # 買進手續費
FEE_RATE_SELL = 0.001425         # 賣出手續費
TAX_TWSE_OTC = 0.003             # 上市/上櫃 證交稅
TAX_EMERGING = 0.0015            # 興櫃 證交稅
SLIPPAGE = 0.001                 # 滑價假設（下單用次日開盤）

# 回測設定
BACKTEST_TRAIN_START = "2019-01-01"
BACKTEST_TRAIN_END   = "2022-12-31"
BACKTEST_TEST_START  = "2023-01-01"
BACKTEST_TEST_END    = "2025-12-31"

# 基本面篩選門檻
FUNDAMENTAL_FILTERS = {
    "min_eps": 1.0,              # 近四季 EPS 合計
    "min_roe": 12.0,             # ROE %
    "min_gross_margin": 15.0,    # 毛利率 %
    "pricing_power_margin": 25.0,# 定價權代理：毛利率門檻 %
    "min_ocf_ratio": 0.6,        # 現金轉換率（OCF/Net Income）
    "top_n_industry": 5,         # 同產業市值前 N 名
}

# 短線策略參數（待回測優化）
SHORT_TERM_PARAMS = {
    "vol_surge_ratio": 2.0,      # 成交量倍數（相對20日均量）
    "price_breakout_days": 20,   # 突破近 N 日高點
    "take_profit": 0.08,
    "stop_loss": 0.05,
    "max_hold_days": 5,
}

# 波段策略參數（待回測優化）
SWING_PARAMS = {
    "inst_buy_days": 3,          # 法人連續買超 N 天
    "kd_k_threshold": 20,        # KD K值低檔
    "take_profit": 0.15,
    "stop_loss": 0.07,
    "max_hold_days": 25,
}

# 中長線策略參數（待回測優化）
LONGTERM_PARAMS = {
    "eps_growth_quarters": 3,    # 連續 N 季 EPS 成長
    "revenue_growth_months": 3,  # 連續 N 月營收年增率 > 0
    "ma_aligned": True,          # 均線多頭排列
    "take_profit": 0.30,
    "stop_loss": 0.10,
    "max_hold_days": 90,
}
