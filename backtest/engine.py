"""
輕量回測引擎：
- 進場：訊號當天收盤後確認，次日開盤成交（保守假設）
- 出場：停利 / 停損 / 最大持有天數，用次日開盤或收盤決定
- 費用：手續費 + 證交稅（依市場別）
- 支援多空都測（預設只做多）
"""
from __future__ import annotations
import logging
import pandas as pd
from dataclasses import dataclass, field
from config import (
    FEE_RATE_BUY, FEE_RATE_SELL,
    TAX_TWSE_OTC, TAX_EMERGING,
    SLIPPAGE,
)

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    stock_id: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    market: str = "TWSE"

    @property
    def pnl_pct(self) -> float | None:
        if self.exit_price is None:
            return None
        tax = TAX_EMERGING if self.market == "Emerging" else TAX_TWSE_OTC
        cost = FEE_RATE_BUY + FEE_RATE_SELL + tax
        raw = (self.exit_price - self.entry_price) / self.entry_price
        return raw - cost

    @property
    def hold_days(self) -> int | None:
        if self.exit_date is None:
            return None
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)

    @property
    def closed(self) -> list[Trade]:
        return [t for t in self.trades if t.exit_price is not None]

    def to_df(self) -> pd.DataFrame:
        if not self.closed:
            return pd.DataFrame()
        rows = []
        for t in self.closed:
            rows.append({
                "stock_id":    t.stock_id,
                "entry_date":  t.entry_date,
                "exit_date":   t.exit_date,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "pnl_pct":     t.pnl_pct,
                "hold_days":   t.hold_days,
                "exit_reason": t.exit_reason,
                "market":      t.market,
            })
        return pd.DataFrame(rows)


def run_backtest(
    price_df: pd.DataFrame,
    signal_col: str,
    take_profit: float,
    stop_loss: float,
    max_hold_days: int,
    start: str,
    end: str,
    stock_id: str = "",
    market: str = "TWSE",
    consec_down_exit: bool = False,
) -> BacktestResult:
    """
    price_df 必須有欄位：date, open, high, low, close, <signal_col>
    signal_col: True/1 表示當天收盤後產生買進訊號，次日開盤進場
    consec_down_exit: True 時，持倉期間連續兩日收低則次日開盤出場
    """
    df = price_df.copy()
    df = df[(df["date"] >= pd.Timestamp(start)) &
            (df["date"] <= pd.Timestamp(end))].reset_index(drop=True)

    if df.empty or signal_col not in df.columns:
        return BacktestResult()

    # 過濾掉價格異常的資料（0 或負數）
    df = df[df["open"] > 0].reset_index(drop=True)
    df = df[df["close"] > 0].reset_index(drop=True)
    if len(df) < 10:
        return BacktestResult()

    result = BacktestResult()
    in_trade = False
    trade: Trade | None = None
    down_streak: int = 0
    prev_close_in_trade: float = 0.0

    for i in range(len(df) - 1):
        row   = df.iloc[i]
        next_ = df.iloc[i + 1]

        # ── 持倉中：檢查出場條件 ──────────────────────────────────────────
        if in_trade and trade is not None:
            ep = trade.entry_price
            if ep <= 0:  # 異常進場價，直接平倉
                trade.exit_date = row["date"]
                trade.exit_price = row["close"] if row["close"] > 0 else ep
                trade.exit_reason = "invalid_price"
                result.trades.append(trade)
                in_trade = False
                trade = None
                down_streak = 0
                prev_close_in_trade = 0.0
                continue

            # 連跌計數（收盤低於前一日收盤）
            if consec_down_exit:
                if prev_close_in_trade > 0:
                    if row["close"] < prev_close_in_trade:
                        down_streak += 1
                    else:
                        down_streak = 0
                prev_close_in_trade = row["close"]

            high_ret = (row["high"] - ep) / ep
            low_ret  = (row["low"]  - ep) / ep
            sl_hit = low_ret <= -stop_loss
            tp_hit = high_ret >= take_profit

            exit_price = None
            reason = ""

            if sl_hit and tp_hit:
                # Both triggered intraday — daily OHLCV can't tell which fired
                # first. Use the open as a tiebreaker: a gap-up open above TP
                # almost certainly hit TP first; otherwise default to SL
                # (conservative). See issue #4.
                if row["open"] >= ep * (1 + take_profit):
                    exit_price = ep * (1 + take_profit)
                    reason = "take_profit"
                else:
                    exit_price = ep * (1 - stop_loss)
                    reason = "stop_loss"
            elif sl_hit:
                exit_price = ep * (1 - stop_loss)
                reason = "stop_loss"
            elif tp_hit:
                exit_price = ep * (1 + take_profit)
                reason = "take_profit"
            elif (row["date"] - trade.entry_date).days >= max_hold_days:
                # 到期用次日開盤出場（與進場邏輯一致，訊號T日確認→T+1執行）
                exit_price = next_["open"] * (1 - SLIPPAGE)
                reason = "max_hold"
            elif consec_down_exit and down_streak >= 2:
                # 連跌兩日動能止損，次日開盤出
                exit_price = next_["open"] * (1 - SLIPPAGE)
                reason = "consec_down"

            if exit_price is not None:
                trade.exit_date  = next_["date"] if reason in ("max_hold", "consec_down") else row["date"]
                trade.exit_price = exit_price
                trade.exit_reason = reason
                result.trades.append(trade)
                in_trade = False
                trade = None
                down_streak = 0
                prev_close_in_trade = 0.0
                continue

        # ── 無持倉：檢查進場訊號 ─────────────────────────────────────────
        if not in_trade and bool(row.get(signal_col, False)):
            # 次日開盤進場，加滑價
            entry_px = next_["open"] * (1 + SLIPPAGE)
            trade = Trade(
                stock_id=stock_id,
                entry_date=next_["date"],
                entry_price=entry_px,
                market=market,
            )
            in_trade = True
            down_streak = 0
            prev_close_in_trade = 0.0

    # 回測結束時強制平倉未了結部位
    if in_trade and trade is not None:
        last = df.iloc[-1]
        trade.exit_date   = last["date"]
        trade.exit_price  = last["close"]
        trade.exit_reason = "end_of_period"
        result.trades.append(trade)

    return result


def run_portfolio_backtest(
    stock_price_map: dict[str, pd.DataFrame],
    signal_col: str,
    take_profit: float,
    stop_loss: float,
    max_hold_days: int,
    start: str,
    end: str,
    market_map: dict[str, str] | None = None,
    consec_down_exit: bool = False,
) -> BacktestResult:
    """多股票批次回測，整合所有交易到一個 BacktestResult"""
    combined = BacktestResult()
    for stock_id, price_df in stock_price_map.items():
        market = (market_map or {}).get(stock_id, "TWSE")
        r = run_backtest(
            price_df, signal_col,
            take_profit, stop_loss, max_hold_days,
            start, end, stock_id, market,
            consec_down_exit=consec_down_exit,
        )
        combined.trades.extend(r.trades)
    return combined
