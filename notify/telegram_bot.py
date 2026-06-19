"""
Telegram 通知：用同步 requests 發訊息，不需要 async（GitHub Actions 環境簡單用）
訊號格式：每個時間框架一則訊息，清楚列出股票代號、市場、關鍵指標
"""
import os
import time
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from data.cache import load_universe

load_dotenv(override=True)
logger = logging.getLogger(__name__)

TOKEN    = os.getenv("TELEGRAM_TOKEN", "").strip()
API_URL  = f"https://api.telegram.org/bot{TOKEN}"
# 支援多個接收者：逗號分隔，例如 "123456,-100987654321"
CHAT_IDS = [c.strip() for c in os.getenv("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]

MARKET_EMOJI = {"TWSE": "🔵", "TPEx": "🟢", "Emerging": "🟡"}
TF_LABEL = {
    "short": "⚡ 短線（1-5天）",
    "swing": "📈 波段（1-4週）",
    "long":  "🏔 中長線（1-3月）",
}


def send_message(text: str) -> bool:
    if not TOKEN or not CHAT_IDS:
        logger.warning("Telegram not configured (TOKEN or CHAT_ID missing)")
        return False
    ok = True
    for chat_id in CHAT_IDS:
        try:
            r = requests.post(
                f"{API_URL}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            r.raise_for_status()
        except Exception as e:
            logger.error(f"Telegram send failed (chat_id={chat_id}): {e}")
            ok = False
    return ok


MAX_POSITIONS      = 10   # 最多同時持有 10 檔（S4/S5/S7：回測顯示 10 檔已足/最佳）
MAX_POSITIONS_S6   = 15   # S6 例外：回測顯示 S6 平均需 ~14 檔，上限 10 時回撤
                          # 由 −8% 惡化到 −14%，放寬到 15 才回到合理分散度
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


def _name_map() -> dict[str, str]:
    try:
        u = load_universe()
        return dict(zip(u["stock_id"], u["stock_name"])) if not u.empty else {}
    except Exception:
        return {}


def _aqs_plain(score: float, stage: str) -> str:
    """AQS 分數+階段 → 一句白話判讀（不露數字/階段詞）。"""
    if pd.isna(score):
        return ""
    if score >= 70:
        if "早期" in stage:
            return "✅ 真累積，進場好時機"
        if "末段" in stage:
            return "⚠️ 已經漲一段，要進就減半"
        return "✅ 真累積，可進但別追高"      # 中期/其他高分
    if score >= 50:
        return "🟡 不夠強，別主動追"
    if "派發" in stage:
        return "🚫 可能在騙散戶，千萬不要追"
    return "🚫 籌碼很差，避開"


def format_signals(signals: dict[str, pd.DataFrame], date: str) -> list[str]:
    """
    只主推策略四（中長線）。
    訊號超過 10 支時按成交量排序（流動性優先）。
    附近期績效監控與執行紀律提醒。
    """
    messages = []
    names = _name_map()

    long_df    = signals.get("long",    pd.DataFrame())
    revenue_df = signals.get("revenue", pd.DataFrame())
    growth_df  = signals.get("growth",  pd.DataFrame())
    accum_df   = signals.get("accum",   pd.DataFrame())
    combo_df   = signals.get("combo_47", pd.DataFrame())
    meta_df    = signals.get("_meta",   pd.DataFrame())
    regime_label = (meta_df.iloc[0]["regime_label"]
                    if not meta_df.empty and "regime_label" in meta_df.columns else "")
    regime_ret = (meta_df.iloc[0]["regime_60d_return"]
                  if not meta_df.empty and "regime_60d_return" in meta_df.columns else 0.0)

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
    regime_line = (f"\n大盤 regime：{regime_label}（0050 60日 {regime_ret*100:+.1f}%）"
                   if regime_label else "")
    header = (
        f"📊 <b>台股選股報告 {date}</b>\n"
        f"主力訊號（中長線）：{len(long_df)} 支"
        f"{regime_line}"
    )
    messages.append(header)

    # ── 📤 訊號出場/注意（系統發過進場訊號者，籌碼惡化才提醒；非個人持股）──
    exits_df = signals.get("exits", pd.DataFrame())
    if isinstance(exits_df, pd.DataFrame) and not exits_df.empty:
        order = {"🚨 出場": 0, "⚠️ 注意": 1}
        exits_df = (exits_df.assign(_o=exits_df["level"].map(lambda x: order.get(x, 2)))
                    .sort_values("_o"))
        e_lines = ["📤 <b>訊號出場/注意</b>（系統發過進場訊號者，籌碼惡化才提醒）"]
        for _, r in exits_df.iterrows():
            sid = str(r["stock_id"])
            nm = r.get("name") or names.get(sid, "")
            e_lines.append(
                f"{r['level']} <b>{sid} {nm}</b>（{r['strategy']}）  "
                f"進場 {r['entry_date']} @{r['entry_price']:.1f} → 今 {r['close']:.1f}"
                f"（{r['pnl_pct']:+.1f}%）\n  {r['reason']}"
            )
        messages.append("\n".join(e_lines))

    # ── 🎯 高信心：S4 ∩ S7 兩月內交集 ─────────────────────────────────────
    if not combo_df.empty:
        c_lines = [
            f"🎯 <b>高信心進場 (S4 ∩ S7, 20d 窗)</b>  共 {len(combo_df)} 支",
            "<i>回測 60 日勝率 66.3%、平均 +11.33%（vs S7 only 56.9%）</i>\n",
        ]
        for _, row in combo_df.head(MAX_POSITIONS).iterrows():
            emoji = MARKET_EMOJI.get(row.get("market", "TWSE"), "⚪")
            sid   = row["stock_id"]
            close = row.get("close", "—")
            f60   = row.get("f_60d", float("nan"))
            t60   = row.get("t_60d", float("nan"))
            inst60 = (0 if pd.isna(f60) else f60) + (0 if pd.isna(t60) else t60)
            inst_str = f"+{int(inst60//1000):,}張" if inst60 > 0 else f"{int(inst60//1000):,}張"
            name = names.get(sid, "")
            s4_today = row.get("s4_today", False)
            s7_today = row.get("s7_today", False)
            trigger = "今日 S4+S7" if s4_today and s7_today else ("今日 S4" if s4_today else "今日 S7")
            plain = _aqs_plain(row.get("aqs_score", float("nan")), row.get("aqs_stage", ""))
            sub = f"{trigger} · {plain}" if plain else trigger
            c_lines.append(
                f"{emoji} <b>{sid} {name}</b>  ${close}  法人60日{inst_str}\n  {sub}"
            )
        messages.append("\n".join(c_lines))

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
            "停利 +30%  停損 -10%  最長 90 天",
            "<i>超過 10 支時取成交金額最高者（流動性優先）</i>\n",
        ]
        for _, row in long_df.head(MAX_POSITIONS).iterrows():
            emoji = MARKET_EMOJI.get(row.get("market", "TWSE"), "⚪")
            sid   = row["stock_id"]
            close = row.get("close", "—")
            f60   = row.get("f_60d", float("nan"))
            t60   = row.get("t_60d", float("nan"))
            inst60  = (0 if pd.isna(f60) else f60) + (0 if pd.isna(t60) else t60)
            inst_str = f"+{int(inst60//1000):,}張" if inst60 > 0 else f"{int(inst60//1000):,}張"
            name = names.get(sid, "")
            plain = _aqs_plain(row.get("aqs_score", float("nan")), row.get("aqs_stage", ""))
            line = f"{emoji} <b>{sid} {name}</b>  ${close}  法人60日{inst_str}"
            if plain:
                line += f"\n  {plain}"
            lines.append(line)
        messages.append("\n".join(lines))
        _append_signal_log(long_df, date)

    # ── 策略五：月營收動能（每月 10 日後才有，其他日子不顯示）──────────────
    if not revenue_df.empty:
        # Regime-aware：2026 多頭 S5 80% 勝率 +42%；空頭時 0%；以 0050 60d 報酬為閘
        if "多頭" in regime_label:
            s5_priority = "🔥 <b>主力升級</b>（2026 多頭，回測 80% 勝率 +42%）"
        elif "空頭" in regime_label:
            s5_priority = "🥶 <b>建議暫停或減半</b>（空頭環境 S5 失效）"
        else:
            s5_priority = "🟡 中性環境，正常部位"
        rev_lines = [
            f"📊 <b>策略五：月營收動能</b>  共 {len(revenue_df)} 支",
            f"停利 +40%  停損 -12%  最長 120 天   {s5_priority}",
            "<i>今為公布後第一交易日，基本面轉折訊號</i>\n",
        ]
        for _, row in revenue_df.head(MAX_POSITIONS).iterrows():
            emoji = MARKET_EMOJI.get(row.get("market", "TWSE"), "⚪")
            sid   = row["stock_id"]
            close = row.get("close", "—")
            yoy   = row.get("revenue_yoy", float("nan"))
            f20   = row.get("f_20d", float("nan"))
            yoy_str = f"+{yoy:.0f}%" if not pd.isna(yoy) else "—"
            f20_val = 0 if pd.isna(f20) else f20
            f20_str = f"+{int(f20_val//1000):,}張" if f20_val > 0 else f"{int(f20_val//1000):,}張"
            name = names.get(sid, "")
            rev_lines.append(
                f"{emoji} <b>{sid} {name}</b>  ${close}  營收年增{yoy_str}  外資20日{f20_str}"
            )
        messages.append("\n".join(rev_lines))

    # ── 策略六：高成長突破（regime-conditional，AI bull 時加強）─────────────
    if not growth_df.empty:
        # 按法人 60 日由大到小排
        if "f_60d" in growth_df.columns and "t_60d" in growth_df.columns:
            growth_df = growth_df.assign(
                _inst60=growth_df["f_60d"].fillna(0) + growth_df["t_60d"].fillna(0)
            ).sort_values("_inst60", ascending=False).drop(columns="_inst60")
        g_lines = [
            f"🚀 <b>策略六：高成長突破</b>  共 {len(growth_df)} 支",
            "停利 +30%  停損 -10%  trailing +80%/-15%  最長 90 天",
            "<i>⚠️ regime-conditional，勝率僅 ~38%（少數大贏家拉抬）</i>",
            "<i>⚠️ 部位應比主力小（建議 S4 的 1/2 ~ 2/3）</i>\n",
        ]
        for _, row in growth_df.head(MAX_POSITIONS_S6).iterrows():
            emoji = MARKET_EMOJI.get(row.get("market", "TWSE"), "⚪")
            sid   = row["stock_id"]
            close = row.get("close", "—")
            f60   = row.get("f_60d", float("nan"))
            t60   = row.get("t_60d", float("nan"))
            inst60 = (0 if pd.isna(f60) else f60) + (0 if pd.isna(t60) else t60)
            inst_str = f"+{int(inst60//1000):,}張" if inst60 > 0 else f"{int(inst60//1000):,}張"
            name = names.get(sid, "")
            plain = _aqs_plain(row.get("aqs_score", float("nan")), row.get("aqs_stage", ""))
            line = f"{emoji} <b>{sid} {name}</b>  ${close}  法人60日{inst_str}"
            if plain:
                line += f"\n  {plain}"
            g_lines.append(line)
        messages.append("\n".join(g_lines))

    # ── 策略七：累積前夕（狙擊手型，每年 ~70 筆訊號）──────────────────────
    if not accum_df.empty:
        # 排序：先按 above_ma20（優先級高的在上），再按法人 60 日由大到小
        if "above_ma20" in accum_df.columns:
            accum_df = accum_df.assign(
                _sort_key=accum_df["above_ma20"].astype(int)
            )
            if "f_60d" in accum_df.columns and "t_60d" in accum_df.columns:
                accum_df["_inst60"] = accum_df["f_60d"].fillna(0) + accum_df["t_60d"].fillna(0)
                accum_df = accum_df.sort_values(["_sort_key", "_inst60"],
                                                ascending=[False, False])
                accum_df = accum_df.drop(columns=["_sort_key", "_inst60"])
            else:
                accum_df = accum_df.sort_values("_sort_key", ascending=False)
                accum_df = accum_df.drop(columns="_sort_key")
        elif "f_60d" in accum_df.columns and "t_60d" in accum_df.columns:
            accum_df = accum_df.assign(
                _inst60=accum_df["f_60d"].fillna(0) + accum_df["t_60d"].fillna(0)
            ).sort_values("_inst60", ascending=False).drop(columns="_inst60")
        a_lines = [
            f"🎯 <b>策略七：累積前夕</b>  共 {len(accum_df)} 支",
            "停損 -20%（寬）  trailing +80%/-15%  最長 180 天",
            "<i>📡 抓「法人偷收貨、股價還沒反映」的早中期累積</i>",
            "<i>⚡ = 已破 MA20（趨勢較確定、勝率較高，優先看）</i>",
            "<i>⚠️ 勝率 ~52%，連虧 3-4 筆是正常；部位建議 S4 的 1/3</i>\n",
        ]
        for _, row in accum_df.head(MAX_POSITIONS).iterrows():
            emoji = MARKET_EMOJI.get(row.get("market", "TWSE"), "⚪")
            sid   = row["stock_id"]
            close = row.get("close", "—")
            f60   = row.get("f_60d", float("nan"))
            t60   = row.get("t_60d", float("nan"))
            inst60 = (0 if pd.isna(f60) else f60) + (0 if pd.isna(t60) else t60)
            inst_str = f"+{int(inst60//1000):,}張" if inst60 > 0 else f"{int(inst60//1000):,}張"
            name = names.get(sid, "")
            plain = _aqs_plain(row.get("aqs_score", float("nan")), row.get("aqs_stage", ""))
            # ⚡ 標籤：close > MA20（基於失敗分析：勝率 66%→72%）
            tag = "⚡ " if row.get("above_ma20", False) else "  "
            line = f"{tag}{emoji} <b>{sid} {name}</b>  ${close}  法人60日{inst_str}"
            if plain:
                line += f"\n  {plain}"
            a_lines.append(line)
        messages.append("\n".join(a_lines))

    # ── 執行紀律提醒 ──────────────────────────────────────────────────────
    messages.append(
        "📌 <i>紀律提醒：連續虧損時不可修改參數。"
        "停損是策略的一部分，不是失敗。</i>"
    )

    return messages


def notify(signals: dict[str, pd.DataFrame]) -> None:
    from datetime import datetime
    date = datetime.today().strftime("%Y-%m-%d")
    msgs = format_signals(signals, date)
    for msg in msgs:
        send_message(msg)
        time.sleep(0.3)  # Telegram rate limit
