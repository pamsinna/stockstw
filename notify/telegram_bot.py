"""
Telegram 通知：用同步 requests 發訊息，不需要 async（GitHub Actions 環境簡單用）
訊號格式：照動作分三層（🚨要處理 → 🛒新進場候選 → 👀觀察），規則另發置頂訊息
"""
import os
import time
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from data.cache import load_universe, load_prices

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

MOM_LOOKBACK = 20   # 候選排序用動能窗口（20 交易日；實測 20-60 為穩定平台、最不易過擬合）


def _mom20(stock_id: str, date: str) -> float:
    """過去 MOM_LOOKBACK 交易日報酬，供候選排序 + 弱動能標籤。取不到回 nan。"""
    try:
        start = (pd.Timestamp(date) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        px = load_prices(str(stock_id), start=start, end=pd.Timestamp(date).strftime("%Y-%m-%d"))
        if px is None or px.empty or len(px) < MOM_LOOKBACK + 1:
            return float("nan")
        c = px.sort_values("date")["close"].to_numpy(dtype=float)
        return c[-1] / c[-(MOM_LOOKBACK + 1)] - 1.0
    except Exception:
        return float("nan")


def _rank_mom(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """候選按動能 20 日由強到弱排序（弱者沉底、na 墊底）。

    實測（2023-25 樣本外）：候選內按動能排序，前半 vs 後半淨報酬差 +0.68pp，
    且 20-60 日窗口一致；法人/成交金額排序無此效果。
    """
    if df is None or df.empty or "stock_id" not in df.columns:
        return df
    df = df.copy()
    df["_mom20"] = [_mom20(s, date) for s in df["stock_id"]]
    return df.sort_values("_mom20", ascending=False, na_position="last")


def _mom_str(row) -> str:
    """header 尾綴：顯示動能；≤0 標「動能未表態」（實測這批買了偏賠、可跳過）。"""
    m = row.get("_mom20", float("nan"))
    if pd.isna(m):
        return ""
    return f"  動能{m:+.0%}" + ("  ⚠️動能未表態" if m <= 0 else "")


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


# ─── 每日通知：照「動作」分三層（要處理 → 新進場候選 → 觀察），不照策略分 ──────
# 2026-10 改版：舊版每策略一則、同股重複出現、每天重印停利停損說明 → 易讀性差。
# 規則類說明移到置頂訊息（rules_message / send_rules），每日只用 [S4] 標籤。

TG_MAX_LEN = 4000           # Telegram 上限 4096，留緩衝
MAX_CANDIDATES = 15         # 進場候選合併後上限（依動能取前 N）
MAX_WATCH = 15              # 觀察名單上限
_TAG_ORDER = [("long", "S4"), ("revenue", "S5"), ("growth", "S6"), ("accum", "S7")]
_WEEKDAY = "一二三四五六日"


def _px(v) -> str:
    """價格依台股升降單位顯示：< 50 元到 0.01、< 500 到 0.1、其餘整數。
    舊版一律 .1f → 24.85→24.90 顯示成「24.9→24.9（+0.2%）」。"""
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    if v < 50:
        return f"{v:,.2f}"
    if v < 500:
        return f"{v:,.1f}"
    return f"{v:,.0f}"


def _zhang(v: float) -> str:
    """股 → 張，帶正負號。"""
    if v is None or pd.isna(v):
        return "—"
    return f"{v / 1000:+,.0f}張"


def _merge_candidates(signals: dict[str, pd.DataFrame], drop: set[str]) -> pd.DataFrame:
    """S4/S5/S6/S7 + S4∩S7 合併成每檔一列，tags = 命中的策略標籤。"""
    rows: dict[str, dict] = {}
    for key, tag in _TAG_ORDER:
        df = signals.get(key)
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            sid = str(r["stock_id"])
            if sid in drop:
                continue
            cur = rows.setdefault(sid, {**r.to_dict(), "stock_id": sid, "tags": []})
            cur["tags"].append(tag)
            # 補齊先前策略沒有的欄位（例：S5 沒有 AQS、S4 沒有營收年增）
            for k, v in r.to_dict().items():
                if _isnan(cur.get(k)) and not _isnan(v):
                    cur[k] = v
    # S4∩S7：今天觸發一邊、另一邊在近 20 日觸發過 → 兩個標籤都補上（另一邊標 *）
    combo = signals.get("combo_47")
    if combo is not None and not combo.empty:
        for _, r in combo.iterrows():
            sid = str(r["stock_id"])
            if sid in drop:
                continue
            cur = rows.setdefault(sid, {**r.to_dict(), "stock_id": sid, "tags": []})
            for tag in ("S4", "S7"):
                if tag not in cur["tags"]:
                    cur["tags"].append(f"{tag}*")
            cur["tags"].sort(key=lambda t: t.rstrip("*"))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(list(rows.values()))


def _candidate_line(i: int, r, names: dict[str, str]) -> str:
    sid = r["stock_id"]
    tags = "+".join(r["tags"])
    star = "⭐" if len(r["tags"]) >= 2 else ""
    f60 = r.get("f_60d", float("nan"))
    t60 = r.get("t_60d", float("nan"))
    inst60 = (0 if pd.isna(f60) else f60) + (0 if pd.isna(t60) else t60)
    parts = [f"{i}. <b>{sid} {names.get(sid, '')}</b>  {_px(r.get('close'))}  [{tags}{star}]"]
    detail = []
    if any(t.startswith("S5") for t in r["tags"]) and not pd.isna(r.get("revenue_yoy", float("nan"))):
        detail.append(f"營收年增{r['revenue_yoy']:+.0f}%")
    if inst60:
        detail.append(f"法人60日{_zhang(inst60)}")
    m = r.get("_mom20", float("nan"))
    if not pd.isna(m):
        detail.append(f"動能{m:+.0%}" + ("⚠️未表態" if m <= 0 else ""))
    plain = _aqs_plain(r.get("aqs_score", float("nan")), r.get("aqs_stage", "") or "")
    line = parts[0] + ("\n   " + "  ".join(detail) if detail else "")
    if plain:
        line += f"\n   {plain}"
    return line


def _watch_section(watch: pd.DataFrame, names: dict[str, str]) -> list[str]:
    """觀察名單：依產業分組（檔數多的族群在前），組內 🆕 優先、再按觸發後漲幅。"""
    if watch is None or watch.empty:
        return []
    w = watch.copy()
    w["stock_id"] = w["stock_id"].astype(str)
    w = w.sort_values(["is_new", "since_pct"], ascending=[False, False])
    n_total, n_new = len(w), int(w["is_new"].sum())
    counts = w["industry"].value_counts()
    w["_ind_n"] = w["industry"].map(counts)
    w = w.sort_values(["_ind_n", "industry", "is_new", "since_pct"],
                      ascending=[False, True, False, False]).head(MAX_WATCH)
    lines = [f"👀 <b>營收爆發觀察</b>（近20日 {n_total} 檔，🆕 今日 {n_new}）",
             "<i>不是進場訊號；看哪個族群集中、觸發後有沒有續漲，題材自己判斷</i>"]
    for ind, g in w.groupby("industry", sort=False):
        lines.append(f"\n<b>{ind}</b> {counts[ind]}")
        for _, r in g.iterrows():
            new = "🆕" if r["is_new"] else ""
            since = "" if r["is_new"] else f"（觸發後{r['since_pct']:+.0f}%）"
            lines.append(
                f" {new}{r['stock_id']} {names.get(r['stock_id'], '')}  {_px(r['close'])}{since}"
                f"  營收年增{r['burst_yoy']:+.0f}% 近3月{r['burst_g3']:+.0f}%  外資60日{_zhang(r['f_60d'])}"
            )
    if n_total > len(w):
        lines.append(f"\n<i>…另有 {n_total - len(w)} 檔，見 reports/signals_watch CSV</i>")
    return lines


def _pack(blocks: list[str]) -> list[str]:
    """把區塊依序塞進 ≤ TG_MAX_LEN 的訊息（區塊間空一行）；過長的區塊按行切。"""
    msgs: list[str] = []
    cur = ""
    for block in blocks:
        for i, line in enumerate(block.split("\n")):
            sep = "" if not cur else ("\n\n" if i == 0 else "\n")
            if cur and len(cur) + len(sep) + len(line) > TG_MAX_LEN:
                msgs.append(cur)
                cur, sep = "", ""
            cur += sep + line
    if cur:
        msgs.append(cur)
    return msgs


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v))


def format_signals(signals: dict[str, pd.DataFrame], date: str) -> list[str]:
    names = _name_map()
    meta = signals.get("_meta", pd.DataFrame())

    def mget(k, default=""):
        if meta is None or meta.empty or k not in meta.columns:
            return default
        v = meta.iloc[0][k]
        return default if _isnan(v) else v

    regime_label, regime_ret = mget("regime_label"), mget("regime_60d_return", 0.0)

    # ── Header：一行大盤脈絡 ───────────────────────────────────────────────
    wd = _WEEKDAY[pd.Timestamp(date).weekday()]
    head = f"📊 <b>{pd.Timestamp(date):%m/%d}（{wd}）</b>"
    if regime_label:
        head += f"  {regime_label}｜0050 60日 {regime_ret*100:+.1f}%"
    blocks = [head]

    # ── 🚨 要處理：系統發過訊號者籌碼惡化 ──────────────────────────────────
    exits = signals.get("exits", pd.DataFrame())
    exit_ids: set[str] = set()
    if isinstance(exits, pd.DataFrame) and not exits.empty:
        order = {"🚨 出場": 0, "⚠️ 注意": 1}
        exits = exits.assign(_o=exits["level"].map(lambda x: order.get(x, 2))).sort_values("_o")
        # 只有 🚨 出場才從進場候選拿掉（真矛盾）；⚠️ 注意可共存
        exit_ids = set(exits[exits["level"].astype(str).str.contains("出場")]["stock_id"].astype(str))
        # 同一檔跨策略（例：奇鋐 S6+S7）合併成一列：取最嚴重等級、最早進場
        merged = []
        for sid, g in exits.groupby(exits["stock_id"].astype(str), sort=False):
            g = g.sort_values("entry_date")
            first = g.iloc[0]
            reasons = list(dict.fromkeys(str(x) for x in g["reason"] if str(x)))
            merged.append({**first.to_dict(), "level": g.sort_values("_o").iloc[0]["level"],
                           "strategy": "+".join(dict.fromkeys(g["strategy"].astype(str))),
                           "reason": "；".join(reasons), "_o": g["_o"].min()})
        merged = pd.DataFrame(merged).sort_values("_o")
        lines = [f"🚨 <b>要處理</b>（{len(merged)}）"]
        for _, r in merged.iterrows():
            sid = str(r["stock_id"])
            nm = r.get("name") or names.get(sid, "")
            lines.append(
                f"{r['level']} <b>{sid} {nm}</b> [{r['strategy']}] "
                f"{str(r['entry_date'])[5:].replace('-', '/')}進 {_px(r['entry_price'])}→{_px(r['close'])}"
                f"（{r['pnl_pct']:+.1f}%）\n   {r['reason']}"
            )
        blocks.append("\n".join(lines))

    # ── 🛒 新進場候選：合併、每檔一列、動能排序 ─────────────────────────────
    cands = _merge_candidates(signals, exit_ids)
    if not cands.empty:
        cands = _rank_mom(cands, date)
        lines = [f"🛒 <b>新進場候選</b>（{len(cands)}）依動能排序"]
        for i, (_, r) in enumerate(cands.head(MAX_CANDIDATES).iterrows(), 1):
            lines.append(_candidate_line(i, r, names))
        if len(cands) > MAX_CANDIDATES:
            lines.append(f"<i>…另有 {len(cands) - MAX_CANDIDATES} 檔動能較弱，略</i>")
        blocks.append("\n".join(lines))
        long_df = signals.get("long", pd.DataFrame())
        if long_df is not None and not long_df.empty:
            _append_signal_log(long_df[~long_df["stock_id"].astype(str).isin(exit_ids)], date)

    # ── 👀 觀察名單 ────────────────────────────────────────────────────────
    w_lines = _watch_section(signals.get("watch", pd.DataFrame()), names)
    if w_lines:
        blocks.append("\n".join(w_lines))

    if len(blocks) == 1:
        blocks.append("今日無新訊號，持股不動")

    # ── 只在異常時才出現：近期實盤勝率紅燈 ──────────────────────────────────
    recent = _load_recent_log(20)
    if len(recent) >= 10:
        wr = (recent["result"] == "win").mean()
        if wr < WIN_RATE_PAUSE_THR:
            blocks.append(f"🔴 <b>近 {len(recent)} 筆實盤勝率 {wr*100:.0f}%</b>，建議暫停並檢視策略是否失效")

    # ── 📎 參考（脈絡，放最後）──────────────────────────────────────────────
    ref = [x for x in (mget("credit_stress"), mget("telecom_flow")) if x]
    if ref:
        blocks.append("📎 <b>參考</b>\n" + "\n".join(ref))

    return _pack(blocks)


def rules_message() -> str:
    """策略規則對照（置頂用）。數字直接讀 STRATEGIES，避免文字與設定脫節。"""
    from technical.signals import STRATEGIES
    by = {s["timeframe"]: s for s in STRATEGIES}
    size = {"long": "1 單位", "revenue": "1 單位（大盤 60 日 ≤ −5% 時減半）",
            "growth": "S4 的 1/2～2/3", "accum": "S4 的 1/3（大盤跌破 MA60 兩週以上暫停）"}

    def exit_rule(st) -> str:
        sl = f"停損 −{st['default_sl']:.0%}"
        if st.get("trail_trigger"):
            tp = f"漲 +{st['trail_trigger']:.0%} 後從高點回落 {st.get('trail_pct', 0.15):.0%} 出場"
        else:
            tp = f"停利 +{st['default_tp']:.0%}"
        return f"{sl}｜{tp}｜最長 {st['default_hold']} 天"

    lines = ["📌 <b>策略規則對照</b>（每日通知只用標籤）"]
    for key, tag in _TAG_ORDER:
        st = by.get(key)
        if st:
            lines.append(f"\n<b>[{tag}] {st['name']}</b>\n{exit_rule(st)}\n部位：{size[key]}")
    lines += [
        "\n⭐ = 兩個以上策略同時看好（回測只有 S4+S7 證實勝率較高：66% vs 57%）",
        "S4* = 該策略不是今天、而是近 20 個交易日內觸發過",
        "動能 = 近 20 日漲跌；⚠️未表態 = 價格還沒動，可跳過",
        "👀 營收爆發觀察 = 不是進場訊號，題材與時機自己判斷",
        "🚨 出場 / ⚠️ 注意 = 系統發過訊號的股票籌碼惡化（不是你的持股）",
        "\n紀律：連續虧損時不可修改參數。停損是策略的一部分，不是失敗。",
    ]
    return "\n".join(lines)


def send_rules(pin: bool = True) -> bool:
    """發送規則對照並置頂（群組需 bot 有置頂權限；失敗只記 log）。"""
    if not TOKEN or not CHAT_IDS:
        logger.warning("Telegram not configured (TOKEN or CHAT_ID missing)")
        return False
    ok = True
    for chat_id in CHAT_IDS:
        try:
            r = requests.post(f"{API_URL}/sendMessage",
                              json={"chat_id": chat_id, "text": rules_message(), "parse_mode": "HTML"},
                              timeout=10)
            r.raise_for_status()
            if pin:
                mid = r.json()["result"]["message_id"]
                p = requests.post(f"{API_URL}/pinChatMessage",
                                  json={"chat_id": chat_id, "message_id": mid,
                                        "disable_notification": True}, timeout=10)
                if not p.ok:
                    logger.warning(f"Pin failed (chat_id={chat_id}): {p.text[:200]}")
        except Exception as e:
            logger.error(f"Telegram rules send failed (chat_id={chat_id}): {e}")
            ok = False
    return ok


def report_date(signals: dict[str, pd.DataFrame]) -> str:
    """報告日 = 資料的最後交易日（_meta.trade_date），不是程式跑完的時鐘時間。
    GitHub cron 常延遲數小時，10/02 那批跑過午夜被標成「10/03（六）」。"""
    meta = signals.get("_meta")
    if meta is not None and not meta.empty and "trade_date" in meta.columns:
        v = meta.iloc[0]["trade_date"]
        if isinstance(v, str) and v:
            return v
    from datetime import datetime
    return datetime.today().strftime("%Y-%m-%d")


def notify(signals: dict[str, pd.DataFrame]) -> None:
    date = report_date(signals)
    msgs = format_signals(signals, date)
    for msg in msgs:
        send_message(msg)
        time.sleep(0.3)  # Telegram rate limit
