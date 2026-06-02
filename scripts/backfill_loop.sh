#!/usr/bin/env bash
# 連續觸發 daily_screen CI runs 直到 universe stale <= 50
# 用法: bash scripts/backfill_loop.sh
#
# 每跑完一輪會：
#   1. 等 CI 完成
#   2. 同步本地 DB 看 stale 數
#   3. 如果 stale > 50 → 觸發下一輪
#   4. 重複直到 stale ≤ 50 或達 MAX_ROUNDS
set -e

REPO="pamsinna/stockstw"
THRESHOLD=50
MAX_ROUNDS=10
ROUND=0

cd "$(dirname "$0")/.."

while [ $ROUND -lt $MAX_ROUNDS ]; do
    ROUND=$((ROUND + 1))
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "🔁 Round $ROUND / $MAX_ROUNDS"
    echo "═══════════════════════════════════════════════════════════"

    # 觸發新 run
    echo "→ 觸發新 CI run..."
    gh workflow run "Daily Stock Screen" --repo "$REPO"
    sleep 10

    # 找到剛觸發的 run id（最新的 workflow_dispatch）
    RUN_ID=$(gh run list --repo "$REPO" --workflow "Daily Stock Screen" \
             --event workflow_dispatch --limit 1 --json databaseId -q '.[0].databaseId')
    echo "→ Run ID: $RUN_ID"
    echo "→ 等待完成（~1-3 小時）..."

    # 等完成（包含 queued 狀態）
    until [ "$(gh run view "$RUN_ID" --repo "$REPO" --json status -q .status 2>/dev/null)" = "completed" ]; do
        sleep 120
    done

    CONCL=$(gh run view "$RUN_ID" --repo "$REPO" --json conclusion -q .conclusion)
    echo "→ Run 完成: $CONCL"

    if [ "$CONCL" != "success" ]; then
        echo "❌ Run 失敗（$CONCL）。停止 backfill loop。"
        echo "   檢查: gh run view $RUN_ID --repo $REPO --log-failed"
        exit 1
    fi

    # 同步本地 DB
    echo "→ 同步本地 DB..."
    bash scripts/sync_db.sh > /dev/null 2>&1

    # 算 stale
    STALE=$(python3 -c "
import sqlite3
from datetime import date, timedelta
con = sqlite3.connect('data/cache.db')
cutoff = (date.today() - timedelta(days=7)).isoformat()
n = con.execute('''SELECT COUNT(*) FROM stock_universe u
                   WHERE NOT EXISTS (SELECT 1 FROM fetch_log f
                                     WHERE f.stock_id=u.stock_id AND f.dataset='price'
                                       AND f.last_date >= ? AND f.last_date < '9999-01-01')''',
                (cutoff,)).fetchone()[0]
print(n)
")
    echo "→ Universe stale: $STALE (門檻 $THRESHOLD)"

    if [ "$STALE" -le "$THRESHOLD" ]; then
        echo ""
        echo "✅ 完成！stale=$STALE ≤ $THRESHOLD"
        exit 0
    fi
done

echo ""
echo "⚠️ 達到 MAX_ROUNDS=$MAX_ROUNDS 仍有 $STALE 支 stale，停止。"
echo "   可重新執行此腳本繼續，或等 daily cron 自然補。"
