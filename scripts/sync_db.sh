#!/usr/bin/env bash
# 從 GitHub Release 同步 GitHub Actions 上的最新 DB 到本地
# 用法: bash scripts/sync_db.sh
#
# 需要 gh CLI 已登入：gh auth login
set -e

REPO="pamsinna/stockstw"
TAG="db-daily-latest"
DB_PATH="data/cache.db"

cd "$(dirname "$0")/.."

# 備份目前 DB（如果存在）
if [ -f "$DB_PATH" ]; then
    BACKUP="$DB_PATH.before-sync"
    echo "Backing up current DB → $BACKUP"
    cp "$DB_PATH" "$BACKUP"
fi

mkdir -p data

echo "Downloading $TAG from GitHub Release..."
# 先清舊 .gz（如果存在，避免 gh download 寫成 .gz.gz 或 gzip 找不到）
rm -f data/cache.db.gz
gh release download "$TAG" --repo "$REPO" \
    --pattern 'cache.db.gz' \
    --output data/cache.db.gz \
    --clobber

# 驗證下載
if [ ! -s data/cache.db.gz ]; then
    echo "❌ 下載失敗（檔案不存在或為空）。檢查網路或 gh auth status"
    exit 1
fi
echo "  下載完成: $(du -sh data/cache.db.gz | cut -f1)"

echo "Decompressing..."
gunzip -f data/cache.db.gz

echo ""
echo "✅ DB synced from $TAG"
echo "   Size: $(du -sh $DB_PATH | cut -f1)"
# 顯示最新一筆價格日期，驗證 DB 真的有最新資料
python -c "
import sqlite3
con = sqlite3.connect('$DB_PATH')
row = con.execute('SELECT MAX(date) FROM daily_price').fetchone()
print(f'   Latest price date in DB: {row[0]}')
"
