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
gh release download "$TAG" --repo "$REPO" \
    --pattern 'cache.db.gz' \
    --output data/cache.db.gz \
    --clobber

echo "Decompressing..."
gzip -d -f data/cache.db.gz

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
