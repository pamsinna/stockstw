#!/bin/bash
# 一鍵建立本地開發虛擬環境
set -e

echo "建立虛擬環境 .venv ..."
python3 -m venv .venv

echo "啟動虛擬環境 ..."
source .venv/bin/activate

echo "安裝依賴 ..."
pip install --upgrade pip
pip install -r requirements.txt

echo "複製 .env 範本 ..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "請編輯 .env 填入 token"
fi

echo ""
echo "完成！之後每次開發前執行："
echo "  source .venv/bin/activate"
echo ""
echo "第一步（下載所有歷史資料，約需 1-2 小時）："
echo "  python main.py download"
echo ""
echo "第二步（跑回測，看策略驗證結果）："
echo "  python main.py backtest"
echo ""
echo "第三步（每日選股）："
echo "  python main.py screen"
