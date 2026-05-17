#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "======================================"
echo "  Podcast Text Editor 安裝程式"
echo "======================================"
echo ""

# 檢查 Apple Silicon
if [ "$(uname -m)" != "arm64" ]; then
    echo "錯誤：此 app 只支援 Apple Silicon Mac（M1/M2/M3/M4）"
    echo "你的電腦是 Intel Mac，無法使用。"
    read -rp "按 Enter 關閉..."
    exit 1
fi
echo "✓ Apple Silicon 確認"

# 尋找 Python 3.11+
PYTHON=""
for cmd in python3.13 python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "錯誤：找不到 Python 3.11 或更新版本"
    echo "請先到以下網址下載安裝 Python："
    echo "  https://www.python.org/downloads/"
    echo ""
    read -rp "按 Enter 關閉..."
    exit 1
fi
echo "✓ $($PYTHON --version)"

# 建立虛擬環境
if [ ! -d ".venv" ]; then
    echo "→ 建立虛擬環境..."
    "$PYTHON" -m venv .venv
fi
echo "✓ 虛擬環境就緒"

# 安裝套件
echo "→ 安裝套件（首次需要幾分鐘）..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt

# 預先下載 Whisper 模型
echo ""
echo "→ 下載 Whisper 模型（約 800MB，視網路速度需要幾分鐘）..."
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/whisper-large-v3-turbo')
"
echo "✓ 模型下載完成"

# 設定執行權限
chmod +x launch.command

echo ""
echo "======================================"
echo "  安裝完成！"
echo "  請雙擊 launch.command 啟動 app"
echo "======================================"
echo ""
read -rp "按 Enter 關閉..."
