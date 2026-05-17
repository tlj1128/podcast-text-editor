# Podcast Text Editor

個人 Podcast 後製輔助工具，透過 AI 語音辨識自動產生逐字稿，並提供波形編輯介面快速標記靜音、刪除區段，最終匯出剪輯後的音檔。

## 功能

- 使用 Whisper（mlx-whisper）進行語音轉文字，支援自訂提示詞提升辨識率
- 波形視覺化，可拖曳標記刪除 / 靜音區間
- VAD（靜音偵測）自動標記非語音區段
- 逐字稿點擊同步波形，支援段落文字編輯
- 匯出剪輯後音檔（.m4a）、逐字稿（TXT / SRT）
- Session 自動儲存，下次開啟自動還原進度

## 系統需求

- **Apple Silicon Mac**（M1 / M2 / M3 / M4）
- macOS 13 Ventura 以上
- Python 3.11 或更新版本

> Intel Mac 不支援，因為語音辨識引擎（mlx-whisper）僅支援 Apple Silicon。

## 安裝

### 步驟一：安裝 Python 3.11+

若尚未安裝，請至 [python.org](https://www.python.org/downloads/) 下載安裝。

### 步驟二：下載此專案

點擊頁面右上角 **Code → Download ZIP**，解壓縮到任意位置。

### 步驟三：執行安裝腳本

在 Finder 中，對 `install.command` 按右鍵 → **開啟**（首次需要這樣繞過 Gatekeeper）。

腳本會自動：
1. 建立 Python 虛擬環境
2. 安裝所有相依套件
3. 下載 Whisper 模型（約 800MB，需要幾分鐘）

### 步驟四：啟動 App

安裝完成後，雙擊 `launch.command` 即可啟動。

## 使用方式

1. **檔案 → 開啟音檔**，選擇 .m4a / .mp3 / .wav
2. **檔案 → 轉錄**，等待 AI 產生逐字稿
3. 在波形或逐字稿上右鍵標記要刪除或靜音的區段
4. **檔案 → 匯出音檔**，輸出剪輯結果

## 授權

MIT License
