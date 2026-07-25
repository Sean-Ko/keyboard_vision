# Keyboard Vision

這是一個精簡版鍵盤按鍵中心點偵測專案。

目前只保留：

- OpenCV 演算法。
- GUI 人工檢查。
- `input/` 圖片輸入。
- `output/` 圖片與 JSON 輸出。
- GitHub 協作設定。

已移除：

- YOLO / SAM / AI 相關內容。
- feedback 學習系統。
- 自動迭代學習系統。
- CSV 輸出。
- 舊版巢狀模組資料夾。
- 大量歷史輸出與本機虛擬環境。

## 快速開始

```powershell
cd keyboard_key_detection
python -m pip install -r requirements.txt
python calibrate_keyboard.py
```

指定圖片：

```powershell
python calibrate_keyboard.py --image "input/white_BIG_clear.png"
```

不開 GUI：

```powershell
python calibrate_keyboard.py --image "input/white_BIG_clear.png" --no-gui
```

## 目前結構

```text
keyboard_vision/
  .github/
  .gitignore
  .gitattributes
  CONTRIBUTING.md
  README.md
  keyboard_key_detection/
    app.py
    calibrate_keyboard.py
    config.py
    detector.py
    detector_core.py
    gui_review.py
    image_io.py
    outputs.py
    requirements.txt
    README.md
    input/
    output/
```

## 核心流程

```text
input 圖片
-> detector_core.py 使用 OpenCV 找按鍵候選
-> detector.py 轉成標準 key records
-> gui_review.py 可人工修正
-> outputs.py 輸出標註圖片與 JSON
```

## 輸出

只輸出：

```text
*_calibrated.png
*_keys.json
*_batch_report.json
```

不輸出 CSV。
