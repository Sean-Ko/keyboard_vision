# 貢獻指南

這個專案目前採用精簡架構，請盡量保持簡單。

## 修改重點

- OpenCV 偵測核心：`keyboard_key_detection/detector_core.py`
- 主流程與 CLI：`keyboard_key_detection/app.py`
- GUI 人工修正：`keyboard_key_detection/gui_review.py`
- 輸出格式：`keyboard_key_detection/outputs.py`

## 驗證

```powershell
cd keyboard_key_detection
python -m py_compile calibrate_keyboard.py app.py config.py detector.py detector_core.py gui_review.py image_io.py outputs.py
python calibrate_keyboard.py --image "input/white_BIG_clear.png" --no-gui
```

## 不要提交

- `.venv/`
- `.vscode/`
- `__pycache__/`
- `keyboard_key_detection/output/` 內的執行結果

`output/README.md` 可以提交，其他輸出檔不要提交。
