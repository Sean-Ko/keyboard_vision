# Keyboard Vision

Keyboard Vision 是一套鍵盤按鍵中心偵測與標註資料收集工具。它目前使用 Python + OpenCV，先從圖片輪廓找出可能的鍵帽，再用鍵盤排列結構輔助過濾錯誤候選點，最後輸出綠點標註圖片與可供日後訓練使用的 JSON。

## 專案定位

這個專案目前最適合當作：

```text
維修輔助工具 + 標註資料收集工具
```

實際流程：

```text
拍攝鍵盤
-> OpenCV 自動產生初始綠點
-> GUI 人工修正錯誤點
-> 輸出標註圖片與訓練友善 JSON
-> 累積資料後可轉成 YOLO / keypoint / segmentation 訓練資料
```

## 為什麼目前先不用 AI

目前採用 OpenCV 規則演算法，優勢是：

- 可解釋：判斷來自輪廓、尺寸、亮度與排列結構。
- 輕量：不用模型權重，不需要 GPU。
- 容易除錯：錯誤通常能追到特定前處理或過濾規則。
- 適合固定拍攝條件：如果角度與光線穩定，OpenCV 可以先產生可用初始點。
- 方便累積資料：人工修正後的 JSON 可以成為未來訓練資料。

但單純 OpenCV 不應被視為最終泛用解法。不同鍵盤顏色、字體、LED、特殊布局會讓規則越來越複雜。長期更務實的路線是先用本工具累積乾淨標註資料，未來再訓練模型。

## 安裝

```powershell
cd keyboard_key_detection
python -m pip install -r requirements.txt
```

## 執行

選擇 `input/` 裡的圖片：

```powershell
python calibrate_keyboard.py
```

指定圖片並開啟人工微調 GUI：

```powershell
python calibrate_keyboard.py --image "input/white_BIG_clear.png"
```

只跑自動偵測，不開 GUI：

```powershell
python calibrate_keyboard.py --image "input/white_BIG_clear.png" --no-gui
```

批次處理 `input/` 裡全部圖片：

```powershell
python calibrate_keyboard.py --batch-input
```

## 輸出

輸出位置：

```text
keyboard_key_detection/output/
```

每張圖片只輸出兩個檔案：

```text
*_calibrated.png
*_keys.json
```

目前不輸出 CSV，也不輸出 batch report。

## JSON 格式

`*_keys.json` 是資料集標註格式：

```text
schema_version
task
created_at
model_name
image
annotation_summary
annotations[]
```

每個 `annotations[]` 包含：

```text
annotation_id
label
type
point.x / point.y
normalized_point.x / normalized_point.y
source
confidence
template
```

`point` 是原圖像素座標，`normalized_point` 是 0 到 1 的正規化座標。這樣日後比較容易轉成 YOLO、keypoint detection、COCO 或其他訓練格式。

## 專案結構

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

## 演算法流程

```text
input 圖片
-> OpenCV 前處理
-> 輪廓候選鍵帽
-> 尺寸與顏色雜訊過濾
-> 局部排列結構輔助修正
-> GUI 人工微調
-> 輸出標註圖片與資料集 JSON
```
