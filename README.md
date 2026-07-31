# Keyboard Vision

Keyboard Vision 是一套鍵盤維修輔助、按鍵中心標註與未來機台自動化資料收集工具。它目前使用 Python + OpenCV 自動偵測鍵盤按鍵中心，搭配 GUI 人工修正，輸出可供維修追溯與未來模型訓練使用的資料。

## 專案定位

目前定位不是「完全取代人工」的最終 AI 系統，而是：

```text
維修輔助工具 + 標註資料收集工具 + 未來三軸機台控制前置系統
```

建議完整流程：

```text
固定相機拍攝鍵盤原圖
-> OpenCV 自動產生初始綠點
-> 維修師用 GUI 修正錯誤點
-> 輸出 raw / calibrated / JSON 三件套
-> JSON 座標轉換成機台座標
-> 三軸機台或機械手臂自動按鍵
-> 累積原圖與正確答案供未來 YOLO / keypoint model 訓練
```

## 為什麼目前先用 OpenCV

OpenCV 規則演算法的優勢：

- 可解釋：錯誤通常能追到輪廓、尺寸、亮度或排列規則。
- 輕量：不用模型權重，不需要 GPU。
- 適合固定拍攝條件：維修工作室可以固定相機、燈光與鍵盤平台。
- 適合產生初始標註：不用從零手動點每顆按鍵。
- 適合累積資料：人工修正後的 JSON 可以變成未來訓練資料。

長期來看，如果要支援大量不同鍵盤、不同顏色、不同布局，單純 OpenCV 會越來越吃力。比較務實的路線是先用本工具累積乾淨資料，再訓練 YOLO、keypoint detection 或 segmentation 模型。

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

## 輸出資料

輸出位置：

```text
keyboard_key_detection/output/
```

每次處理一張圖片會輸出三個檔案：

```text
*_raw.png
*_calibrated.png
*_keys.json
```

- `*_raw.png`：乾淨原圖，未來訓練模型用。
- `*_calibrated.png`：綠點標註圖，給維修師檢查用。
- `*_keys.json`：按鍵中心座標與資料集標註資訊。

目前不輸出 CSV，也不輸出 batch report。

## JSON 格式

`*_keys.json` 是訓練友善格式：

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

`point` 是原圖像素座標，`normalized_point` 是 0 到 1 的正規化座標。日後可以轉成 YOLO、keypoint detection、COCO 或其他訓練格式。

## 實體機台發展方向

建議第一版先做三軸機台，而不是機械手臂：

```text
X 軸：左右移動
Y 軸：前後移動
Z 軸：下壓按鍵
```

核心模組：

```text
相機拍照
鍵盤定位治具
OpenCV 按鍵中心偵測
GUI 人工確認
image px -> machine mm 座標校正
三軸平台運動控制
按鍵觸發結果紀錄
資料集保存
```

最重要的是座標轉換：

```text
image_x, image_y -> machine_x, machine_y
```

建議用平台上固定的 4 個校正點建立 homography 或 affine transform，讓圖片座標能轉成機台實體座標。

## 專案結構

```text
keyboard_vision/
  .github/
  .gitignore
  .gitattributes
  CONTRIBUTING.md
  README.md
  docs/
    MACHINE_ARCHITECTURE.md
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

## GitHub 命名建議

如果專案未來會包含視覺、維修、機台控制與資料集，建議名稱不要只叫 `keyboard_vision`。更好的名稱：

```text
keyboard-repair-automation
```

其他可選：

```text
keyboard-vision-lab
keyboard-keypoint-dataset
keyboard-test-rig
keyboard-repair-vision
```

我最建議 `keyboard-repair-automation`，因為它涵蓋維修、視覺辨識、未來機台按鍵測試與資料收集。
