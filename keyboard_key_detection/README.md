# Keyboard Key Detection

這個資料夾是主程式所在位置。入口檔案是 `calibrate_keyboard.py`，實際 CLI 流程在 `app.py`，核心影像演算法在 `detector_core.py`。

目前定位是「維修輔助 + 標註資料收集」。OpenCV 會先產生初始綠點，使用者可用 GUI 修正，最後輸出圖片與訓練友善 JSON，日後可轉成模型訓練資料。

## 安裝

```powershell
python -m pip install -r requirements.txt
```

## 執行

從 `input/` 清單選圖：

```powershell
python calibrate_keyboard.py
```

指定圖片並開啟 GUI：

```powershell
python calibrate_keyboard.py --image "input/logitech k100.jpg"
```

指定圖片但不開 GUI：

```powershell
python calibrate_keyboard.py --image "input/white_BIG_clear.png" --no-gui
```

批次處理全部輸入圖片：

```powershell
python calibrate_keyboard.py --batch-input
```

清理舊輸出：

```powershell
python calibrate_keyboard.py --clean-outputs
```

## GUI 操作

```text
滑鼠拖曳      移動錯誤的點
A             新增一個點
D             刪除選取點
U             復原刪除
R             重設本次修正
Enter         儲存
Esc           取消
```

## 模組說明

```text
calibrate_keyboard.py  主程式入口
app.py                 指令列流程
config.py              專案路徑設定
detector_core.py       OpenCV 核心偵測演算法
detector.py            將核心演算法輸出轉成標準座標資料
gui_review.py          人工微調 GUI
image_io.py            圖片讀取與選圖
outputs.py             輸出圖片與資料集 JSON
input/                 放待偵測圖片
output/                放輸出結果
```

## 輸出

每張圖片只輸出：

```text
*_calibrated.png
*_keys.json
```

`*_calibrated.png` 是綠點標註圖片。  
`*_keys.json` 是資料集標註 JSON。

目前不輸出 CSV，也不輸出 batch report。

## JSON 結構

```text
schema_version          標註格式版本，例如 keyboard_keypoints.v1
task                    任務名稱
created_at              建立時間
model_name              使用者指定的模型/資料集名稱
image                   原圖與標註圖資訊
annotation_summary      標註總覽
annotations             每顆按鍵中心點
```

每個 `annotations[]`：

```text
annotation_id           標註 ID
label                   標籤名稱
type                    key_center
point                   像素座標
normalized_point        0 到 1 正規化座標
source                  auto_opencv / manual_fine_tuned / manual_added
confidence              信心值
template                目前的 row/col 輔助資訊
```

這種格式保留了維修時要看的座標，也保留了未來訓練模型需要的原圖尺寸、正規化座標與人工修正來源。

## 演算法設計

目前採用非 AI 的 OpenCV 流程：

```text
圖片前處理
-> 找輪廓候選鍵帽
-> 過濾 LED、文字、外殼雜訊
-> 用局部排列結構輔助判斷
-> 保留方向鍵、導覽鍵等非主鍵區的小鍵群
-> 需要時由使用者做少量人工微調
```

OpenCV 在這個階段的價值是快速產生初始標註，減少人工從零點選的工作量。如果未來累積足夠多「原圖 + 修正後 JSON」，可以再把資料轉成 YOLO、keypoint detection 或 segmentation 訓練格式。
