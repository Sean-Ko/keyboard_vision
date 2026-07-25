# 鍵盤按鍵中心點偵測

這個資料夾是主程式。所有 Python 模組都放在同一層，方便閱讀與修改。

## 安裝

```powershell
python -m pip install -r requirements.txt
```

## 執行

列出 `input/` 圖片並選擇：

```powershell
python calibrate_keyboard.py
```

指定圖片並開啟 GUI：

```powershell
python calibrate_keyboard.py --image "input/logitech k100.jpg"
```

只跑自動偵測，不開 GUI：

```powershell
python calibrate_keyboard.py --image "input/white_BIG_clear.png" --no-gui
```

批次跑全部圖片：

```powershell
python calibrate_keyboard.py --batch-input
```

清理舊輸出：

```powershell
python calibrate_keyboard.py --clean-outputs
```

## GUI 操作

```text
滑鼠拖曳      移動錯誤綠點
A             新增缺少的點
D             刪除選取點
U             復原上一個刪除
R             重設回自動偵測
Enter         儲存
Esc           取消離開
```

## 檔案分工

```text
calibrate_keyboard.py  入口檔
app.py                 CLI 與主流程
config.py              路徑與常數
detector_core.py       OpenCV 核心演算法
detector.py            將核心演算法輸出轉成標準格式
gui_review.py          人工修正 GUI
image_io.py            讀圖與選圖
outputs.py             輸出圖片與 JSON
input/                 放鍵盤圖片
output/                放輸出結果
```

## 輸出

只輸出：

```text
*_calibrated.png
*_keys.json
*_batch_report.json
```
