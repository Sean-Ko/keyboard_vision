# 實體機台架構規劃

這份文件描述 Keyboard Vision 後續接三軸機台或機械手臂的建議方向。

## 建議先做三軸機台

鍵盤自動按鍵測試本質上比較像 CNC 或 3D 印表機，不需要一開始就用多自由度機械手臂。

```text
X 軸：左右移動
Y 軸：前後移動
Z 軸：下壓按鍵
```

三軸機台的優勢：

- 結構簡單。
- 座標比較穩定。
- 校正比較容易。
- 成本較低。
- 適合大量重複按鍵測試。

## 硬體組成

```text
固定上視相機
固定光源
鍵盤定位治具
XY 龍門平台
Z 軸按壓機構
軟膠或矽膠按壓頭
彈簧或壓力限制結構
控制器
```

按壓頭不建議用硬金屬直接接觸鍵帽，避免刮傷或壓壞鍵帽。

## 軟體流程

```text
拍攝 raw image
-> OpenCV 偵測按鍵中心
-> GUI 人工確認/修正
-> 儲存 raw image、calibrated image、keys.json
-> 建立 image px 到 machine mm 的座標轉換
-> 產生按鍵測試路徑
-> 三軸機台依序按鍵
-> 記錄按鍵觸發結果
```

## 座標校正

圖片座標不能直接給機台使用，需要轉換：

```text
image_x, image_y -> machine_x, machine_y
```

建議平台設計 4 個固定校正點，例如左上、右上、左下、右下。程式可以用 OpenCV 的 homography 或 affine transform，把圖片座標轉成機台座標。

## 資料保存

每次維修或測試建議保存：

```text
*_raw.png
*_calibrated.png
*_keys.json
```

其中 `*_raw.png` 是未來訓練模型最重要的圖片，`*_calibrated.png` 給人檢查，`*_keys.json` 是正確答案。

## 後續可增加的欄位

未來接上機台後，JSON 可以增加：

```text
machine_point.x
machine_point.y
press_depth_mm
press_force_g
press_duration_ms
test_result
tested_at
operator
device_id
```

這樣同一份資料可以同時支援維修紀錄、機台控制與模型訓練。
