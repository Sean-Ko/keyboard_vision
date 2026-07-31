# Output

這個資料夾放程式輸出結果。

每張輸入圖片會產生三個檔案：

```text
*_raw.png
*_calibrated.png
*_keys.json
```

- `*_raw.png`：乾淨原圖，未來訓練模型用。
- `*_calibrated.png`：綠點標註圖，給人檢查用。
- `*_keys.json`：訓練友善標註格式，包含原圖尺寸、標註圖路徑、像素座標、正規化座標、來源與信心值。

目前不輸出 CSV，也不輸出 batch report。
