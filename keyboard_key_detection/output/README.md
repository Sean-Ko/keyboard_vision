# Output

這個資料夾放程式輸出結果。

每張輸入圖片只會產生兩個檔案：

```text
*_calibrated.png
*_keys.json
```

目前不輸出 CSV，也不輸出 batch report。

`*_keys.json` 是訓練友善標註格式，包含原圖尺寸、標註圖路徑、像素座標、正規化座標、來源與信心值。
