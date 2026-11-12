# Rotten Tomatoes 情感分類器 — 評估紀錄

資料：polarity.train 3,554 則（訓練）＋ polarity.dev 3,554 則（驗證），共 7,108 則。
正類（positive）＝ 正評 (+1)。模型與 grader.py 相同：word features、40 epochs、α = 0.01。

## 1. Confusion matrix（驗證集）

| | 模型說正評 | 模型說負評 |
|---|---|---|
| 真的是正評 (1,751) | TP 1,282 | FN 469 |
| 真的是負評 (1,803) | FP 481 | TN 1,322 |

- Accuracy 73.3%（validation error 26.7%；training error 3.6%）
- Precision = TP/(TP+FP) = 72.7%
- Recall = TP/(TP+FN) = 73.2%
- F1 = 2PR/(P+R) = 0.730
- FP 481 vs. FN 469：錯誤幾乎對稱，模型沒有偏向某一邊。

## 2. Benchmark：word features vs. character n-grams

| 特徵 | epochs | training error | validation error | validation F1 |
|---|---|---|---|---|
| word | 40 | 3.6% | 26.7% | 0.730 |
| word | 20 | 7.4% | 27.7% | 0.715 |
| char 3-gram | 20 | 1.2% | 31.0% | 0.683 |
| char 4-gram | 20 | 0.1% | 27.5% | 0.721 |
| char 5-gram | 20 | 0.0% | 26.5% | 0.733 |
| char 6-gram | 20 | 0.0% | 27.3% | 0.725 |

n ≥ 5 的 training error 降到 0%（等於把訓練集背起來），validation F1 卻沒有比 word features 好 → 過擬合，不是真的進步。

## 3. Error slice：含轉折詞的影評

- 含 but / yet / though / although / however / despite：679 則，錯誤率 30.5%
- 其他：2,875 則，錯誤率 25.8%
- 差距 4.6%，z ≈ 2.4（以這個樣本數來看不太像隨機誤差）

詞袋模型把每個字分開看，「前面誇、後面貶」這種轉折句容易被前半段的正面字帶走。
