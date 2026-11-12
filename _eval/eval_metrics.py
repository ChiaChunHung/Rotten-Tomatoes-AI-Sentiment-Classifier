"""
Reproducible evaluation of the Rotten Tomatoes sentiment classifier.

Usage (from anywhere):  python3 _eval/eval_metrics.py > _eval/results.md

Trains submission.learnPredictor with the same settings grader.py uses
(word features, 40 epochs, alpha = 0.01) and reports, on polarity.dev:
  1. confusion matrix, accuracy, precision, recall, F1
  2. a benchmark of word features vs. character n-grams (n = 3-6)
  3. an error slice: reviews with contrast words vs. without
Training has no randomness (fixed data order), so results are identical
on every run. Writes nothing into the repo.
"""
import contextlib, io, os, re, sys
sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import submission, util

with contextlib.redirect_stdout(io.StringIO()):
    TR = util.readExamples(os.path.join(REPO, 'polarity.train'))
    DV = util.readExamples(os.path.join(REPO, 'polarity.dev'))

def train(fe, epochs):
    with contextlib.redirect_stdout(io.StringIO()):
        return submission.learnPredictor(TR, DV, fe, numEpochs=epochs, alpha=0.01)

def predict(examples, fe, w):
    # same decision rule as grader.py: positive only if score > 0
    return [(x, y, 1 if util.dotProduct(fe(x), w) > 0 else -1) for x, y in examples]

def scores(preds):
    tp = sum(1 for _, y, p in preds if y == 1 and p == 1)
    fp = sum(1 for _, y, p in preds if y == -1 and p == 1)
    fn = sum(1 for _, y, p in preds if y == 1 and p == -1)
    tn = sum(1 for _, y, p in preds if y == -1 and p == -1)
    n = tp + fp + fn + tn
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=n, acc=(tp + tn) / n, err=(fp + fn) / n,
                p=prec, r=rec, f1=2 * prec * rec / (prec + rec))

print("# Rotten Tomatoes 情感分類器 — 評估紀錄\n")
print(f"資料：polarity.train {len(TR):,} 則（訓練）＋ polarity.dev {len(DV):,} 則（驗證），共 {len(TR)+len(DV):,} 則。")
print("正類（positive）＝ 正評 (+1)。模型與 grader.py 相同：word features、40 epochs、α = 0.01。\n")

fe_w = submission.extractWordFeatures
w = train(fe_w, 40)
dev = predict(DV, fe_w, w); s = scores(dev); t = scores(predict(TR, fe_w, w))
print("## 1. Confusion matrix（驗證集）\n")
print("| | 模型說正評 | 模型說負評 |\n|---|---|---|")
print(f"| 真的是正評 ({s['tp']+s['fn']:,}) | TP {s['tp']:,} | FN {s['fn']:,} |")
print(f"| 真的是負評 ({s['fp']+s['tn']:,}) | FP {s['fp']:,} | TN {s['tn']:,} |\n")
print(f"- Accuracy {s['acc']:.1%}（validation error {s['err']:.1%}；training error {t['err']:.1%}）")
print(f"- Precision = TP/(TP+FP) = {s['p']:.1%}")
print(f"- Recall = TP/(TP+FN) = {s['r']:.1%}")
print(f"- F1 = 2PR/(P+R) = {s['f1']:.3f}")
print(f"- FP {s['fp']} vs. FN {s['fn']}：錯誤幾乎對稱，模型沒有偏向某一邊。\n")

print("## 2. Benchmark：word features vs. character n-grams\n")
print("| 特徵 | epochs | training error | validation error | validation F1 |\n|---|---|---|---|---|")
rows = [('word', fe_w, 40), ('word', fe_w, 20)] + [(f'char {n}-gram', submission.extractCharacterFeatures(n), 20) for n in (3, 4, 5, 6)]
for name, fe, ep in rows:
    ww = w if (name == 'word' and ep == 40) else train(fe, ep)
    a = scores(predict(TR, fe, ww)); b = scores(predict(DV, fe, ww))
    print(f"| {name} | {ep} | {a['err']:.1%} | {b['err']:.1%} | {b['f1']:.3f} |")
print("\nn ≥ 5 的 training error 降到 0%（等於把訓練集背起來），validation F1 卻沒有比 word features 好 → 過擬合，不是真的進步。\n")

print("## 3. Error slice：含轉折詞的影評\n")
rx = re.compile(r"\b(but|yet|though|although|however|despite)\b")
inn = [d for d in dev if rx.search(d[0])]; out = [d for d in dev if not rx.search(d[0])]
ei = sum(1 for _, y, p in inn if p != y) / len(inn); eo = sum(1 for _, y, p in out if p != y) / len(out)
se = (ei * (1 - ei) / len(inn) + eo * (1 - eo) / len(out)) ** 0.5
print(f"- 含 but / yet / though / although / however / despite：{len(inn):,} 則，錯誤率 {ei:.1%}")
print(f"- 其他：{len(out):,} 則，錯誤率 {eo:.1%}")
print(f"- 差距 {ei-eo:.1%}，z ≈ {(ei-eo)/se:.1f}（以這個樣本數來看不太像隨機誤差）")
print("\n詞袋模型把每個字分開看，「前面誇、後面貶」這種轉折句容易被前半段的正面字帶走。")
