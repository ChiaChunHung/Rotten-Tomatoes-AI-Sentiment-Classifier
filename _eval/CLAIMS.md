# Where the numbers come from

Every number Tony cites for this project, mapped to its source. Numbers marked **verified** come from `_eval/results.md`, which `_eval/eval_metrics.py` generates. Training has no randomness, so rerunning gives the same numbers:

```bash
python3 _eval/eval_metrics.py > /tmp/check.md   # pure Python, no API key; about 40 seconds
diff _eval/results.md /tmp/check.md            # no output = still matches
```

Last checked 2026-11-11: a fresh rerun matched `results.md` exactly.

Settings: word features, 40 epochs, alpha = 0.01, the same as `grader.py`. Data: `polarity.train` 3,554 + `polarity.dev` 3,554 reviews.

## Resume: "Movie Review Sentiment Analytics & NLP Pipeline"

| Resume claim | Value | Source | Status |
|---|---|---|---|
| 7K+ movie reviews | 7,108 (3,554 train + 3,554 dev) | `results.md` header | verified |
| Logistic regression from scratch (sigmoid, cross-entropy, gradient descent) | `learnPredictor`: h = sigmoid(w·φ(x)), update w += −α(h − y)φ(x), the gradient of cross-entropy loss; no ML library | `submission.py` (~line 56–64) | verified (code) |
| <4% training, <30% validation error | 3.6% train, 26.7% validation (word, 40 epochs) | `results.md` §1, §2 | verified |
| Errors split evenly: 481 FP vs 469 FN, 0.73 F1 | FP 481, FN 469, F1 0.730 | `results.md` §1 | verified |
| Contrastive reviews misread 30% vs 26% | 30.5% (679 reviews with but/yet/though/although/however/despite) vs 25.8% (2,875 others), z ≈ 2.4 | `results.md` §3 | verified |
| Character 5-grams: 0% training error on 3.5K reviews, only matched word features at 0.73 F1 | char 5-gram (20 epochs): 0.0% train, F1 0.733 vs word (40 epochs) 0.730 | `results.md` §2 | verified |

## README.md

Rewritten on 2026-11-12 so every number matches the current code. Where each number comes from:

| README number | Source |
|---|---|
| 96.4% / 73.3% accuracy, F1 0.730, 481 vs 469, 20- and 40-epoch rows, character n-gram comparisons | `results.md` |
| 70- and 80-epoch rows (1.5% / 26.1%, 1.2% / 26.3%) | retrain with `learnPredictor(numEpochs=70 or 80, alpha=0.01)`; not in `results.md` |
| Top words (performances +1.47 … bad -1.75), the "nothing short of a masterpiece" error-analysis sample, example score -0.26 / 43.5% | a fresh `python3 grader.py` run (100/100), which rewrites `weights` and `error-analysis` |
| Sparse vectors: 11,622 words x 3,554 reviews = 41.3M entries vs 66,737 non-zero (about 18.8 per review), 99.84% fewer | counted over `polarity.train` with `extractWordFeatures` |
| Milestone 1 final weights | one pass of logistic SGD, alpha = 0.5, labels {0, 1}; recomputed and matches |
| "well under a second" | scoring all 3,554 dev reviews took about 0.03 s |

Removed because nothing in this folder reproduced them: 96.2% / 70.2% accuracy, masterpiece +2.71 and the old top-word lists, "overfitting at epoch 70 (33.1%)", the old epoch table, 80% workload reduction, 5 min → 1 min per review, the 73.4% confidence and prediction interval.

**Note:** the `weights` and `error-analysis` files currently in this folder are still the old ones, from an older version of `submission.py` (similar word ranking, larger weights). Run `python3 grader.py` to regenerate them so they match the README.
