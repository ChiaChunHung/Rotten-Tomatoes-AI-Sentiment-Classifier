# CLAUDE.md

Rotten Tomatoes sentiment classifier: logistic regression written from scratch (`submission.py`), trained on `polarity.train`, evaluated on `polarity.dev`. Originally a stanCode SC201 assignment (`grader.py`, `graderUtil.py`).

## Where the numbers come from

Read `_eval/CLAIMS.md` first. It maps every resume and README number to its source and flags README numbers that the current code does not reproduce.

- `_eval/results.md` is generated output. Don't hand-edit it; regenerate it with `python3 _eval/eval_metrics.py > _eval/results.md` (pure Python, deterministic, about 40 seconds).
- `error-analysis` and `weights` are large generated files. Don't read them whole; grep for specific words instead.
