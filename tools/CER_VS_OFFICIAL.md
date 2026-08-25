# CER scoring semantics: our `tools/score.py` vs PaddleOCR `RecMetric`

This is a **review document**, not a change request. The author of `score.py`
(TOOLS) writes a detailed comparison so the decision-maker can decide
whether the contract needs an update. **No code in `score.py` or
`run_reference.py` is being changed in this PR** per the task brief.

## TL;DR

`tools/score.py` CER is **not identical** to PaddleOCR's `RecMetric` CER.
The four differences (in increasing order of impact on numbers):

| # | Aspect | `tools/score.py` | PaddleOCR `RecMetric` | Effect |
|---|---|---|---|---|
| 1 | per-line vs per-image | one CER per **image** (joined text) | one CER per **line** (each rec_texts element) | bigger samples; smoother mean |
| 2 | space handling | keeps spaces in both pred and target | `ignore_space=True` strips all spaces by default | strict if GT has spaces; soft if pred has them |
| 3 | case handling | case-sensitive | case-sensitive (`is_filter=False` default) | identical |
| 4 | normalization | `lev / len(target)` | `lev / max(len(pred), len(target))` (rapidfuzz `normalized_distance`) | strict if pred is short; soft if pred is long |

All four are *defensible* design choices. They are **not bugs** — but the
contract should be aware of them. The biggest practical impact is #4:
`score.py` treats a short prediction that misses characters as **worse**
than official PaddleOCR scoring does, because official divides by
`max(len_p, len_t)` (which can be the longer pred) while we divide by
`len_t` only.

## Reference: PaddleOCR `RecMetric` (verbatim)

From `ppocr/metrics/rec_metric.py` in the cloned PaddleOCR tree:

```python
class RecMetric(object):
    def __init__(self, main_indicator="acc", is_filter=False,
                 ignore_space=True, **kwargs):
        ...
    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        norm_edit_dis = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)   # digits+letters, lowercase
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
        return {"acc": ..., "norm_edit_dis": 1 - norm_edit_dis/(n+eps)}
```

`Levenshtein.normalized_distance` is the `rapidfuzz.distance` definition:

```
normalized_distance(a, b) = levenshtein(a, b) / max(len(a), len(b))
```

There is also `e2e_metric.py` which scores the **det+rec pipeline jointly**
using polygon IoU + Hungarian matching + the per-line `RecMetric`. Our
`run_reference.py` / `score.py` are *not* doing E2E: we score only the
joined text per image, ignoring box geometry. That is by design — the
baseline for our 811-cell matrix is exactly a list of `{rec_texts,
rec_scores, det_polys}` per image, and the cell-level pass condition is
CER ≤ 0.05 on the joined text, per the contract.

## Side-by-side worked example

For the PP-OCRv4 mobile-det + v4 mobile-rec / zh / 03.jpg baseline:

- baseline `rec_texts` (joined) = `QUEEN VICTORIA ST\n域多利皇\nQUEEN VICTOR\n街后皇利域\n路牌最初位放\n为制`
- baseline `rec_scores` per line ≈ `[0.93, 1.00, …]`

If we hypothetically had a 1-char deletion in the third line
("QUEEN VICTOR" → "QUEEN VICTOR"), then for this image:

| formula | result |
|---|---|
| `score.py` (per-image join, `lev/len(target)`) | `lev=1, len_t=46 → 0.0217` |
| `RecMetric` per-line mean (`normalized_distance`, ignore_space) | line 3 ≈ `1/12 ≈ 0.083`; mean of 6 lines ≈ `0.01–0.03` |

So a one-character error is similar under both formulas, because the
denominators (image length vs line length) are both in the 10–20 char
range, giving a per-element CER in the 0.05–0.1 ballpark.

If we had a **shorter** prediction (e.g. a 5-char total: only "QUEEN"):

| formula | result |
|---|---|
| `score.py` | `lev=many, len_t=46 → 0.89` |
| `RecMetric` mean of line-CERs | each line ≈ `1.0` (since pred is empty line); `mean ≈ 1.0` |

In practice, for partial-detection scenarios the two formulas agree that
the result is bad, but disagree on the **number**. Numbers from `score.py`
will tend to look **pessimistic for the easy cases** (1-line edits look
huge when divided into a 46-char join) and **optimistic for the
catastrophic cases** (one missing line in a 10-line page makes us divide
by `len_t` of 90 not by `max(len_p, 90)`).

## Recommendations (NOT IMPLEMENTED)

1. **Keep the current `score.py` formula as the contract default.** It
   is documented, simple, and gives pass/fail at the right granularity
   for a multi-line image. The 0.05 threshold is calibrated against this
   formula by the decision-maker (look at `manifest_full.json`).

2. **If we want PaddleOCR-compatible numbers** (e.g. for cross-publishing
   accuracy to a paper), add a *secondary* output:
   `--rec-metric` mode in `score.py` that:
   - iterates `rec_texts` element-by-element (not joined);
   - applies `ignore_space` (default on);
   - uses `lev / max(len_p, len_t)`.
   This is **additive** and would not change the existing cell-pass
   decisions (CER 0.05 is a rough threshold either way).

3. **If we want full E2E metric** (det+rec joint): that's a much bigger
   change (Hungarian matching on polygons, IoU thresholds, "ignore"
   tags). Out of scope for the tools team; should be a dedicated
   decision-maker-approved milestone.

4. **The audit tool `tools/cer_audit.py`** uses our formula. Its
   result is consistent with the rest of the pipeline. The decision-
   maker should look at it to confirm baseline *format* consistency
   (which it does, via the self-check) and not at the absolute CER
   number (which is meaningless vs Wikimedia-metadata "GT").

## How to verify the formula difference locally

```sh
cd /root/pp-ws/tools
python3 -c "
import sys; sys.path.insert(0, 'tools')
from score import levenshtein
# Case A: pred shorter than target (missed characters) — formulas agree
pred, target = 'QUEEN', 'QUEEN VICTORIA ST'  # 5 vs 18
print('A: ours =', levenshtein(pred,target)/len(target),
      'official =', levenshtein(pred,target)/max(len(pred),len(target)))
# Case B: pred longer than target (spurious chars) — official is softer
pred, target = 'QUEEN VICTORIA STEE', 'QUEEN VICTORIA ST'  # 18 vs 18
print('B: ours =', levenshtein(pred,target)/len(target),
      'official =', levenshtein(pred,target)/max(len(pred),len(target)))
# Case C: with spaces — ignoring spaces halves the number
pred, target = 'HELLO WORLD', 'HELLO'
print('C spaces kept:    ours =', levenshtein(pred,target)/len(target))
p2, t2 = pred.replace(' ',''), target.replace(' ','')
print('C spaces stripped: ours =', levenshtein(p2,t2)/max(len(p2),len(t2)))
"
```

Output:
```
A: ours = 0.7059     official = 0.7059
B: ours = 0.1176     official = 0.1053
C spaces kept:    ours = 1.2
C spaces stripped: ours = 0.5
```

So in case **A** the two formulas agree (the denominator is the longer
side — the target). In case **B** the official is slightly softer
(`max(18,18) = 18` here, so they should agree; the small gap is because
I picked an example where `len_p > len_t` only by 1). In case **C**
dropping spaces halves the CER.

## What this PR does NOT change

- `tools/score.py` (formula, schema, exit code).
- `tools/run_reference.py` (pred.json schema).
- `docs/CONTRACT.md` (acceptance threshold).

These are contract-frozen per task brief. The author of this document
recommends the decision-maker add a clause to `docs/CONTRACT.md` *if* the
formula choice is to be made explicit (e.g. "per-image join,
`lev/len(target)`, case-sensitive, spaces preserved"). That is a
decision-maker call.
