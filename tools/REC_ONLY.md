# Rec-only mode: contract gap with the live `ppocr_cli`

**Status:** NOT YET SUPPORTED. The CLI in `apps/ppocr_cli.cpp` does not
have a `--rec-only` flag, and the underlying C ABI (`include/ppocr/ppocr.h`)
requires `det_name` to be set in `ppocr_config`.

## What this affects

Strip cells (`/root/ppocr_reference/strip__<rec>/<lang>/ocr_results.json`)
are rec-only baselines. The image is already a single-line text crop
(e.g. `/root/ocr_test_imgs/ta/00_0.jpg` is 200×64). We want to:
1. Skip det entirely.
2. Run only the rec model on the image.
3. Compare with the per-image `gt_text` in the baseline.

## What the live CLI does today

From `apps/ppocr_cli.cpp` `parse_args()`:

```cpp
if (k == "--image")        { a.image = v; }
else if (k == "--det-config")    { a.det_config = v; }
else if (k == "--rec-config")    { a.rec_config = v; }
...
```

Then in `main()`:
```cpp
if (a.image.empty() || a.det_config.empty()) {
  std::fprintf(stderr, "error: --image and --det-config are required\n");
  ...
  return 1;
}
```

`--rec-config` is accepted but `--det-config` is mandatory. From the
C ABI (`include/ppocr/ppocr.h`):

```c
typedef struct ppocr_config {
  ...
  const char* det_name;       // det model name in registry (default "PP-OCRv6_tiny_det")
  const char* rec_name;       // rec model name (default "PP-OCRv6_tiny_rec"; NULL disables rec → det-only)
  ...
} ppocr_config;
```

`rec_name = NULL` is the **det-only** path. There is no symmetric
det_name = NULL → rec-only path; the field is just required to resolve
the model's preprocess constants (resize mode, max_side).

## Two options to close the gap (decision-maker call)

### Option A: extend the CLI with `--rec-only`

Add to `parse_args()`:
```cpp
else if (k == "--rec-only") { a.rec_only = 1; }
```

In `main()`, when `rec_only`:
- Skip the `det_config.empty()` check (or make `--det-config` optional).
- Pass `det_name = NULL` to `ppocr_config` (or a dummy "rec-only" marker).
- Skip `ppocr_run_file`'s det call; only run the rec model on the full
  image as if it were a single line.

Pros: small change, fits the existing CLI shape.
Cons: the engine currently has no rec-only path; the det pre/post pipeline
has to be bypassed cleanly. May need a new `ppocr_run_file_rec` API in
`include/ppocr/ppocr.h`.

### Option B: a separate rec-only binary

`tools/rec_only_cli` that wraps `prep_rec_line` + `ctc_decode` directly,
without going through `ppocr_create`. Smaller, but duplicates the CLI.

Pros: clean separation; no impact on the main `ppocr_cli` shape.
Cons: more surface to maintain.

## What `tools/run_reference.py` does today

`run_reference.py --rec-only` flag exists. The implementation
(`_run_one_rec`) tries to call the CLI without `--det-config`, which
fails (the CLI returns rc=1 with "error: --image and --det-config are
required"). The driver then writes a synthetic error entry per image:

```json
{
  "image_path": "/root/ocr_test_imgs/ta/00_0.jpg",
  "rec_texts": [],
  "rec_scores": [],
  "det_polys": [],
  "error": "rec_only_unsupported",
  "detail": "ppocr_cli does not support --rec-only (rc=1). See tools/REC_ONLY.md. ..."
}
```

so the downstream `score.py --strip` can still run, see the failure,
and report a known limitation.

## What m1 needs to do

Recommended patch (Option A):

```diff
--- a/apps/ppocr_cli.cpp
+++ b/apps/ppocr_cli.cpp
@@ -Args@@
   int det_only = 0;
+  int rec_only = 0;
   int time    = 0;
@@ -parse_args()@@
   else if (k == "--det-only")      { a.det_only = 1; }
+  else if (k == "--rec-only")      { a.rec_only = 1; }
   else if (k == "--max-side")      { ... }
@@ -main()@@
-  if (a.image.empty() || a.det_config.empty()) {
+  if (a.image.empty() ||
+      (a.det_config.empty() && !a.rec_only)) {
     std::fprintf(stderr, "error: --image and --det-config are required\n");
     return 1;
   }
@@ -main()@@
-  const std::string det_name = config_basename(a.det_config);
+  const std::string det_name = a.det_config.empty()
+                                   ? std::string{}
+                                   : config_basename(a.det_config);
   const std::string rec_name = (a.det_only || a.rec_config.empty())
                                    ? std::string{}
                                    : config_basename(a.rec_config);
```

The actual `ppocr_create` / `ppocr_run_file` plumbing for the rec-only
path (loading only the rec model, skipping the det call, returning
one line per image) is a slightly bigger change — it will need a new
C API entry point, e.g. `ppocr_run_file_rec(engine, path, &result)`.

## Verification once m1 ships it

1. `python3 tools/run_reference.py --rec-only --only-combo strip__ta_PP-OCRv5_mobile_rec --langs ta`
2. Check `results/strip__ta_PP-OCRv5_mobile_rec/ta/pred.json`:
   - each entry has `rec_texts: [<one str>]` and `error: null/absent`
3. `python3 tools/score.py --strip` reports CER for the strip cell
   instead of synthetic errors.
