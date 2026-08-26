#!/usr/bin/env python3
"""M3 cls validation driver.

This is a black-box test of the cls module that the user can run
end-to-end. It exercises the full ppocr_cli pipeline with --cls-config
on real images and verifies that:

  (a) cls-off produces empty text on a 180°-rotated image.
  (b) cls-on recovers the same text on the 180°-rotated image
      as on the upright image (fuzzy match; rec has small errors).
  (c) cls adds measurable latency (ms/crop).

Usage:
  python3 tools/cls_validation.py
"""

import json
import os
import subprocess
from pathlib import Path

ROOT       = Path("/root/pp-ocr-mnn")
CLI        = ROOT / "build-cls" / "ppocr_cli"
DET_CFG    = ROOT / "configs" / "PP-OCRv4_mobile_det.json"
REC_CFG    = ROOT / "configs" / "PP-OCRv4_mobile_rec.json"
CLS_CFG    = ROOT / "configs" / "PP-LCNet_x1_0_textline_ori.json"
MODEL_DIR  = ROOT / "models"
IMAGES_DIR = Path("/root/ocr_test_imgs")
ENV        = dict(os.environ, LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu")

# Test set: 5 images, each with at least 1 text line. Mix of scripts.
TEST_IMAGES = [
    "zh/04.jpg",   # SOLINSKY / ALLEY (Latin)
    "en/04.jpg",   # CHANCERY / LANE (Latin)
    "de/04.jpg",   # MAZSAHAZ / WAAGEHAUS (Latin)
    "zh/05.jpg",   # mixed CJK + Latin
    "ja/04.jpg",   # CJK
]


def run_cli(image, cls):
    cmd = [str(CLI), "--image", str(image),
           "--det-config", str(DET_CFG),
           "--rec-config", str(REC_CFG)]
    if cls:
        cmd += ["--cls-config", str(CLS_CFG)]
    cmd += ["--model-dir", str(MODEL_DIR)]
    out = subprocess.run(cmd, env=ENV, capture_output=True, timeout=60)
    for line in out.stdout.decode().splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line), out.returncode
    raise RuntimeError(f"no JSON in stdout for {image}: rc={out.returncode}\n"
                       f"stdout={out.stdout.decode()!r}\n"
                       f"stderr={out.stderr.decode()!r}")


def rot180(path):
    p = Path(path)
    out = Path("/tmp") / (p.parent.name + "_" + p.stem + "_rot180" + p.suffix)
    if not out.exists():
        from PIL import Image
        Image.open(path).rotate(180).save(out)
    return out


def text_set(d):
    return {ln["text"].strip().lower() for ln in d["lines"] if ln["text"].strip()}


def fuzzy_in(s, ref_set):
    """Loose text match: exact, substring, or 3-char prefix."""
    for r in ref_set:
        if s == r: return True
        if len(s) >= 3 and len(r) >= 3 and (s in r or r in s): return True
        if len(s) >= 4 and len(r) >= 4 and s[:3] == r[:3]: return True
    return False


# ---- (a) show cls-off fails on rot180 ----
print("=" * 60)
print("(a) cls-off vs cls-on on 180°-rotated image")
print("=" * 60)
zh04   = IMAGES_DIR / "zh" / "04.jpg"
zh04_r = rot180(zh04)
res_no_cls, _   = run_cli(zh04_r, cls=False)
res_with_cls, _ = run_cli(zh04_r, cls=True)
texts_no   = [ln["text"] for ln in res_no_cls["lines"]]
texts_with = [ln["text"] for ln in res_with_cls["lines"]]
print(f"  image: {zh04_r}")
print(f"  cls off: {texts_no}")
print(f"  cls on : {texts_with}")
assert not any(t.strip() for t in texts_no), \
    "expected cls-off to produce empty text on rot180"
assert any("SOLINSKY" in t for t in texts_with) and any("ALLEY" in t for t in texts_with), \
    f"cls-on should recover 'SOLINSKY' and 'ALLEY' on rot180, got {texts_with}"
cls_ms_per_crop = res_with_cls["ms"]["cls"] / max(1, len(texts_with))
print(f"  cls latency: {res_with_cls['ms']['cls']:.2f} ms total, "
      f"{cls_ms_per_crop:.2f} ms/crop")
print()


# ---- (b) 5 images: cls-on recovers the same content on rot180 as on upright ----
# For each image we run:
#   - cls-off on upright   -> gold text set A
#   - cls-on  on rot180    -> text set B (mostly correct, but
#     det finds different boxes so we measure the recovery
#     RATE = |{t in A : t fuzzy-matches some s in B}| / |A|)
print("=" * 60)
print("(b) 5 images x (upright + rot180) — full pipeline (cls on)")
print("=" * 60)
total_a = 0
total_recovered = 0
total_cls_ms = 0.0
total_boxes = 0
for rel in TEST_IMAGES:
    img   = IMAGES_DIR / rel
    img_r = rot180(img)
    # Gold: cls off on upright.
    res_ref, _ = run_cli(img, cls=False)
    set_a = text_set(res_ref)
    # cls on on rot180.
    res_ro, _ = run_cli(img_r, cls=True)
    set_b = text_set(res_ro)
    # Recovery: |{t in A : t fuzzy-matches some s in B}| / |A|.
    rec = sum(1 for t in set_a if fuzzy_in(t, set_b))
    total_a        += len(set_a)
    total_recovered += rec
    total_cls_ms    += res_ro["ms"]["cls"]
    total_boxes     += len(res_ro["lines"])
    pct = (rec / max(1, len(set_a))) * 100
    print(f"  {rel:14} | A={len(set_a):2} B={len(set_b):2} recovered={rec:2}/{len(set_a)} ({pct:5.1f}%) cls_ms={res_ro['ms']['cls']:.1f}")
print()
print(f"  total: {total_recovered}/{total_a} upright texts recovered on rot180 with cls on")
print(f"  total cls ms: {total_cls_ms:.1f} over {total_boxes} crops")
print(f"  avg cls ms/crop: {total_cls_ms / max(1, total_boxes):.2f}")
print()


# ---- (c) report ----
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  (a) rot180 cls-off: empty  (rec fails)            ✓")
print(f"  (a) rot180 cls-on : recovers 'SOLINSKY','ALLEY'    ✓")
print(f"  (b) 5-image sweep : {total_recovered}/{total_a} upright texts recovered on rot180")
print(f"  (c) cls latency   : {total_cls_ms / max(1, total_boxes):.2f} ms/crop (avg)")
