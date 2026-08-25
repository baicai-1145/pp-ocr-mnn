#!/usr/bin/env python3
"""Empirical baseline detector geometry probe.

For every (det model, image) cell, instantiate the model via the
PaddleOCR API the baselines were generated with (see
`tools/gen_parallel.py:run_cell`), then print the effective PaddleX
detection config (`limit_side_len`, `limit_type`, `max_side_limit`,
`thresh`, `box_thresh`, `unclip_ratio`) and the actual resize shape
PaddleX produces for the given input image.

This is the **geometric contract** the C++ pre/postprocess must
reproduce.  v3 PaddleOCR used `limit_min 736` everywhere; v3
PaddleX `OCR.yaml` (the runtime used to regenerate the 811-cell
baseline) has a **single unified config for all 7 det models**:
`limit_side_len: 64, limit_type: min, max_side_limit: 4000,
thresh: 0.3, box_thresh: 0.6, unclip_ratio: 1.5`. The per-model
`inference.yml` `DetResizeForTest` and `PostProcess` values are
ignored by PaddleX at runtime (PaddleX overrides them with the
pipeline defaults).

This script was added in M2-FIX after the M2-PIPE commit's CER
audit traced the high-CER root cause to a geometry mismatch: M1
prep_det used `limit_min 736` (PaddleOCR v3 reference), but the
PaddleX baseline generator uses `limit_min 64`. Same input
1280x720 → M1 feeds 1280x736 → baseline fed 1280x704 → 32px in H
→ different DB contours → different boxes → different rec crops.
"""
from __future__ import annotations
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

PPOCR_MODELS = "/root/ppocr_models"
IMG_DIR = "/root/ocr_test_imgs"
PROBE_IMG = "zh/04.jpg"   # 1280x720; canonical street-sign
DETS = [
    "PP-OCRv4_mobile_det",
    "PP-OCRv4_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv5_server_det",
    "PP-OCRv6_tiny_det",
    "PP-OCRv6_small_det",
    "PP-OCRv6_medium_det",
]


def probe(det_name: str, img_path: str) -> dict:
    """Instantiate the det model via PaddleOCR, dump its effective
    det config and the resize shape it would produce for `img_path`."""
    from paddleocr import PaddleOCR
    import numpy as np
    import cv2

    o = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name=det_name,
        text_detection_model_dir=os.path.join(PPOCR_MODELS, det_name),
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        text_recognition_model_dir=os.path.join(PPOCR_MODELS, "PP-OCRv5_mobile_rec"),
    )
    ip = o.paddlex_pipeline._pipeline
    detm = ip.text_det_model
    res_op = detm.pre_tfs["Resize"]
    img = cv2.imread(img_path)
    ih, iw = img.shape[:2]
    # Resize the image as the PaddleX pipeline would.
    resized, shape = res_op.resize(
        img, detm.limit_side_len, detm.limit_type, detm.max_side_limit
    )
    rh, rw = resized.shape[:2]
    # Real PaddleX predict; print the polys it actually emits.
    res = o.predict(img_path)
    info = res[0] if res else {}
    polys = info.get("rec_polys", info.get("rec_boxes", [])) or []
    polys_flat = []
    for p in polys:
        a = np.array(p).flatten().tolist()
        polys_flat.append([round(x, 1) for x in a])
    return {
        "det": det_name,
        "image": os.path.relpath(img_path, IMG_DIR),
        "image_w": iw,
        "image_h": ih,
        "paddlex_pipeline": {
            "limit_side_len": ip.text_det_limit_side_len,
            "limit_type": ip.text_det_limit_type,
            "max_side_limit": ip.text_det_max_side_limit,
            "thresh": ip.text_det_thresh,
            "box_thresh": ip.text_det_box_thresh,
            "unclip_ratio": ip.text_det_unclip_ratio,
        },
        "predictor": {
            "limit_side_len": detm.limit_side_len,
            "limit_type": detm.limit_type,
            "max_side_limit": detm.max_side_limit,
            "thresh": detm.thresh,
            "box_thresh": detm.box_thresh,
            "unclip_ratio": detm.unclip_ratio,
        },
        "resize_op": {
            "resize_type": res_op.resize_type,
            "limit_side_len": res_op.limit_side_len,
            "limit_type": res_op.limit_type,
            "resize_long": getattr(res_op, "resize_long", None),
        },
        "resize_out_WxH": [rw, rh],
        "ratio_w_h": [float(shape[3]), float(shape[2])],
        "n_polys": len(polys),
        "polys": polys_flat,
    }


def main():
    results = []
    img = os.path.join(IMG_DIR, PROBE_IMG)
    for d in DETS:
        try:
            r = probe(d, img)
        except Exception as e:
            import traceback
            r = {"det": d, "image": PROBE_IMG, "error": repr(e),
                 "tb": traceback.format_exc(limit=6)}
        results.append(r)
        # one-line summary
        if "error" in r:
            print(f"{d:25s}  ERR: {r['error']}")
        else:
            cfg = r["predictor"]
            print(f"{d:25s}  cfg=ls{cfg['limit_side_len']}/{cfg['limit_type']}"
                  f"/max{cfg['max_side_limit']} th{cfg['thresh']} bx{cfg['box_thresh']}"
                  f" uc{cfg['unclip_ratio']}  in={r['image_w']}x{r['image_h']}"
                  f" -> resize={r['resize_out_WxH'][0]}x{r['resize_out_WxH'][1]}"
                  f"  n_polys={r['n_polys']}")
    out = {"probe_image": PROBE_IMG, "results": results}
    out_path = "/tmp/baseline_geom.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n# wrote {out_path}")


if __name__ == "__main__":
    main()
