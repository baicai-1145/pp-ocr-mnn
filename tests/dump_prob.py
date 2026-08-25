#!/usr/bin/env python3
"""
Dump PaddleOCR det prob map + polys for a real image, so we can run our C++
db_postprocess on the same prob map and compare the resulting boxes.

Usage:
  python3 tests/dump_prob.py

Writes for each (image, det_model):
  tests/data/dump/<model>__<lang>__<idx>__prob.npy   [H, W] float32 in [0, 1]
  tests/data/dump/<model>__<lang>__<idx>__polys.json  Paddle's boxes, list of
    {"poly": [x0,y0,...,x3,y3] int, "score": float, "src_h": int, "src_w": int}
"""
from __future__ import annotations
import json
import os
import sys

import cv2
import numpy as np


# Test images from the contract (each model picks its supported languages).
TESTS = [
    # (model_name, model_dir, image_path, key)
    ("PP-OCRv4_mobile_det",
     "/root/ppocr_models/PP-OCRv4_mobile_det",
     "/root/ocr_test_imgs/zh/04.jpg",
     "v4_mobile_zh04"),
    ("PP-OCRv4_mobile_det",
     "/root/ppocr_models/PP-OCRv4_mobile_det",
     "/root/ocr_test_imgs/en/01.jpg",
     "v4_mobile_en01"),
    # v6 tiny also useful since M1/v6-tiny is the e2e gate.
    ("PP-OCRv6_tiny_det",
     "/root/ppocr_models/PP-OCRv6_tiny_det",
     "/root/ocr_test_imgs/zh/04.jpg",
     "v6_tiny_zh04"),
    ("PP-OCRv6_tiny_det",
     "/root/ppocr_models/PP-OCRv6_tiny_det",
     "/root/ocr_test_imgs/en/01.jpg",
     "v6_tiny_en01"),
]


def dump_one(model_name: str, model_dir: str, image_path: str, key: str,
             out_dir: str, limit_side_len=None, limit_type=None,
             thresh=None, box_thresh=None, unclip_ratio=None):
    """Run PaddleOCR det via paddlex directly, capture the prob map + boxes.
    PaddleOCR 3.x wraps paddlex; the cleanest way to grab the prob map is to
    call the model's `process()` method and intercept the post_op.
    """
    from paddleocr import PaddleOCR

    # We instantiate PaddleOCR with the det model only, no rec. The model
    # config (thresh, box_thresh, unclip_ratio, limit_side_len, limit_type)
    # is loaded from the model's inference.yml inside model_dir.
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name=model_name,
        text_detection_model_dir=model_dir,
        text_recognition_model_name="PP-OCRv4_mobile_rec",  # any dummy
        text_recognition_model_dir="/root/ppocr_models/PP-OCRv4_mobile_rec",
    )

    # Reach into the paddlex pipeline and grab the det predictor.
    pipeline = ocr.paddlex_pipeline._pipeline
    det_predictor = pipeline.text_det_model
    if det_predictor is None:
        raise RuntimeError("text_det_model is None")

    # Hook the post_op to capture the raw prob map.
    captured: dict = {}
    original_post = det_predictor.post_op

    def hooked_post(batch_preds, batch_shapes, thresh=None, box_thresh=None,
                   unclip_ratio=None):
        # batch_preds is a list of [1, 1, H, W] arrays (one per image).
        captured["probs"] = [p[0, 0].astype(np.float32) for p in batch_preds]
        captured["shapes"] = list(batch_shapes)
        return original_post(
            batch_preds, batch_shapes,
            thresh=thresh, box_thresh=box_thresh, unclip_ratio=unclip_ratio,
        )

    det_predictor.post_op = hooked_post

    # Run the predictor on the single image.
    overrides = {}
    if limit_side_len is not None: overrides["limit_side_len"] = limit_side_len
    if limit_type is not None: overrides["limit_type"] = limit_type
    if thresh is not None: overrides["thresh"] = thresh
    if box_thresh is not None: overrides["box_thresh"] = box_thresh
    if unclip_ratio is not None: overrides["unclip_ratio"] = unclip_ratio

    out = list(det_predictor([image_path if False else cv2.imread(image_path)]))
    # Restore (not strictly needed since we re-instantiate each call).
    det_predictor.post_op = original_post

    assert len(out) == 1, f"expected 1 result, got {len(out)}"
    res = out[0]
    polys = res["dt_polys"]  # list of (N, 2) arrays for the image
    scores = res["dt_scores"]  # list of floats
    raw_img = res["input_img"]  # BGR
    src_h, src_w = raw_img.shape[:2]
    prob = captured["probs"][0]
    H, W = prob.shape

    # Normalize polys to (x0,y0,...,x3,y3) ints, as in baseline JSON.
    out_polys = []
    for poly, score in zip(polys, scores):
        # poly is (N, 2) float (N=4 for quad)
        arr = np.asarray(poly)
        if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] != 2:
            continue
        flat = arr[:4].flatten()
        pts = []
        for i in range(4):
            pts.append(int(round(float(flat[2 * i]))))
            pts.append(int(round(float(flat[2 * i + 1]))))
        out_polys.append({"poly": pts, "score": float(score)})

    # Write outputs.
    npy_path = os.path.join(out_dir, f"{key}__prob.npy")
    poly_path = os.path.join(out_dir, f"{key}__polys.json")
    np.save(npy_path, prob)
    with open(poly_path, "w") as f:
        json.dump({
            "model": model_name,
            "image": image_path,
            "src_h": int(src_h),
            "src_w": int(src_w),
            "prob_h": int(H),
            "prob_w": int(W),
            "ratio_h": float(src_h / H),
            "ratio_w": float(src_w / W),
            "thresh": float(det_predictor.post_op.thresh),
            "box_thresh": float(det_predictor.post_op.box_thresh),
            "unclip_ratio": float(det_predictor.post_op.unclip_ratio),
            "limit_side_len": int(getattr(det_predictor, "limit_side_len", 0) or 0),
            "limit_type": str(getattr(det_predictor, "limit_type", "min") or "min"),
            "polys": out_polys,
        }, f, indent=2)
    print(f"[ok] {key}: src=({src_h},{src_w}) prob=({H},{W}) "
          f"n_boxes={len(out_polys)} -> {npy_path} + {poly_path}")


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "dump")
    os.makedirs(out_dir, exist_ok=True)
    for model, mdir, img, key in TESTS:
        if not os.path.exists(mdir):
            print(f"[skip] {key}: missing model dir {mdir}")
            continue
        if not os.path.exists(img):
            print(f"[skip] {key}: missing image {img}")
            continue
        try:
            dump_one(model, mdir, img, key, out_dir)
        except Exception as e:
            print(f"[fail] {key}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
