#!/usr/bin/env python3
"""gen_baseline_direct.py — regenerate PP-OCR baseline using paddle.inference direct.

Replaces the existing PaddleX-pipeline baseline (which was 5.86e-3 different
from MNN at the model output level — see tools/M2_EXPORT_SWEEP.md) with a
canonical baseline that:
  * uses PaddleX-style preprocessing (matches our C++'s prep_det / rec prep)
  * uses paddle.inference direct as the model backend (matches MNN at
    ~float32 noise)
  * uses PaddleX DBPostProcess + CTCLabelDecode (matches our C++'s
    db_postprocess / ctc_decode)

This produces output in the existing baseline JSON schema, so `score.py`
can compare the MNN matrix directly.

Usage:
  python3 tools/gen_baseline_direct.py --pilot zh      # 1 lang, 1 combo
  python3 tools/gen_baseline_direct.py --pilot          # 1 cell (zh v6_tiny)
  python3 tools/gen_baseline_direct.py --langs zh,en,ja # multiple langs
  python3 tools/gen_baseline_direct.py --full           # all 811 cells
  python3 tools/gen_baseline_direct.py --combo "PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec" --langs zh

The tool writes /root/ppocr_reference/<combo>/<lang>/ocr_results.json
(overwriting existing). BACKUP FIRST:
    mv /root/ppocr_reference /root/ppocr_reference.paddlex.bak
"""
import os, sys, json, math, time, gc, glob, argparse, traceback, warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import paddle.inference as paddle_infer

# --- combo enumeration (mirrors tools/gen_parallel.py) ---
MAIN_DETS = [
    "PP-OCRv4_mobile_det", "PP-OCRv4_server_det",
    "PP-OCRv5_mobile_det", "PP-OCRv5_server_det",
    "PP-OCRv6_tiny_det", "PP-OCRv6_small_det", "PP-OCRv6_medium_det",
]
MAIN_RECS = [
    "PP-OCRv4_mobile_rec", "PP-OCRv4_server_rec",
    "PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec",
    "PP-OCRv6_tiny_rec", "PP-OCRv6_small_rec", "PP-OCRv6_medium_rec",
]
LANGS = ["zh", "en", "ja", "ko", "ru", "ar", "th", "el", "hi", "vi",
         "de", "fr", "es", "it", "pt", "tr"]
PER_LANG_RECS = {
    "arabic_PP-OCRv5_mobile_rec":      ("PP-OCRv5_mobile_det", ["ar", "en"]),
    "cyrillic_PP-OCRv5_mobile_rec":    ("PP-OCRv5_mobile_det", ["ru", "en"]),
    "devanagari_PP-OCRv5_mobile_rec":  ("PP-OCRv5_mobile_det", ["hi", "en"]),
    "el_PP-OCRv5_mobile_rec":          ("PP-OCRv5_mobile_det", ["el", "en"]),
    "en_PP-OCRv4_mobile_rec":          ("PP-OCRv4_mobile_det", ["en"]),
    "en_PP-OCRv5_mobile_rec":          ("PP-OCRv5_mobile_det", ["en"]),
    "eslav_PP-OCRv5_mobile_rec":       ("PP-OCRv5_mobile_det", ["ru", "en"]),
    "korean_PP-OCRv5_mobile_rec":      ("PP-OCRv5_mobile_det", ["ko", "en"]),
    "latin_PP-OCRv5_mobile_rec":       ("PP-OCRv5_mobile_det", ["vi", "en"]),
    "th_PP-OCRv5_mobile_rec":          ("PP-OCRv5_mobile_det", ["th", "en"]),
}
DOC_REC = {
    "model": "PP-OCRv4_server_rec_doc",
    "det": "PP-OCRv4_server_det",
    "langs": ["zh", "en", "ja", "ru", "el"],
}
SEAL_DETS = ["PP-OCRv4_mobile_seal_det", "PP-OCRv4_server_seal_det"]
# Strip cells: rec-only, no det
STRIP_RECS = {
    "ta_PP-OCRv5_mobile_rec": ["ta"],
    "te_PP-OCRv5_mobile_rec": ["te"],
}

REF_ROOT = "/root/ppocr_reference"
IMG_ROOT = "/root/ocr_test_imgs"
MODEL_ROOT = "/root/ppocr_models"
CONFIG_ROOT = "/root/pp-ocr-mnn/configs"


# --- helpers ---

def warp_crop(raw, points):
    """PaddleX-style GetRotateCropImage port (cv2.warpPerspective)."""
    points = np.asarray(points, dtype=np.float32)
    img_crop_width = int(
        max(np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    return cv2.warpPerspective(raw, M, (img_crop_width, img_crop_height),
                               borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)


def levenshtein(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    if len(a) < len(b): a, b = b, a
    n, m = len(b), len(a)
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i; ca = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[n]


def cer(pred, base):
    if not base: return 0.0 if not pred else 1.0
    return levenshtein(pred, base) / len(base)


# --- Paddle inference direct wrappers ---

def det_preprocess(raw, det_cfg):
    """PaddleX-style DetResizeForTest (type0 limit_min) + Normalize + ToCHW.

    For models with `DetResizeForTest: null` in their inference.yml, the
    OCR.yaml override at runtime uses limit_min=64.
    """
    src_h, src_w = raw.shape[:2]
    rc = det_cfg["resize"]
    limit_side = rc.get("limit_side_len", 64)
    stride = rc.get("stride", 32)
    ratio = 1.0
    if min(src_h, src_w) < limit_side:
        ratio = float(limit_side) / min(src_h, src_w)
    resize_h = max(int(round(src_h * ratio / stride) * stride), stride)
    resize_w = max(int(round(src_w * ratio / stride) * stride), stride)
    resized = cv2.resize(raw, (resize_w, resize_h))
    resized = resized.astype("float32") / 255.0
    resized = (resized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    chw = resized.transpose((2, 0, 1)).astype("float32")
    return chw, resize_h, resize_w


def det_infer(det_name, chw, resize_h, resize_w):
    cfg = paddle_infer.Config(
        os.path.join(MODEL_ROOT, det_name, "inference.json"),
        os.path.join(MODEL_ROOT, det_name, "inference.pdiparams"))
    cfg.disable_glog_info(); cfg.disable_gpu(); cfg.disable_mkldnn()
    pred = paddle_infer.create_predictor(cfg)
    inp = pred.get_input_handle(pred.get_input_names()[0])
    inp.reshape([1, 3, resize_h, resize_w])
    inp.copy_from_cpu(chw[np.newaxis, ...])
    pred.run()
    return np.asarray(pred.get_output_handle(pred.get_output_names()[0]).copy_to_cpu())


def db_postprocess(det_out, src_h, src_w, resize_h, resize_w, det_cfg):
    from paddlex.inference.models.text_detection.processors import DBPostProcess
    db = DBPostProcess(
        thresh=det_cfg["thresh"],
        box_thresh=det_cfg["box_thresh"],
        max_candidates=det_cfg.get("max_candidates", 3000),
        unclip_ratio=det_cfg["unclip_ratio"],
        use_dilation=False, score_mode="fast", box_type="quad",
    )
    ratio_h = resize_h / src_h
    ratio_w = resize_w / src_w
    img_shapes = [np.array([src_h, src_w, ratio_h, ratio_w])]
    boxes, scores = db([det_out], img_shapes)
    return boxes[0], scores[0]


def rec_infer(rec_name, crop):
    DICT = json.load(open(f"{CONFIG_ROOT}/{rec_name}.json"))["rec"]["dict"]
    use_space = json.load(open(f"{CONFIG_ROOT}/{rec_name}.json"))["rec"].get("use_space", True)
    H, W = crop.shape[:2]
    imgC, imgH, imgW = 3, 48, 320
    wh_ratio = W / float(H)
    max_wh_ratio = max(imgW / imgH, wh_ratio)
    batch_w = int(imgH * max_wh_ratio)
    if batch_w > 3200:
        batch_w = 3200
    resized_w = min(int(math.ceil(imgH * wh_ratio)), batch_w)
    resized = cv2.resize(crop, (resized_w, imgH))
    resized = resized.astype("float32")
    resized = resized.transpose((2, 0, 1)) / 255
    resized -= 0.5; resized /= 0.5
    chw = np.zeros((imgC, imgH, batch_w), dtype=np.float32)
    chw[:, :, 0:resized_w] = resized
    cfg = paddle_infer.Config(
        os.path.join(MODEL_ROOT, rec_name, "inference.json"),
        os.path.join(MODEL_ROOT, rec_name, "inference.pdiparams"))
    cfg.disable_glog_info(); cfg.disable_gpu(); cfg.disable_mkldnn()
    pred = paddle_infer.create_predictor(cfg)
    inp = pred.get_input_handle(pred.get_input_names()[0])
    inp.reshape([1, 3, 48, batch_w])
    inp.copy_from_cpu(chw[np.newaxis, ...])
    pred.run()
    rout = np.asarray(pred.get_output_handle(pred.get_output_names()[0]).copy_to_cpu())
    logits = rout[0]
    T, C = logits.shape
    argmax = logits.argmax(axis=1)
    sm = np.exp(logits - logits.max(axis=1, keepdims=True))
    sm /= sm.sum(axis=1, keepdims=True)
    idx_list, prev, score_list = [], -1, []
    for t in range(T):
        c = int(argmax[t])
        if c != 0 and c != prev:
            idx_list.append(c)
            score_list.append(float(sm[t, c]))
        prev = c
    n_classes = C
    dict_size = n_classes - 1 - (1 if use_space else 0)
    chars = []
    for i in idx_list:
        if i == 0: continue
        if i - 1 < dict_size:
            chars.append(DICT[i - 1])
        elif use_space and i == 1 + dict_size:
            chars.append(' ')
    return "".join(chars), float(np.mean(score_list)) if score_list else 0.0


# --- image loading ---

def images_for(lang):
    pats = ("*.jpg", "*.jpeg", "*.png")
    out = []
    for pat in pats:
        out += glob.glob(os.path.join(IMG_ROOT, lang, pat))
    return sorted(set(out))


# --- core: run one (det, rec) combo on one lang ---

def run_ocr_cell(det_name, rec_name, lang):
    out_dir = os.path.join(REF_ROOT, f"{det_name}__{rec_name}", lang)
    os.makedirs(out_dir, exist_ok=True)
    jf = os.path.join(out_dir, "ocr_results.json")
    images = images_for(lang)
    if not images:
        return f"no_images:{det_name}__{rec_name}/{lang}"
    det_cfg = json.load(open(f"{CONFIG_ROOT}/{det_name}.json"))["det"]
    out_records = []
    for img in images:
        try:
            raw = cv2.imread(img)
            src_h, src_w = raw.shape[:2]
            chw, rh, rw = det_preprocess(raw, det_cfg)
            det_out = det_infer(det_name, chw, rh, rw)
            boxes, scores = db_postprocess(det_out, src_h, src_w, rh, rw, det_cfg)
            order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0][1], boxes[i][0][0]))
            rec_texts, rec_scores, det_polys = [], [], []
            for i in order:
                crop = warp_crop(raw, boxes[i])
                if crop is None or crop.size == 0:
                    continue
                text, score = rec_infer(rec_name, crop)
                if score > 0.0:  # match PaddleX default
                    poly = [int(round(x)) for x in np.array(boxes[i]).flatten()[:8]]
                    rec_texts.append(text)
                    rec_scores.append(score)
                    det_polys.append(poly)
            detections = [
                {"poly": det_polys[i], "rec_text": rec_texts[i], "rec_score": rec_scores[i]}
                for i in range(len(rec_texts))
            ]
            out_records.append({
                "image_path": img,
                "text": "\n".join(rec_texts),
                "rec_texts": rec_texts,
                "rec_scores": rec_scores,
                "det_polys": det_polys,
                "detections": detections,
            })
        except Exception as e:
            out_records.append({"image_path": img, "error": f"{type(e).__name__}: {e}"})
    tmp = jf + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, jf)
    gc.collect()
    return f"ok:{det_name}__{rec_name}/{lang} ({len(out_records)} imgs)"


# --- rec-only cell (for strip__ cells) ---

def run_strip_cell(rec_name, lang):
    out_dir = os.path.join(REF_ROOT, f"strip__{rec_name}", lang)
    os.makedirs(out_dir, exist_ok=True)
    jf = os.path.join(out_dir, "ocr_results.json")
    images = images_for(lang)
    if not images:
        return f"no_images:strip__{rec_name}/{lang}"
    gt_map = {}
    gtf = os.path.join(IMG_ROOT, "strip_gt.json")
    if os.path.exists(gtf):
        gt_map = json.load(open(gtf, encoding="utf-8"))
    out_records = []
    for img in images:
        try:
            raw = cv2.imread(img)
            text, score = rec_infer(rec_name, raw)
            out_records.append({
                "image_path": img,
                "rec_texts": [text],
                "rec_scores": [score],
                "gt_text": gt_map.get(img, None),
            })
        except Exception as e:
            out_records.append({"image_path": img, "error": str(e)})
    tmp = jf + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, jf)
    return f"ok:strip__{rec_name}/{lang}"


# --- build plan ---

def build_plan(pilot_lang=None, only_combo=None, langs=None):
    tasks = []
    langs_ = [pilot_lang] if pilot_lang else (langs or LANGS)
    for det in MAIN_DETS:
        for rec in MAIN_RECS:
            for l in langs_:
                if only_combo and f"{det}__{rec}" != only_combo:
                    continue
                tasks.append(("ocr", det, rec, l))
    for rec, (det, ls) in PER_LANG_RECS.items():
        for l in ls:
            if l not in langs_: continue
            if only_combo and f"{det}__{rec}" != only_combo:
                continue
            tasks.append(("ocr", det, rec, l))
    for l in DOC_REC["langs"]:
        if l not in langs_: continue
        if only_combo and f"{DOC_REC['det']}__{DOC_REC['model']}" != only_combo:
            continue
        tasks.append(("ocr", DOC_REC["det"], DOC_REC["model"], l))
    # strip
    for rec, ls in STRIP_RECS.items():
        for l in ls:
            if l not in langs_: continue
            tasks.append(("strip", rec, l))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", nargs="?", const="zh", default=None,
                    help="Run only zh+en+ja (3 langs) on 1 combo (default v6_tiny_det__v6_tiny_rec)")
    ap.add_argument("--langs", default=None, help="comma-separated langs")
    ap.add_argument("--combo", default=None, help="only run this combo")
    ap.add_argument("--workers", type=int, default=1, help="parallel workers")
    ap.add_argument("--full", action="store_true", help="all 811 cells")
    args = ap.parse_args()

    if args.pilot:
        langs = ["zh", "en", "ja"]
        only_combo = args.combo or "PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec"
    elif args.langs:
        langs = [s.strip() for s in args.langs.split(",")]
        only_combo = args.combo
    elif args.full:
        langs = None
        only_combo = None
    else:
        print("must specify --pilot, --langs, or --full", file=sys.stderr)
        return 2

    tasks = build_plan(lang if False else None, only_combo, langs)
    if not tasks:
        print("no tasks", file=sys.stderr)
        return 0
    print(f"[plan] {len(tasks)} tasks", flush=True)
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for i, status in enumerate(pool.imap_unordered(worker, tasks), 1):
                print(f"[{i}/{len(tasks)}] {status}", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {worker(t)}", flush=True)
    print(f"[done] {time.time()-t0:.0f}s")
    return 0


def worker(task):
    kind = task[0]
    try:
        if kind == "ocr":
            _, det, rec, lang = task
            return run_ocr_cell(det, rec, lang)
        elif kind == "strip":
            _, rec, lang = task
            return run_strip_cell(rec, lang)
    except Exception:
        return f"ERR:{traceback.format_exc(limit=3)}"


if __name__ == "__main__":
    sys.exit(main())
