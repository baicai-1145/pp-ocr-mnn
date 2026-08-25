#!/usr/bin/env python3
"""Parallel baseline regeneration for pp-ocr-mnn.

Generates, with a multiprocessing pool (GPU, ~4 workers):
  A. main-matrix cells removed by `gen_reference.py clean`  (91 cells)
  B. 12 per-lang rec models x v5_mobile_det, native + en test images
  C. PP-OCRv4_server_rec_doc x v4_server_det, native-lang cells
  D. 2 seal det models x official SealRecognition pipeline x seal imgs

Each cell = /root/ppocr_reference/<combo>/<lang>/ocr_results.json
Usage: python3 gen_parallel.py [workers]
"""
import os, sys, json, glob, time, gc, traceback, warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")

IMG_ROOT = "/root/ocr_test_imgs"
OUT_ROOT = "/root/ppocr_reference"
PPOCR_MODELS = "/root/ppocr_models"

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
LANG_TO_PADDLE = {
    "zh": "ch", "en": "en", "ja": "japan", "ko": "korean", "ru": "ru",
    "ar": "ar", "th": "th", "el": "el", "hi": "hi", "vi": "vi",
    "de": "de", "fr": "fr", "es": "es", "it": "it", "pt": "pt", "tr": "tr",
}

# per-lang rec: companion det (same gen) -> native-lang test images + en
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
    "ta_PP-OCRv5_mobile_rec":          ("PP-VRModule", None),   # strip-mode cell, see below
    "te_PP-OCRv5_mobile_rec":          ("PP-VRModule", None),
    "th_PP-OCRv5_mobile_rec":          ("PP-OCRv5_mobile_det", ["th", "en"]),
}
# strip-mode: no det stage — synthesized text-line strips, rec-only baseline
STRIP_RECS = {
    "ta_PP-OCRv5_mobile_rec": ["ta"],
    "te_PP-OCRv5_mobile_rec": ["te"],
}
# drop strip-mode entries from the OCR-path table
PER_LANG_RECS = {k: v for k, v in PER_LANG_RECS.items() if v[1] is not None}

# doc rec: companion det is same-gen v4_server; native coverage per dict
DOC_REC = {
    "model": "PP-OCRv4_server_rec_doc",
    "det": "PP-OCRv4_server_det",
    "langs": ["zh", "en", "ja", "ru", "el"],
}

SEAL_DETS = ["PP-OCRv4_mobile_seal_det", "PP-OCRv4_server_seal_det"]

def combo_name(d, r):
    return f"{d}__{r}"

def cell_done(combo, lang):
    jf = os.path.join(OUT_ROOT, combo, lang, "ocr_results.json")
    if not os.path.exists(jf):
        return False
    try:
        d = json.load(open(jf))
        if not d:
            return False
        # OCR schema cells carry rec_texts; seal cells carry det_scores only
        if "rec_texts" in d[0]:
            return len(d) >= 10 and all("rec_texts" in r for r in d)
        return len(d) >= 30
    except Exception:
        return False

def images_for(lang):
    pats = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for pat in pats:
        files += glob.glob(os.path.join(IMG_ROOT, lang, pat))
    return sorted(set(files))

# ---------------------------------------------------------------- OCR cell
def run_ocr_cell(det_v, rec_v, lang_code):
    paddle_lang = LANG_TO_PADDLE[lang_code]
    cname = combo_name(det_v, rec_v)
    out_dir = os.path.join(OUT_ROOT, cname, lang_code)
    os.makedirs(out_dir, exist_ok=True)
    jf = os.path.join(out_dir, "ocr_results.json")
    images = images_for(lang_code)
    if not images:
        return f"no_images:{cname}/{lang_code}"

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        lang=paddle_lang,
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name=det_v,
        text_detection_model_dir=os.path.join(PPOCR_MODELS, det_v),
        text_recognition_model_name=rec_v,
        text_recognition_model_dir=os.path.join(PPOCR_MODELS, rec_v),
    )
    recs_out = []
    for img in images:
        try:
            res = ocr.predict(img)
            info = res[0] if res else {}
        except Exception as e:
            recs_out.append({"image_path": img, "error": str(e)})
            continue
        texts = list(info.get("rec_texts") or [])
        scores = list(info.get("rec_scores") or [])
        polys = list(info.get("rec_polys", info.get("rec_boxes", [])) or [])
        det_polys, detections = [], []
        for i, t in enumerate(texts):
            poly = polys[i] if i < len(polys) else None
            box = None
            if poly is not None:
                try:
                    box = [int(x) for x in list(np.array(poly).flatten())[:8]]
                except Exception:
                    box = None
            det_polys.append(box)
            detections.append({
                "poly": box,
                "rec_text": str(t),
                "rec_score": float(scores[i]) if i < len(scores) else 1.0,
            })
        recs_out.append({
            "image_path": img,
            "text": "\n".join(str(t) for t in texts),
            "rec_texts": [str(t) for t in texts],
            "rec_scores": [float(s) for s in scores],
            "det_polys": det_polys,
            "detections": detections,
        })
    tmp = jf + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(recs_out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, jf)
    del ocr
    gc.collect()
    return f"ok:{cname}/{lang_code}"

# ---------------------------------------------------------------- seal cell
def run_seal_cell(det_v):
    """Seal det baseline: official TextDetection dt_polys/dt_scores on synth stamps.
    Recognition GT lives in /root/ocr_test_imgs/seal/ground_truth.json (M4 uses it).
    """
    import numpy as np
    from paddleocr import TextDetection
    out_dir = os.path.join(OUT_ROOT, det_v, "seal")
    os.makedirs(out_dir, exist_ok=True)
    jf = os.path.join(out_dir, "ocr_results.json")
    images = sorted(glob.glob(os.path.join(IMG_ROOT, "seal", "*.jpg")))
    det = TextDetection(
        model_name=det_v,
        model_dir=os.path.join(PPOCR_MODELS, det_v),
    )
    out = []
    for img in images:
        try:
            res = det.predict(img)[0]
            polys = [[ [float(x), float(y)] for x, y in p ] for p in (res.get("dt_polys") or [])]
            scores = [float(s) for s in (res.get("dt_scores") or [])]
        except Exception as e:
            out.append({"image_path": img, "error": str(e)})
            continue
        out.append({
            "image_path": img,
            "det_polys": polys,
            "det_scores": scores,
        })
    tmp = jf + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, jf)
    del det
    gc.collect()
    return f"ok:{det_v}/seal"

# ---------------------------------------------------------------- strip cell
def run_strip_cell(rec_v, lang_code):
    """Rec-only baseline on synthesized text-line strips (ta/te).
    Full-OCR det cannot detect isolate text lines reliably; strips are the
    native input distribution of rec models anyway.
    Stores both official prediction and ground truth text.
    """
    import numpy as np
    from paddleocr import TextRecognition
    cname = f"strip__{rec_v}"
    out_dir = os.path.join(OUT_ROOT, cname, lang_code)
    os.makedirs(out_dir, exist_ok=True)
    jf = os.path.join(out_dir, "ocr_results.json")
    images = images_for(lang_code)
    if not images:
        return f"no_images:{cname}/{lang_code}"
    rec = TextRecognition(
        model_name=rec_v,
        model_dir=os.path.join(PPOCR_MODELS, rec_v),
    )
    gt_map = {}
    gtf = os.path.join(IMG_ROOT, "strip_gt.json")
    if os.path.exists(gtf):
        gt_map = json.load(open(gtf, encoding="utf-8"))
    out = []
    for img in images:
        try:
            r = rec.predict(img)[0]
            pred = str(r.get("rec_text") or "")
            score = float(r.get("rec_score") or 0.0)
        except Exception as e:
            out.append({"image_path": img, "error": str(e)})
            continue
        out.append({
            "image_path": img,
            "rec_texts": [pred],
            "rec_scores": [score],
            "gt_text": gt_map.get(img, None),
        })
    tmp = jf + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, jf)
    del rec
    gc.collect()
    return f"ok:{cname}/{lang_code}"

# ---------------------------------------------------------------- plan
def build_plan():
    tasks = []  # ("ocr", det, rec, lang) | ("seal", det)
    # A. main matrix gaps
    for det in MAIN_DETS:
        for rec in MAIN_RECS:
            c = combo_name(det, rec)
            for l in LANGS:
                if not cell_done(c, l):
                    tasks.append(("ocr", det, rec, l))
    # B. per-lang recs
    for rec, (det, langs_) in PER_LANG_RECS.items():
        for l in langs_:
            if not cell_done(combo_name(det, rec), l):
                tasks.append(("ocr", det, rec, l))
    # B2. strip-mode cells (ta/te)
    for rec, langs_ in STRIP_RECS.items():
        for l in langs_:
            if not cell_done(f"strip__{rec}", l):
                tasks.append(("strip", rec, l))
    # C. doc rec
    for l in DOC_REC["langs"]:
        if not cell_done(combo_name(DOC_REC["det"], DOC_REC["model"]), l):
            tasks.append(("ocr", DOC_REC["det"], DOC_REC["model"], l))
    # D. seal
    for det in SEAL_DETS:
        if not cell_done(det, "seal"):
            tasks.append(("seal", det))
    return tasks

def worker(task):
    kind = task[0]
    t0 = time.time()
    try:
        if kind == "ocr":
            _, det, rec, lang = task
            return task, run_ocr_cell(det, rec, lang), time.time() - t0
        elif kind == "strip":
            _, rec, lang = task
            return task, run_strip_cell(rec, lang), time.time() - t0
        else:
            return task, ("seal", run_seal_cell(task[1])), time.time() - t0
    except Exception:
        return task, f"ERR:{traceback.format_exc(limit=3)}", time.time() - t0

if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    plan = build_plan()
    print(f"[plan] {len(plan)} tasks, {n_workers} workers", flush=True)
    for t in plan:
        print("  todo", t, flush=True)
    if not plan:
        print("[done] nothing to do")
        sys.exit(0)
    t0 = time.time()
    ok = err = 0
    with mp.Pool(n_workers) as pool:
        for i, (task, status, dt) in enumerate(pool.imap_unordered(worker, plan), 1):
            line = str(status).splitlines()[0][:120]
            print(f"[{i}/{len(plan)}] {dt:6.1f}s {line}", flush=True)
            if str(status).startswith(("ok", "('seal'")) or str(status).startswith("seal"):
                ok += 1
            else:
                err += 1
    print(f"[done] ok={ok} err={err} in {time.time()-t0:.0f}s", flush=True)
