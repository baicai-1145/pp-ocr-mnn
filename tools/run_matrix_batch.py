#!/usr/bin/env python3
"""run_matrix_batch.py — drive ppocr_cli --batch-dir per (combo, lang) cell.

Each cell = one CLI invocation (one engine load for 10 images) instead of
10 invocations. Output converted to the same per-image pred.json schema
run_reference.py writes (one JSON array of per-image dicts).

Usage:
  python3 tools/run_matrix_batch.py [--backend cpu|cuda] [--jobs N]
      [--results-dir DIR] [--resume]
"""
import argparse, glob, json, math, os, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "build-main" / "ppocr_cli"
CONFIG = ROOT / "configs"
IMG_ROOT = "/root/ocr_test_imgs"
REF_ROOT = "/root/ppocr_reference"

LANGS = ["ar","de","el","en","es","fr","hi","it","ja","ko","ru","th","tr","vi","zh","pt"]
MAIN_DETS = ["PP-OCRv4_mobile_det","PP-OCRv4_server_det","PP-OCRv5_mobile_det",
             "PP-OCRv5_server_det","PP-OCRv6_tiny_det","PP-OCRv6_small_det",
             "PP-OCRv6_medium_det"]
MAIN_RECS = ["PP-OCRv4_mobile_rec","PP-OCRv4_server_rec","PP-OCRv5_mobile_rec",
             "PP-OCRv5_server_rec","PP-OCRv6_tiny_rec","PP-OCRv6_small_rec",
             "PP-OCRv6_medium_rec"]

NAN_RE = re.compile(r':(-?)(nan|NaN|inf|Infinity)')

def cell_done(results_dir, combo, lang):
    f = Path(results_dir) / combo / lang / "pred.json"
    if not f.exists(): return False
    try:
        d = json.load(open(f))
        return isinstance(d, list) and len(d) >= 1 and "error" not in d[0]
    except Exception:
        return False

def images_for(lang):
    pats = ("*.jpg", "*.jpeg", "*.png")
    out = []
    for pat in pats:
        out += glob.glob(os.path.join(IMG_ROOT, lang, pat))
    return sorted(set(out))

def _ld_for(backend):
    base_ld = os.environ.get("LD_LIBRARY_PATH", "")
    extra = ("/root/mnn-debug/build_dbg:/root/mnn-debug/build_dbg/source/backend/cuda"
             if backend == "cuda" else "")
    ld = "/usr/lib/x86_64-linux-gnu"
    if base_ld: ld = ld + ":" + base_ld
    if extra and extra not in ld: ld = ld + ":" + extra
    return {**os.environ, "LD_LIBRARY_PATH": ld}


def _run_cli(cmd, env, timeout=600):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0:
        return None, f"rc{r.returncode}: {r.stderr[:120]}"
    return r, None


def run_cell(task):
    combo, lang, backend, results_dir = task
    det, rec = combo.split("__")
    out_dir = Path(results_dir) / combo / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = images_for(lang)
    if not imgs: return f"noimg:{combo}/{lang}"
    jp = out_dir / "pred.json"
    env = _ld_for(backend)
    if backend == "cuda" and os.environ.get("PPOCR_RUNNER_PER_IMAGE") == "1":
        # Optional per-image mode (legacy workaround for the pre-fix MNN
        # int32 overflow which corrupted batch-mode outputs; the fixed
        # build batches fine, engine-side CPU fallback covers OOM bigs).
        recs = []
        for im in imgs:
            tmp_json = out_dir / f".tmp_{Path(im).stem}.json"
            cmd = [str(CLI), "--image", str(im),
                   "--det-config", str(CONFIG/f"{det}.json"),
                   "--rec-config", str(CONFIG/f"{rec}.json"),
                   "--model-dir", str(ROOT/"models"),
                   "--backend", backend, "--threads", "2", "--json", str(tmp_json)]
            r, err = _run_cli(cmd, env)
            if r is None:
                # occasional transient CUDA init race under parallel jobs:
                # retry once before giving up
                r, err = _run_cli(cmd, env)
            if r is None:
                return f"{err}:{combo}/{lang}:{Path(im).name}"
            txt = tmp_json.read_text()
            # empty-detection transient: retry once when a non-empty
            # baseline exists but this run found nothing
            import re as _re
            if '"lines":[]' in txt.replace(" ", ""):
                r2, _ = _run_cli(cmd, env)
                if r2 is not None:
                    txt2 = tmp_json.read_text()
                    if '"lines":[]' not in txt2.replace(" ", ""):
                        txt = txt2
            recs.append((txt, Path(im)))
        # merge single-image JSON objects into an array the
        # conversion step below expects (isinstance(dict)->[dict] path)
        objs = [json.loads(NAN_RE.sub(r':null', txt)) for txt, _ in recs]
        with open(jp, "w") as tf:
            json.dump(objs, tf)
    else:
        # staging dir with symlinks named 00..09 to keep order
        stage = Path(tempfile.mkdtemp(prefix=f"bm_{lang}_"))
        for i, im in enumerate(imgs):
            lnk = stage / f"{i:02d}_{Path(im).name}"
            try: lnk.symlink_to(im)
            except FileExistsError: pass
        cmd = [str(CLI), "--batch-dir", str(stage), "--det-config", str(CONFIG/f"{det}.json"),
               "--rec-config", str(CONFIG/f"{rec}.json"), "--model-dir", str(ROOT/"models"),
               "--backend", backend, "--threads", "2", "--json", str(jp)]
        r, err = _run_cli(cmd, env)
        if r is None:
            return f"{err}:{combo}/{lang}"
    # JSONL -> per-image array in baseline-comparable schema
    try:
        recs = []
        with open(jp) as f:
            txt = f.read()
        arr = json.loads(NAN_RE.sub(r':null', txt))
        if isinstance(arr, dict): arr = [arr]
        arr = [d for d in arr if "_bench" not in d]
        for d in arr:
                img_name = d.get("image", "")
                # map staged name back to original path
                base = os.path.basename(img_name)
                orig = img_name
                if len(base) > 3 and base[:2].isdigit() and base[2] == "_":
                    tail = base[3:]
                    orig = next((im for im in imgs if os.path.basename(im) == tail), img_name)
                recs.append({"image_path": orig,
                             "rec_texts": [l.get("text", "") for l in d.get("lines", [])],
                             "rec_scores": [l.get("score") for l in d.get("lines", [])],
                             "det_polys": [l.get("poly") for l in d.get("lines", [])]})
        tmp = str(jp) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(recs, f, ensure_ascii=False)
        os.replace(tmp, jp)
        return f"ok:{combo}/{lang} ({len(recs)})"
    except Exception as e:
        return f"convfail:{combo}/{lang}: {e}"
    finally:
        for c in out_dir.glob(".tmp_*.json"):
            try: c.unlink()
            except OSError: pass
        if 'stage' in dir():
            for p in stage.iterdir():
                try: p.unlink()
                except OSError: pass
            try: stage.rmdir()
            except OSError: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dets", default=None, help="comma-separated det filter")
    ap.add_argument("--recs", default=None, help="comma-separated rec filter")
    args = ap.parse_args()

    dets = args.dets.split(",") if args.dets else MAIN_DETS
    recs = args.recs.split(",") if args.recs else MAIN_RECS
    tasks = []
    for det in dets:
        for rec in recs:
            combo = f"{det}__{rec}"
            for lang in LANGS:
                if args.resume and cell_done(args.results_dir, combo, lang):
                    continue
                tasks.append((combo, lang, args.backend, args.results_dir))
    print(f"[plan] {len(tasks)} cells", flush=True)
    t0 = os.times()
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for i, res in enumerate(ex.map(run_cell, tasks), 1):
            done += 1
            print(f"[{i}/{len(tasks)}] {res}", flush=True)
    print(f"[done] {done} cells")

if __name__ == "__main__":
    main()
