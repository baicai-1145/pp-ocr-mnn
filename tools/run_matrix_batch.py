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

def run_cell(task):
    combo, lang, backend, results_dir = task
    det, rec = combo.split("__")
    out_dir = Path(results_dir) / combo / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = images_for(lang)
    if not imgs: return f"noimg:{combo}/{lang}"
    # staging dir with symlinks named 00..09 to keep order
    stage = Path(tempfile.mkdtemp(prefix=f"bm_{lang}_"))
    for i, im in enumerate(imgs):
        lnk = stage / f"{i:02d}_{Path(im).name}"
        try: lnk.symlink_to(im)
        except FileExistsError: pass
    jp = out_dir / "pred.json"
    cmd = [str(CLI), "--batch-dir", str(stage), "--det-config", str(CONFIG/f"{det}.json"),
           "--rec-config", str(CONFIG/f"{rec}.json"), "--model-dir", str(ROOT/"models"),
           "--backend", backend, "--threads", "2", "--json", str(jp)]
    env = {**os.environ, "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu" + (
        ":/root/pp-ocr-mnn/third_party/MNN/build_cuda" if backend == "cuda" else "")}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        return f"timeout:{combo}/{lang}"
    if r.returncode != 0:
        return f"rc{r.returncode}:{combo}/{lang}: {r.stderr[:120]}"
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
    args = ap.parse_args()

    tasks = []
    for det in MAIN_DETS:
        for rec in MAIN_RECS:
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
