#!/usr/bin/env python3
"""convert_models.py — full paddle→onnx→mnn conversion for the 30 PP-OCR models.

Reuses pre-converted ONNX files in `models/_onnx/<name>.onnx` (already produced
by previous runs) when present; only the cls model is converted fresh
(via paddle2onnx from the official PaddlePaddle model dir).

Emits:
  models/<name>.mnn                         (the converted model file)
  configs/<name>.json                       (model config; dict inline)
  configs/registry.json                     (name → {file, sha256, bytes, url, type})

Usage:
  convert_models.py [--models-dir DIR] [--configs-dir DIR] [--mnnconvert PATH]
                    [--paddle2onnx PATH] [--onnx-dir DIR] [--cls-dir PATH]
                    [--force] [--only NAME] [--skip NAME]... [--no-regen-onnx]
                    [--jobs N] [-v]

Behavior:
  * For each of the 30 catalog names, if models/<name>.mnn exists and not --force
    the .mnn is reused (sha256 is recomputed over the existing file).
  * If models/_onnx/<name>.onnx exists (or --no-regen-onnx), ONNX stage is skipped.
  * Otherwise the script invokes paddle2onnx on /root/ppocr_models/<name>/.
  * MNNConvert: third_party/MNN/build/MNNConvert by default (overridable).
  * All subprocess failures abort with non-zero exit; per-model try/except lets
    one bad model not stop the run unless --strict.

No external state beyond the listed dirs; idempotent.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_dict import (  # noqa: E402
    ExtractedModel, extract, extract_cls, write_config,
)


# ---------------------------------------------------------------------------
# Model catalog (30 names: 7 det + 2 seal det + 20 rec + 1 cls)
# Order matters for the registry; kept stable.
# ---------------------------------------------------------------------------

DET_NAMES: List[str] = [
    "PP-OCRv4_mobile_det",
    "PP-OCRv4_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv5_server_det",
    "PP-OCRv6_tiny_det",
    "PP-OCRv6_small_det",
    "PP-OCRv6_medium_det",
]

SEAL_DET_NAMES: List[str] = [
    "PP-OCRv4_mobile_seal_det",
    "PP-OCRv4_server_seal_det",
]

REC_NAMES: List[str] = [
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_rec",
    "PP-OCRv4_server_rec_doc",
    "PP-OCRv5_mobile_rec",
    "PP-OCRv5_server_rec",
    "PP-OCRv6_tiny_rec",
    "PP-OCRv6_small_rec",
    "PP-OCRv6_medium_rec",
    "en_PP-OCRv4_mobile_rec",
    "en_PP-OCRv5_mobile_rec",
    "arabic_PP-OCRv5_mobile_rec",
    "cyrillic_PP-OCRv5_mobile_rec",
    "devanagari_PP-OCRv5_mobile_rec",
    "el_PP-OCRv5_mobile_rec",
    "eslav_PP-OCRv5_mobile_rec",
    "korean_PP-OCRv5_mobile_rec",
    "latin_PP-OCRv5_mobile_rec",
    "ta_PP-OCRv5_mobile_rec",
    "te_PP-OCRv5_mobile_rec",
    "th_PP-OCRv5_mobile_rec",
]

CLS_NAME = "PP-LCNet_x1_0_textline_ori"
CLS_DIR = Path(os.path.expanduser(
    "~/.paddlex/official_models/PP-LCNet_x1_0_textline_ori"
))

PPOCR_MODELS_DIR = Path("/root/ppocr_models")


def all_model_names() -> List[str]:
    return DET_NAMES + SEAL_DET_NAMES + REC_NAMES + [CLS_NAME]


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ConvertResult:
    name: str
    kind: str
    mnn_path: str
    config_path: str
    sha256: str
    bytes: int
    reused_mnn: bool
    reused_onnx: bool
    onnx_path: Optional[str] = None
    error: Optional[str] = None


def _sha256_file(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: List[str], *, timeout: int = 1800) -> Tuple[int, str, str]:
    """Run subprocess; return (rc, stdout, stderr)."""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def _paddle_to_onnx(paddle_dir: Path, out_onnx: Path, paddle2onnx: str) -> None:
    """paddle2onnx --model_dir PADDLE_DIR --model_filename inference.pdmodel
    --params_filename inference.pdiparams --save_file OUT --opset_version 14
    --enable_onnx_checker True
    """
    cmd = [
        paddle2onnx,
        "--model_dir", str(paddle_dir),
        "--model_filename", "inference.pdmodel" if (paddle_dir / "inference.pdmodel").exists() else "inference.json",
        "--params_filename", "inference.pdiparams",
        "--save_file", str(out_onnx),
        "--opset_version", "14",
        "--enable_onnx_checker", "True",
    ]
    rc, out, err = _run(cmd, timeout=1800)
    if rc != 0 or not out_onnx.exists():
        raise RuntimeError(
            f"paddle2onnx failed for {paddle_dir} (rc={rc})\nstdout={out}\nstderr={err}"
        )


def _mnnconvert(onnx_path: Path, mnn_path: Path, mnnconvert: str) -> None:
    # MNNConvert (2.9.x) uses long flags: --modelFile and --MNNModel.
    cmd = [mnnconvert, "-f", "ONNX", "--modelFile", str(onnx_path),
           "--MNNModel", str(mnn_path)]
    rc, out, err = _run(cmd, timeout=1800)
    if rc != 0 or not mnn_path.exists():
        raise RuntimeError(
            f"MNNConvert failed for {onnx_path} (rc={rc})\nstdout={out}\nstderr={err}"
        )


# ---------------------------------------------------------------------------
# Main per-model convert
# ---------------------------------------------------------------------------

def convert_one(name: str, *,
                models_dir: Path,
                configs_dir: Path,
                onnx_dir: Path,
                mnnconvert: str,
                paddle2onnx: str,
                force_mnn: bool = False,
                regen_onnx: bool = False,
                verbose: bool = False) -> ConvertResult:
    is_cls = (name == CLS_NAME)
    if is_cls:
        src_dir = CLS_DIR
        ext: ExtractedModel = extract_cls()
    else:
        src_dir = PPOCR_MODELS_DIR / name
        if not (src_dir / "inference.yml").exists():
            return ConvertResult(name, "?", "", "", "", 0, False, False,
                                 error=f"inference.yml not found under {src_dir}")
        ext = extract(str(src_dir / "inference.yml"))

    onnx_path = onnx_dir / f"{name}.onnx"
    mnn_path = models_dir / f"{name}.mnn"
    reused_onnx = onnx_path.exists() and not regen_onnx
    reused_mnn = mnn_path.exists() and not force_mnn

    try:
        if not reused_mnn:
            if not reused_onnx:
                if verbose:
                    print(f"[onnx] {name} ← {src_dir}", file=sys.stderr)
                onnx_path.parent.mkdir(parents=True, exist_ok=True)
                # Some paddle inference.yml come with .pdmodel or .json as model file.
                # Our _paddle_to_onnx handles both via autodetect.
                _paddle_to_onnx(src_dir, onnx_path, paddle2onnx)
            if verbose:
                print(f"[mnn]  {name} ← {onnx_path}", file=sys.stderr)
            mnn_path.parent.mkdir(parents=True, exist_ok=True)
            _mnnconvert(onnx_path, mnn_path, mnnconvert)
        # sha256 over the produced .mnn (always recompute; cheap and authoritative).
        sha = _sha256_file(str(mnn_path))
        nbytes = mnn_path.stat().st_size
        # Write per-model config (always; we may have updated params).
        write_config(str(configs_dir), ext,
                     file=f"{name}.mnn", sha256=sha, bytes_=nbytes,
                     url=f"{name}.mnn")
        cfg_path = str(configs_dir / f"{name}.json")
        return ConvertResult(name=name, kind=ext.kind, mnn_path=str(mnn_path),
                             config_path=cfg_path, sha256=sha, bytes=nbytes,
                             reused_mnn=reused_mnn, reused_onnx=reused_onnx,
                             onnx_path=str(onnx_path))
    except Exception as e:
        return ConvertResult(name=name, kind=ext.kind if ext else "?", mnn_path="",
                             config_path="", sha256="", bytes=0,
                             reused_mnn=reused_mnn, reused_onnx=reused_onnx,
                             onnx_path=str(onnx_path), error=str(e))


# ---------------------------------------------------------------------------
# Registry writer
# ---------------------------------------------------------------------------

def write_registry(configs_dir: Path, models_dir: Path) -> str:
    """Aggregate configs/<name>.json into a single registry.json keyed by name."""
    reg: Dict[str, Dict[str, object]] = {}
    for cfg in sorted(configs_dir.glob("*.json")):
        if cfg.name == "registry.json":
            continue
        with open(cfg, "r", encoding="utf-8") as f:
            d = json.load(f)
        name = d["name"]
        reg[name] = {
            "name": name,
            "type": d["type"],
            "file": d["file"],
            "sha256": d["sha256"],
            "bytes": d["bytes"],
            "url": d["url"],
        }
    out = configs_dir / "registry.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    # sorted? we already wrote sorted keys.
    return str(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="convert_models",
                                 description="Convert all 30 PP-OCR models to MNN")
    ap.add_argument("--models-dir", default=str(HERE.parent / "models"))
    ap.add_argument("--configs-dir", default=str(HERE.parent / "configs"))
    ap.add_argument("--onnx-dir", default=str((HERE.parent / "models" / "_onnx")))
    ap.add_argument("--mnnconvert", default="/root/pp-ocr-mnn/third_party/MNN/build/MNNConvert")
    ap.add_argument("--paddle2onnx", default="/root/.local/pytools/bin/paddle2onnx")
    ap.add_argument("--force", action="store_true",
                    help="re-convert .mnn even if it exists")
    ap.add_argument("--no-regen-onnx", action="store_true",
                    help="reuse existing onnx even if it exists; default true behavior")
    ap.add_argument("--regen-onnx", action="store_true",
                    help="re-run paddle2onnx even if onnx exists")
    ap.add_argument("--only", action="append", default=[],
                    help="restrict to NAME (repeatable)")
    ap.add_argument("--skip", action="append", default=[],
                    help="skip NAME (repeatable)")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="parallel workers for per-model conversion")
    ap.add_argument("--strict", action="store_true",
                    help="abort on first error")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    models_dir = Path(args.models_dir)
    configs_dir = Path(args.configs_dir)
    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.mnnconvert).exists():
        print(f"ERROR: MNNConvert not found at {args.mnnconvert}", file=sys.stderr)
        return 2
    if not Path(args.paddle2onnx).exists():
        print(f"ERROR: paddle2onnx not found at {args.paddle2onnx}", file=sys.stderr)
        return 2

    names = all_model_names()
    if args.only:
        wanted = set(args.only)
        names = [n for n in names if n in wanted]
    if args.skip:
        names = [n for n in names if n not in set(args.skip)]

    t0 = time.time()
    results: List[ConvertResult] = []

    def _do(n: str) -> ConvertResult:
        return convert_one(
            n,
            models_dir=models_dir,
            configs_dir=configs_dir,
            onnx_dir=onnx_dir,
            mnnconvert=args.mnnconvert,
            paddle2onnx=args.paddle2onnx,
            force_mnn=args.force,
            regen_onnx=args.regen_onnx and not args.no_regen_onnx,
            verbose=args.verbose,
        )

    if args.jobs <= 1:
        for n in names:
            r = _do(n)
            results.append(r)
            if r.error:
                print(f"[FAIL] {n}: {r.error}", file=sys.stderr)
                if args.strict:
                    return 1
            else:
                tag = []
                if r.reused_mnn:
                    tag.append("mnn-reused")
                if r.reused_onnx:
                    tag.append("onnx-reused")
                t = (" " + "/".join(tag)) if tag else ""
                print(f"[OK]   {n:35s} {r.kind:3s} {r.bytes:>10d}B{sha_short(r.sha256)}  {os.path.basename(r.mnn_path)}{t}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_do, n): n for n in names}
            for fut in concurrent.futures.as_completed(futs):
                r = fut.result()
                results.append(r)
                if r.error:
                    print(f"[FAIL] {r.name}: {r.error}", file=sys.stderr)
                    if args.strict:
                        return 1

    # Always write registry.
    reg = write_registry(configs_dir, models_dir)
    dt = time.time() - t0
    ok = sum(1 for r in results if not r.error)
    fail = sum(1 for r in results if r.error)
    print(f"\nregistry: {reg}  ({len(results)} entries; ok={ok} fail={fail}; {dt:.1f}s)")
    return 0 if fail == 0 else 1


def sha_short(s: str) -> str:
    return f"  {s[:8]}" if s else ""


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
