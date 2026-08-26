#!/usr/bin/env python3
"""extract_dict.py — parse PaddleOCR inference.yml → dict + det/rec/cls params.

Library API (used by convert_models.py):
    extract(path) → ExtractedModel (with .kind == 'det'|'rec'|'cls')
    write_config(out_dir, name, ext: ExtractedModel) → path to configs/<name>.json

CLI:
    extract_dict.py INFERENCE_YML OUT_JSON [--name NAME]
    extract_dict.py --cls OUT_JSON  (uses ~/.paddlex/official_models/PP-LCNet_x1_0_textline_ori)

Schema for emitted JSON strictly follows docs/CONTRACT.md.

Det resize / postprocess policy (M2-FIX):
    The 811-cell baseline at /root/ppocr_reference/ was generated with
    PaddleX's default OCR pipeline config
    (`paddlex/configs/pipelines/OCR.yaml`), which hard-codes:

        SubModules.TextDetection:
          limit_side_len: 64
          limit_type: min
          max_side_limit: 4000
          thresh: 0.3
          box_thresh: 0.6
          unclip_ratio: 1.5

    for **every** det model in {v4 mobile/server, v5 mobile/server, v6
    tiny/small/medium, v4 seal mobile/server}. The per-model
    `inference.yml` values for `DetResizeForTest` and `PostProcess` are
    *ignored* at PaddleX runtime (PaddleX takes its defaults from the
    pipeline YAML).

    Empirical verification: tools/probe_baseline_geom.py. The full
    geometric contract lives in docs/DET_GEOMETRY.md.

    Therefore every det config we emit — v4 mobile, v4 server, v5
    mobile, v5 server, v6 tiny, v6 small, v6 medium, and the seal det
    variants — uses the same uniform block:
        resize: { mode: "limit_min", limit_side_len: 64, stride: 32,
                  max_side_limit: 4000 }
        thresh: 0.3
        box_thresh: 0.6
        unclip_ratio: 1.5

    M2-FIX removed the per-model PostProcess overrides and the
    `resize_long=960` (v4/v5) / `limit_min 736` (v6) split because
    PaddleX overrides them at runtime. We reproduce what the baseline
    was generated with, not what the per-model yml reads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # PyYAML
except ImportError as e:  # pragma: no cover
    print(f"extract_dict: PyYAML required ({e})", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Public dataclasses (mirror C++ structs in include/ppocr/config.h)
# ---------------------------------------------------------------------------

@dataclass
class DetResizeCfg:
    mode: str  # "limit_min" | "resize_long"
    limit_side_len: int = 736
    resize_long: int = 960
    stride: int = 32
    max_side_limit: int = 4000


@dataclass
class DetCfg:
    thresh: float = 0.3
    box_thresh: float = 0.6
    unclip_ratio: float = 1.5
    max_candidates: int = 1000
    resize: DetResizeCfg = field(default_factory=DetResizeCfg)


@dataclass
class RecCfg:
    shape: List[int] = field(default_factory=lambda: [3, 48, 320])
    use_space: bool = True
    dict: List[str] = field(default_factory=list)


@dataclass
class ClsCfg:
    shape: List[int] = field(default_factory=lambda: [3, 80, 160])
    labels: List[str] = field(default_factory=list)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ExtractedModel:
    name: str
    kind: str  # "det" | "rec" | "cls"
    det: Optional[DetCfg] = None
    rec: Optional[RecCfg] = None
    cls: Optional[ClsCfg] = None
    src_path: str = ""


# ---------------------------------------------------------------------------
# Resize-mode policy (verified vs PaddleX runtime, M2-FIX)
# ---------------------------------------------------------------------------

# PaddleX baseline generation uses a single uniform det config across
# all 8 det models. See tools/probe_baseline_geom.py and
# docs/DET_GEOMETRY.md. The values below are NOT a function of the
# per-model yml — yml is ignored by PaddleX at runtime.
_PADDLEX_DET_RESIZE = DetResizeCfg(mode="limit_min", limit_side_len=64,
                                   resize_long=960, stride=32,
                                   max_side_limit=4000)
_PADDLEX_DET_POST = dict(thresh=0.3, box_thresh=0.6, unclip_ratio=1.5,
                         max_candidates=1000)


def _resize_policy(model_name: str, resize_node: Optional[Dict[str, Any]]) -> DetResizeCfg:
    """All 7 main det models + the seal det variants use the same
    PaddleX-runtime default. The `resize_node` from inference.yml is
    accepted only for the seal det which PaddleX does not override
    (`DetResizeForTest: { resize_long: 736 }` in seal yml is actually
    used). Returns the uniform cfg for everything else.
    """
    is_seal = "seal" in model_name.lower()
    if is_seal and resize_node and "resize_long" in resize_node:
        # Seal det: PaddleX's OCR pipeline `text_type: seal` reads the
        # seal yml directly (no override of PostProcess). Keep the
        # resize_long path.
        long_ = int(resize_node["resize_long"])
        return DetResizeCfg(mode="resize_long", limit_side_len=736,
                            resize_long=long_, stride=128,
                            max_side_limit=4000)
    return _PADDLEX_DET_RESIZE


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_det(name: str, d: Dict[str, Any]) -> DetCfg:
    pp = d.get("PreProcess", {}) or {}
    op_list = pp.get("transform_ops", []) or []
    resize_node: Optional[Dict[str, Any]] = None
    for op in op_list:
        # yaml keys: "DetResizeForTest", possibly null
        if "DetResizeForTest" in op:
            v = op["DetResizeForTest"]
            if isinstance(v, dict):
                resize_node = v
            break
    resize = _resize_policy(name, resize_node)
    # M2-DET-FINAL: PaddleX overrides the per-model PostProcess at
    # runtime with the values from paddlex/configs/pipelines/OCR.yaml
    # (thresh=0.3, box_thresh=0.6, unclip_ratio=1.5 for all v4/v5/v6
    # det). The per-model inference.yml values (e.g. v6: 0.2/0.4/1.4)
    # are NOT used by the PaddleX `PaddleOCR` runtime path; the
    # yml values are only used if a downstream user explicitly
    # constructs the predictor with `thresh=0.2` etc. and the
    # /root/ppocr_reference/ baselines were generated with the
    # OCR.yaml override. Emitting the OCR.yaml values for the
    # C++ path keeps us matched to the baseline. For the seal
    # det, the seal pipeline does not load OCR.yaml, so we
    # fall back to the yml values (which is what the seal
    # baseline generator uses).
    if "seal" in name.lower():
        pp2 = d.get("PostProcess", {}) or {}
        return DetCfg(
            thresh=float(pp2.get("thresh", 0.2)),
            box_thresh=float(pp2.get("box_thresh", 0.6)),
            unclip_ratio=float(pp2.get("unclip_ratio", 0.5)),
            max_candidates=int(pp2.get("max_candidates", 1000)),
            resize=resize,
        )
    return DetCfg(
        thresh=_PADDLEX_DET_POST["thresh"],
        box_thresh=_PADDLEX_DET_POST["box_thresh"],
        unclip_ratio=_PADDLEX_DET_POST["unclip_ratio"],
        max_candidates=_PADDLEX_DET_POST["max_candidates"],
        resize=resize,
    )


def _extract_rec(name: str, d: Dict[str, Any]) -> RecCfg:
    pp = d.get("PreProcess", {}) or {}
    shape = [3, 48, 320]
    for op in (pp.get("transform_ops") or []):
        if "RecResizeImg" in op and isinstance(op["RecResizeImg"], dict):
            sh = op["RecResizeImg"].get("image_shape")
            if isinstance(sh, list) and len(sh) == 3:
                shape = [int(sh[0]), int(sh[1]), int(sh[2])]
            break
    pp2 = d.get("PostProcess", {}) or {}
    # character_dict may be the empty list for det models, so guard with truthy.
    cd = pp2.get("character_dict", []) or []
    if not isinstance(cd, list):
        cd = []
    return RecCfg(shape=shape, use_space=True, dict=list(cd))


def _extract_cls(d: Dict[str, Any]) -> ClsCfg:
    pp = d.get("PreProcess", {}) or {}
    shape = [3, 80, 160]
    for op in (pp.get("transform_ops") or []):
        if "ResizeImage" in op and isinstance(op["ResizeImage"], dict):
            sz = op["ResizeImage"].get("size")
            # size is [W, H] (e.g. [160, 80]) → shape is [3, H, W] = [3, 80, 160]
            if isinstance(sz, list) and len(sz) == 2:
                shape = [3, int(sz[1]), int(sz[0])]
            break
    pp2 = d.get("PostProcess", {}) or {}
    labels = pp2.get("Topk", {}).get("label_list", []) or ["0_degree", "180_degree"]
    return ClsCfg(
        shape=shape,
        labels=list(labels),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def extract(path: str, *, name: Optional[str] = None,
            kind_hint: Optional[str] = None) -> ExtractedModel:
    """Parse one inference.yml. kind_hint lets the caller force classification."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"{path}: not a YAML mapping at top level")
    name = name or d.get("Global", {}).get("model_name") or p.parent.name
    pp2 = d.get("PostProcess", {}) or {}
    pp_name = pp2.get("name", "")
    if kind_hint:
        kind = kind_hint
    elif pp_name == "CTCLabelDecode":
        kind = "rec"
    elif pp_name == "DBPostProcess":
        kind = "det"
    elif "Topk" in pp2:
        kind = "cls"
    else:
        raise ValueError(f"{path}: cannot infer kind (PostProcess.name={pp_name!r})")
    out = ExtractedModel(name=name, kind=kind, src_path=str(p))
    if kind == "det":
        out.det = _extract_det(name, d)
    elif kind == "rec":
        out.rec = _extract_rec(name, d)
    elif kind == "cls":
        out.cls = _extract_cls(d)
    else:
        raise ValueError(f"{path}: unknown kind {kind}")
    return out


def extract_cls(yml_path: Optional[str] = None) -> ExtractedModel:
    """Special-case cls: model at ~/.paddlex/official_models/PP-LCNet_x1_0_textline_ori."""
    if yml_path is None:
        yml_path = os.path.expanduser(
            "~/.paddlex/official_models/PP-LCNet_x1_0_textline_ori/inference.yml"
        )
    e = extract(yml_path, name="PP-LCNet_x1_0_textline_ori", kind_hint="cls")
    return e


# ---------------------------------------------------------------------------
# Config JSON emission (schema per docs/CONTRACT.md)
# ---------------------------------------------------------------------------

def _det_to_dict(det: DetCfg) -> Dict[str, Any]:
    r = det.resize
    return {
        "thresh": det.thresh,
        "box_thresh": det.box_thresh,
        "unclip_ratio": det.unclip_ratio,
        "max_candidates": det.max_candidates,
        "resize": {
            "mode": r.mode,
            "limit_side_len": r.limit_side_len,
            "resize_long": r.resize_long,
            "stride": r.stride,
            "max_side_limit": r.max_side_limit,
        },
    }


def _rec_to_dict(rec: RecCfg) -> Dict[str, Any]:
    return {
        "shape": list(rec.shape),
        "use_space": rec.use_space,
        "dict": list(rec.dict),
    }


def _cls_to_dict(cls: ClsCfg) -> Dict[str, Any]:
    return {
        "shape": list(cls.shape),
        "labels": list(cls.labels),
        "mean": list(cls.mean),
        "std": list(cls.std),
    }


def build_config(ext: ExtractedModel, *, file: str, sha256: str, bytes_: int,
                 url: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "name": ext.name,
        "type": ext.kind,
        "file": file,
        "sha256": sha256,
        "bytes": int(bytes_),
        "url": url,
    }
    if ext.kind == "det" and ext.det is not None:
        out["det"] = _det_to_dict(ext.det)
    elif ext.kind == "rec" and ext.rec is not None:
        out["rec"] = _rec_to_dict(ext.rec)
    elif ext.kind == "cls" and ext.cls is not None:
        out["cls"] = _cls_to_dict(ext.cls)
    return out


def write_config(out_dir: str, ext: ExtractedModel, *, file: str, sha256: str,
                 bytes_: int, url: str) -> str:
    cfg = build_config(ext, file=file, sha256=sha256, bytes_=bytes_, url=url)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ext.name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="extract_dict",
                                 description="Parse PaddleOCR inference.yml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("one", help="extract a single inference.yml")
    p1.add_argument("yml")
    p1.add_argument("out")
    p1.add_argument("--name", default=None)
    p1.add_argument("--kind", choices=["det", "rec", "cls"], default=None)
    p1.add_argument("--file", default=None,
                    help="filename in model file (used in json); default = <name>.mnn")
    p1.add_argument("--sha256", default="0" * 64)
    p1.add_argument("--bytes", dest="bytes_", type=int, default=0)
    p1.add_argument("--url", default=None)

    p2 = sub.add_parser("cls", help="extract the PP-LCNet_x1_0_textline_ori cls")
    p2.add_argument("out")
    p2.add_argument("--yml", default=None)
    p2.add_argument("--file", default=None)
    p2.add_argument("--sha256", default="0" * 64)
    p2.add_argument("--bytes", dest="bytes_", type=int, default=0)
    p2.add_argument("--url", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "one":
        ext = extract(args.yml, name=args.name, kind_hint=args.kind)
        url = args.url or f"{ext.name}.mnn"
        file = args.file or f"{ext.name}.mnn"
        out = write_config(args.out, ext, file=file, sha256=args.sha256,
                           bytes_=args.bytes_, url=url)
        print(out)
    elif args.cmd == "cls":
        ext = extract_cls(args.yml)
        url = args.url or f"{ext.name}.mnn"
        file = args.file or f"{ext.name}.mnn"
        out = write_config(args.out, ext, file=file, sha256=args.sha256,
                           bytes_=args.bytes_, url=url)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
