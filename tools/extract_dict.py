#!/usr/bin/env python3
"""extract_dict.py — parse PaddleOCR inference.yml → dict + det/rec/cls params.

Library API (used by convert_models.py):
    extract(path) → ExtractedModel (with .kind == 'det'|'rec'|'cls')
    write_config(out_dir, name, ext: ExtractedModel) → path to configs/<name>.json

CLI:
    extract_dict.py INFERENCE_YML OUT_JSON [--name NAME]
    extract_dict.py --cls OUT_JSON  (uses ~/.paddlex/official_models/PP-LCNet_x1_0_textline_ori)

Schema for emitted JSON strictly follows docs/CONTRACT.md.

Det resize policy:
    DetResizeForTest: null               → mode="limit_min", limit_side_len=736, stride=32
    DetResizeForTest: { resize_long: L } → mode="resize_long", resize_long=L, stride=128
    (v4/v5 det: resize_long=960, stride=128;
     v6 det: DetResizeForTest=null → limit_min 736, stride=32;
     seal det: resize_long=736, stride=128)
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
# Resize-mode policy (verified vs AGENTS.md / CONTRACT)
# ---------------------------------------------------------------------------

def _resize_policy(model_name: str, resize_node: Optional[Dict[str, Any]]) -> DetResizeCfg:
    """Decide resize mode + stride. v6 uses LimitMin (DetResizeForTest: null);
    v4/v5 and seal use ResizeLong with stride 128 (per CONTRACT preprocess port).
    """
    is_v6 = "v6" in model_name.lower()
    is_seal = "seal" in model_name.lower()

    if is_v6 and not is_seal:
        # v6 det: null → LimitMin 736, stride 32 (32-align per CONTRACT).
        return DetResizeCfg(mode="limit_min", limit_side_len=736, stride=32,
                            max_side_limit=4000)

    # resize_long path (v4/v5 det, all seal det)
    if resize_node and "resize_long" in resize_node:
        long_ = int(resize_node["resize_long"])
    else:
        long_ = 736 if is_seal else 960
    return DetResizeCfg(mode="resize_long", limit_side_len=736,
                        resize_long=long_, stride=128, max_side_limit=4000)


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
    pp2 = d.get("PostProcess", {}) or {}
    return DetCfg(
        thresh=float(pp2.get("thresh", 0.3)),
        box_thresh=float(pp2.get("box_thresh", 0.6)),
        unclip_ratio=float(pp2.get("unclip_ratio", 1.5)),
        max_candidates=int(pp2.get("max_candidates", 1000)),
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
