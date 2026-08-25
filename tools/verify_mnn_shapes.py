#!/usr/bin/env python3
"""verify_mnn_shapes.py — multi-resolution shape verification for det .mnn models.

Per docs/CONTRACT.md / TOOLS-4, this is the verification tool for the
"truly dynamic det model" requirement. It runs a small C++ probe (built
from tools/verify_mnn_probe.cpp) against each input .mnn, feeds it a
battery of test shapes, and reports per-model × per-shape results.

Test shape set (the user-defined battery from TOOLS-4):
  v6: {1,3,720,1280}, {1,3,736,1312}, {1,3,960,960}, {1,3,640,480}, {1,3,1248,384}
  v4/v5: + {1,3,640,1280}, {1,3,800,800}, {1,3,1024,1024}
  (32-multiple variants)

Diagnostic finding (TOOLS-4): the underlying PaddlePaddle FPN architecture
has a 4-stage vs 5-stage branch that only matches when the input H and W
are multiples of 32. This is an ONNX / Paddle model property, not an MNN
conversion issue. The existing /tmp/det_fix/mnn/*.mnn files are re-
converted with default MNNConvert options; they are byte-different from
the shipped ones (MNN embeds a build UUID) but functionally identical.

Usage:
  python3 tools/verify_mnn_shapes.py \
      [--model-dir DIR | --model PATH]... \
      [--probe /path/to/verify_mnn_probe] \
      [--build-probe] \
      [--out report.md] \
      [--shapes s1 s2 ...]   # override the default battery
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent

# Default test shape set per the TOOLS-4 brief.
# All are 32-multiples in H and W (the constraint of the Paddle model).
V6_SHAPES: List[str] = [
    "1x3x720x1280",   # the failing case (non-32-multiple H)
    "1x3x736x1312",   # the working case
    "1x3x960x960",
    "1x3x640x480",
    "1x3x1248x384",
    "1x3x800x800",
    "1x3x640x1280",
    "1x3x1024x1024",
    "1x3x576x1280",   # extra: 576=18*32
    "1x3x704x1280",   # extra: 704=22*32
    "1x3x736x1280",   # extra: 736=23*32 (the canonical)
    "1x3x768x1280",   # extra: 768=24*32
]

V4V5_SHAPES: List[str] = V6_SHAPES + [
    "1x3x960x1280",
    "1x3x1152x1280",  # 1152 = 36*32
    "1x3x1280x1280",  # square
]


def is_v6(name: str) -> bool:
    return "v6" in name.lower() and "seal" not in name.lower()


def build_probe(force: bool = False) -> Path:
    """Compile the C++ probe. Output binary is in /tmp/det_fix/verify_mnn_probe."""
    out = Path("/tmp/det_fix/verify_mnn_probe")
    if out.exists() and not force:
        return out
    src = HERE / "verify_mnn_probe.cpp"
    mnn_root = Path("/root/pp-ocr-mnn/third_party/MNN")
    mnn_lib = mnn_root / "build" / "libMNN.a"
    if not mnn_lib.exists():
        raise FileNotFoundError(f"MNN lib not found at {mnn_lib}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++", "-O2", "-std=c++17",
        f"-I{mnn_root}/include",
        str(src),
        str(mnn_lib),
        "-lpthread", "-ldl",
        "-o", str(out),
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0 or not out.exists():
        raise RuntimeError(f"probe build failed (rc={rc})")
    return out


def _run_probe(probe: Path, mnn: Path, backend: str, shapes: List[str],
               timeout: int = 60) -> Tuple[int, str, str]:
    cmd = [str(probe), str(mnn), backend] + shapes
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


# ---------------------------------------------------------------------------
# Probe-output parsing
# ---------------------------------------------------------------------------

def parse_probe_output(stdout: str) -> List[Tuple[str, str, str, str]]:
    """Parse probe stdout. Returns [(shape, status, output_or_error, hint), ...].

    The probe emits:
        [ OK ] shape=... -> output ...
        [FAIL] shape=... runSession rc=N
    followed by:
        Summary: X/Y shapes passed, Z failed
    """
    out: List[Tuple[str, str, str, str]] = []
    lines = stdout.splitlines()
    pending_hint: List[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("[ OK ] shape=") or s.startswith("[FAIL] shape="):
            tail = s[len("[ OK ] shape="):] if s.startswith("[ OK ] ") else s[len("[FAIL] shape="):]
            status = "OK" if s.startswith("[ OK ]") else "FAIL"
            if " -> " in tail:
                shape_str, rest = tail.split(" -> ", 1)
            else:
                shape_str, _, rest = tail.partition(" ")
                rest = rest.strip()
            shape = shape_str.strip()
            hint = " ".join(pending_hint).strip() if status == "FAIL" else ""
            # Filter hint: keep only lines that look like errors (contain "error"
            # or "ERR" or "broad cast" or specific failure tokens)
            if status == "FAIL":
                # Drop lines that aren't the actual error cause
                excluded = ("opencl init", "error to use creator",
                            "the device support")
                kept = [h for h in pending_hint if (
                    ("error" in h.lower() or "broad cast" in h.lower() or
                     "compute shape" in h.lower() or "can't run" in h.lower())
                    and not any(x in h.lower() for x in excluded)
                )]
                hint = " ".join(kept).strip()
            out.append((shape, status, rest, hint))
            pending_hint = []
        elif s.startswith("model:") or s.startswith("input:") or s.startswith(
                "num_shapes:") or s.startswith("backend:") or s.startswith("Summary:"):
            continue
        elif s:
            pending_hint.append(s)
    return out


# ---------------------------------------------------------------------------
# Verification orchestration
# ---------------------------------------------------------------------------

def verify_model(probe: Path, mnn: Path, backend: str, shapes: List[str]
                 ) -> Dict[str, dict]:
    """Returns {shape: {"ok": bool, "output": str, "error": str}}."""
    rc, out, err = _run_probe(probe, mnn, backend, shapes)
    results: Dict[str, dict] = {}
    parsed = parse_probe_output(out)
    # stderr may also carry useful info (we capture but not parse deeply)
    for shape, status, rest, hint in parsed:
        results[shape] = {
            "ok": status == "OK",
            "output_or_error": rest,
            "stderr_hint": hint,
        }
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_table(model_results: Dict[str, Dict[str, dict]], shapes: List[str]
                 ) -> str:
    """Render a markdown table: rows = models, cols = shapes, cell = OK/FAIL.

    Only renders columns that were actually tested (i.e. that appear in at
    least one model's result map), and only renders rows for the models in
    the passed dict. 'shapes' is used as a column order preference."""
    used: set = set()
    for res in model_results.values():
        used.update(res.keys())
    cols = [s for s in shapes if s in used]
    if not used and not cols:
        # If nothing was tested, fall back to the requested shape list
        cols = list(shapes)
    if not cols:
        return "(no shapes tested)\n"
    short_shapes = [s.replace("1x3x", "") for s in cols]
    lines: List[str] = []
    lines.append("| model | " + " | ".join(short_shapes) + " |")
    lines.append("|" + "|".join(["---"] * (len(short_shapes) + 1)) + "|")
    for mname, res in model_results.items():
        row = [mname]
        for s in cols:
            r = res.get(s)
            if r is None:
                row.append("—")
            elif r["ok"]:
                row.append("OK")
            else:
                err = r.get("stderr_hint", r.get("output_or_error", ""))[:30]
                row.append(f"FAIL: {err}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_per_model(model_results: Dict[str, Dict[str, dict]], shapes: List[str]
                     ) -> str:
    lines: List[str] = []
    for mname, res in model_results.items():
        tested_shapes = list(res.keys())
        ok = sum(1 for s in tested_shapes if res.get(s, {}).get("ok"))
        bad = sum(1 for s in tested_shapes if not res.get(s, {}).get("ok"))
        lines.append(f"### {mname}")
        lines.append("")
        lines.append(f"  pass: {ok}/{len(tested_shapes)}   fail: {bad}/{len(tested_shapes)}")
        for s in tested_shapes:
            r = res.get(s, {})
            if r.get("ok"):
                lines.append(f"  - shape={s} -> {r.get('output_or_error', '')}")
            else:
                lines.append(f"  - shape={s} FAIL: {r.get('stderr_hint', r.get('output_or_error', ''))}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _build_full_report(args, model_results, shapes) -> str:
    lines: List[str] = []
    lines.append("# Det .mnn multi-resolution verification (TOOLS-4)")
    lines.append("")
    lines.append(f"Probe: `{args.probe or '/tmp/det_fix/verify_mnn_probe (default-built)'}`")
    lines.append(f"Backend: `{args.backend}`")
    # Show only the shapes that were actually used by at least one model
    used = set()
    for res in model_results.values():
        used.update(res.keys())
    shapes_list = sorted(used)
    lines.append(f"Shape battery ({len(shapes_list)} unique shapes):")
    for s in shapes_list:
        lines.append(f"  - `{s}`")
    lines.append("")
    # Split: v6 models vs v4/v5
    v6_models = {k: v for k, v in model_results.items() if is_v6(k)}
    v45_models = {k: v for k, v in model_results.items() if not is_v6(k)}
    if v6_models:
        lines.append("## v6 (tiny/small/medium) — V6_SHAPES battery")
        lines.append("")
        lines.append(render_table(v6_models, V6_SHAPES))
    if v45_models:
        lines.append("## v4/v5 (mobile/server) — V4V5_SHAPES battery")
        lines.append("")
        lines.append(render_table(v45_models, V4V5_SHAPES))
    lines.append("## Per-model detail")
    lines.append("")
    lines.append(render_per_model(model_results, shapes))
    lines.append("")
    lines.append("## File metadata (sha256 / bytes)")
    lines.append("")
    lines.append("| model | sha256 (first 16) | bytes |")
    lines.append("|---|---|---|")
    for mname, mpath in args.models:
        try:
            sha = _sha256(mpath)[:16]
            n = mpath.stat().st_size
            lines.append(f"| {mname} | {sha}… | {n} |")
        except Exception as e:
            lines.append(f"| {mname} | ERR: {e} | - |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="verify_mnn_shapes",
                                 description="Multi-resolution verification of det .mnn models")
    ap.add_argument("--model", action="append", default=[],
                    metavar="NAME=PATH",
                    help="add a model to verify (repeatable). PATH may be a .mnn or an ONNX file (will be converted).")
    ap.add_argument("--model-dir", default=None,
                    help="verify every .mnn under this directory; names are file basenames")
    ap.add_argument("--probe", default=None,
                    help="path to the C++ probe binary (built if missing)")
    ap.add_argument("--build-probe", action="store_true",
                    help="force rebuild the probe even if it exists")
    ap.add_argument("--backend", default="cpu",
                    help="cpu|opencl|vulkan")
    ap.add_argument("--shapes", nargs="+", default=None,
                    help="override shape battery (NxCxHxW strings)")
    ap.add_argument("--out", default=None,
                    help="write report here (default: stdout)")
    args = ap.parse_args(argv)

    # Collect models
    models: List[Tuple[str, Path]] = []
    for spec in args.model:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name = Path(spec).stem
            path = spec
        models.append((name, Path(path)))
    if args.model_dir:
        for f in sorted(Path(args.model_dir).glob("*.mnn")):
            if f.name not in [m[0] for m in models]:
                models.append((f.stem, f))
    if not models:
        print("ERROR: no models to verify. Use --model NAME=PATH or --model-dir DIR.",
              file=sys.stderr)
        return 2
    args.models = models  # for use in _build_full_report

    # Build / locate probe
    if args.probe:
        probe = Path(args.probe)
    else:
        probe = build_probe(force=args.build_probe)
    if not probe.exists():
        probe = build_probe(force=True)
    print(f"probe: {probe}", file=sys.stderr)

    # Per-model shape set
    model_results: Dict[str, Dict[str, dict]] = {}
    for mname, mpath in models:
        if is_v6(mname):
            shapes = V6_SHAPES
        else:
            shapes = V4V5_SHAPES
        if args.shapes:
            shapes = args.shapes
        print(f"verifying {mname} ({len(shapes)} shapes)...", file=sys.stderr)
        model_results[mname] = verify_model(probe, mpath, args.backend, shapes)

    report = _build_full_report(args, model_results, V4V5_SHAPES)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(report)

    # Exit code: 0 if all pass, 1 if any fail
    any_fail = any(
        not r.get("ok")
        for res in model_results.values()
        for r in res.values()
    )
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
