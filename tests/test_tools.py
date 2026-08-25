#!/usr/bin/env python3
"""tests/test_tools.py — verification suite for tools/*.

Run:
    cd /root/pp-ws/tools && python3 tests/test_tools.py
or via pytest:
    pytest -q tests/test_tools.py

Tests are pure stdlib (no numpy). They cover:
  * extract_dict: dict-length sanity for 3 representative models
  * extract_dict: det resize policy (v4/v5 resize_long 960, v6 limit_min 736,
    seal resize_long 736)
  * registry.json: 30 entries, all sha256 match the on-disk .mnn
  * score.cer / levenshtein: known cases
  * run_reference.discover_combos: 60 combos (62 total - 2 strip__), no seal/strip
  * end-to-end score on a hand-crafted fake pred/baseline: PASS/FAIL
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from extract_dict import (  # noqa: E402
    extract, extract_cls,
)
from convert_models import (  # noqa: E402
    DET_NAMES, REC_NAMES, SEAL_DET_NAMES, CLS_NAME,
)
from score import levenshtein, cer, score_image  # noqa: E402
from run_reference import discover_combos  # noqa: E402


PPOCR_MODELS = Path("/root/ppocr_models")
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"


class TestExtractDict(unittest.TestCase):

    def test_ppocrv4_mobile_rec_dict_len(self):
        e = extract(str(PPOCR_MODELS / "PP-OCRv4_mobile_rec" / "inference.yml"))
        self.assertEqual(e.kind, "rec")
        # Per AGENTS.md note: 6623
        self.assertEqual(len(e.rec.dict), 6623)
        self.assertEqual(e.rec.shape, [3, 48, 320])
        self.assertTrue(e.rec.use_space)

    def test_ppocrv6_tiny_rec_dict_len(self):
        e = extract(str(PPOCR_MODELS / "PP-OCRv6_tiny_rec" / "inference.yml"))
        self.assertEqual(e.kind, "rec")
        self.assertEqual(len(e.rec.dict), 6904)

    def test_korean_rec_dict_len(self):
        e = extract(str(PPOCR_MODELS / "korean_PP-OCRv5_mobile_rec" / "inference.yml"))
        self.assertEqual(e.kind, "rec")
        self.assertEqual(len(e.rec.dict), 11945)

    def test_v4_det_resize_long_960(self):
        e = extract(str(PPOCR_MODELS / "PP-OCRv4_mobile_det" / "inference.yml"))
        self.assertEqual(e.kind, "det")
        self.assertEqual(e.det.resize.mode, "resize_long")
        self.assertEqual(e.det.resize.resize_long, 960)
        self.assertEqual(e.det.resize.stride, 128)
        self.assertAlmostEqual(e.det.thresh, 0.3)
        self.assertAlmostEqual(e.det.box_thresh, 0.6)
        self.assertAlmostEqual(e.det.unclip_ratio, 1.5)

    def test_v6_tiny_det_limit_min_736(self):
        e = extract(str(PPOCR_MODELS / "PP-OCRv6_tiny_det" / "inference.yml"))
        self.assertEqual(e.kind, "det")
        self.assertEqual(e.det.resize.mode, "limit_min")
        self.assertEqual(e.det.resize.limit_side_len, 736)
        self.assertEqual(e.det.resize.stride, 32)
        self.assertAlmostEqual(e.det.thresh, 0.2)
        self.assertAlmostEqual(e.det.box_thresh, 0.4)
        self.assertAlmostEqual(e.det.unclip_ratio, 1.4)

    def test_v6_small_det_box_thresh_45(self):
        e = extract(str(PPOCR_MODELS / "PP-OCRv6_small_det" / "inference.yml"))
        self.assertEqual(e.kind, "det")
        self.assertAlmostEqual(e.det.box_thresh, 0.45)

    def test_seal_det_resize_long_736(self):
        e = extract(str(PPOCR_MODELS / "PP-OCRv4_mobile_seal_det" / "inference.yml"))
        self.assertEqual(e.kind, "det")
        self.assertEqual(e.det.resize.mode, "resize_long")
        self.assertEqual(e.det.resize.resize_long, 736)
        self.assertAlmostEqual(e.det.unclip_ratio, 0.5)
        # cls model
        cls = extract_cls()
        self.assertEqual(cls.kind, "cls")
        self.assertEqual(cls.cls.shape, [3, 80, 160])
        self.assertEqual(cls.cls.labels, ["0_degree", "180_degree"])
        self.assertEqual(cls.cls.mean, [0.485, 0.456, 0.406])
        self.assertEqual(cls.cls.std, [0.229, 0.224, 0.225])


class TestRegistry(unittest.TestCase):
    """registry.json must be present and complete. sha256 must match .mnn files."""

    @classmethod
    def setUpClass(cls):
        reg_path = CONFIGS_DIR / "registry.json"
        if not reg_path.exists():
            cls.registry = None
            return
        with open(reg_path, "r", encoding="utf-8") as f:
            cls.registry = json.load(f)

    def test_registry_exists(self):
        self.assertIsNotNone(self.registry, "registry.json missing; run convert_models.py")

    def test_registry_30_entries(self):
        if self.registry is None:
            self.skipTest("no registry")
        expected = set(DET_NAMES + SEAL_DET_NAMES + REC_NAMES + [CLS_NAME])
        self.assertEqual(set(self.registry.keys()), expected)
        self.assertEqual(len(self.registry), 30)

    def test_registry_types(self):
        if self.registry is None:
            self.skipTest("no registry")
        dets = {n for n in self.registry if self.registry[n]["type"] == "det"}
        recs = {n for n in self.registry if self.registry[n]["type"] == "rec"}
        clss = {n for n in self.registry if self.registry[n]["type"] == "cls"}
        self.assertEqual(dets, set(DET_NAMES + SEAL_DET_NAMES))
        self.assertEqual(recs, set(REC_NAMES))
        self.assertEqual(clss, {CLS_NAME})

    def test_sha256_matches_mnn_spotcheck(self):
        """Sample 5 models and verify their sha256 in registry matches the .mnn."""
        if self.registry is None:
            self.skipTest("no registry")
        sample = [
            "PP-OCRv4_mobile_rec", "PP-OCRv4_mobile_det",
            "PP-OCRv6_tiny_rec", "PP-OCRv6_tiny_det",
            "korean_PP-OCRv5_mobile_rec", "PP-OCRv4_mobile_seal_det",
            "PP-LCNet_x1_0_textline_ori",
        ]
        for n in sample:
            entry = self.registry[n]
            path = MODELS_DIR / entry["file"]
            self.assertTrue(path.exists(), f"missing .mnn for {n}")
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
            self.assertEqual(entry["sha256"], h.hexdigest(),
                             f"sha256 mismatch for {n}")
            self.assertEqual(entry["bytes"], path.stat().st_size)

    def test_per_model_configs_match_registry(self):
        """For a few models, configs/<name>.json should agree with registry.json."""
        if self.registry is None:
            self.skipTest("no registry")
        for n in ["PP-OCRv4_mobile_rec", "PP-OCRv6_tiny_det",
                  "PP-LCNet_x1_0_textline_ori"]:
            with open(CONFIGS_DIR / f"{n}.json") as f:
                cfg = json.load(f)
            reg = self.registry[n]
            self.assertEqual(cfg["name"], reg["name"])
            self.assertEqual(cfg["type"], reg["type"])
            self.assertEqual(cfg["sha256"], reg["sha256"])
            self.assertEqual(cfg["bytes"], reg["bytes"])
            self.assertEqual(cfg["file"], reg["file"])


class TestLevenshtein(unittest.TestCase):
    def test_known(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("abc", "abc"), 0)
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", ""), 3)
        self.assertEqual(levenshtein("flaw", "lawn"), 2)
        # Unicode (CJK)
        self.assertEqual(levenshtein("深圳", "上海"), 2)
        self.assertEqual(levenshtein("深圳", "深圳"), 0)


class TestCer(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(cer("hello", "hello"), 0.0)
    def test_empty_base(self):
        self.assertEqual(cer("", ""), 0.0)
        self.assertEqual(cer("x", ""), 1.0)
    def test_basic(self):
        # 1 insertion in 4-char base: lev=1, len=4, cer=0.25
        self.assertAlmostEqual(cer("hello", "hell"), 1 / 4, places=6)
        self.assertAlmostEqual(cer("hell", "hello"), 1 / 5, places=6)


class TestScoreEndToEnd(unittest.TestCase):
    """Build a fake baseline + pred under tempdir, run score.image on it."""

    def test_score_image_pass(self):
        # base = "深圳\n上海", pred = "深圳\n上海" → cer=0
        self.assertEqual(score_image(["深圳", "上海"], "深圳\n上海"), 0.0)
    def test_score_image_fail(self):
        # base = "深圳" (2 chars). pred = ["深"] → join = "深" → 1 sub / 2 = 0.5
        c = score_image(["深"], "深圳")
        self.assertAlmostEqual(c, 0.5, places=6)
    def test_score_image_partial(self):
        # base = "foo" (3 chars). pred = ["bar"] → 3 subs / 3 = 1.0 (totally wrong)
        c = score_image(["bar"], "foo")
        self.assertAlmostEqual(c, 1.0, places=6)
    def test_score_image_close(self):
        # base = "深圳" (2). pred = ["深圳"] → 0
        # base = "上海" (2). pred = ["上海"] → 0
        # mean = 0 ≤ 0.05
        c = (score_image(["深圳"], "深圳") + score_image(["上海"], "上海")) / 2
        self.assertLessEqual(c, 0.05)
    def test_score_image_threshold(self):
        # cer > 0.05 → FAIL
        c = score_image(["xxxxx"], "深圳")
        self.assertGreater(c, 0.05)
        # cer < 0.05 → PASS
        c = score_image(["深圳"], "深圳")
        self.assertLessEqual(c, 0.05)


class TestScoreCliE2E(unittest.TestCase):
    """Run score.py as a subprocess against a hand-built fake reference tree."""

    def _make_tree(self, td: Path, *, perfect: bool) -> None:
        (td / "ref" / "DET__REC" / "zh").mkdir(parents=True)
        base = [
            {"image_path": "/i/zh/00.jpg", "rec_texts": ["hello", "world"],
             "rec_scores": [0.9, 0.9], "det_polys": []},
            {"image_path": "/i/zh/01.jpg", "rec_texts": ["foo"],
             "rec_scores": [0.9], "det_polys": []},
        ]
        (td / "ref" / "DET__REC" / "zh" / "ocr_results.json").write_text(
            json.dumps(base))
        if perfect:
            pred = [
                {"image_path": "/i/zh/00.jpg", "rec_texts": ["hello", "world"],
                 "rec_scores": [0.9, 0.9], "det_polys": []},
                {"image_path": "/i/zh/01.jpg", "rec_texts": ["foo"],
                 "rec_scores": [0.9], "det_polys": []},
            ]
        else:
            pred = [
                {"image_path": "/i/zh/00.jpg", "rec_texts": ["hello", "world"],
                 "rec_scores": [0.9, 0.9], "det_polys": []},
                {"image_path": "/i/zh/01.jpg", "rec_texts": ["bar"],
                 "rec_scores": [0.9], "det_polys": []},
            ]
        (td / "pred" / "DET__REC" / "zh").mkdir(parents=True)
        (td / "pred" / "DET__REC" / "zh" / "pred.json").write_text(json.dumps(pred))

    def test_score_cli_pass(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self._make_tree(td, perfect=True)
            # Symlink ref into place expected by score.py is /root/ppocr_reference.
            # So we use a wrapper script that monkey-patches the root via env.
            # Simpler: call score.main_full with patched REF_ROOT by direct import.
            import score  # type: ignore
            old_root = score.REF_ROOT
            try:
                score.REF_ROOT = td / "ref"
                rc = score.main_full(td / "pred", threshold=0.05,
                                     report_path=td / "report.md")
            finally:
                score.REF_ROOT = old_root
            self.assertEqual(rc, 0)
            self.assertTrue((td / "report.md").exists())
            txt = (td / "report.md").read_text()
            self.assertIn("PASS", txt)
            self.assertIn("FAIL=0", txt)

    def test_score_cli_fail(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self._make_tree(td, perfect=False)
            import score  # type: ignore
            old_root = score.REF_ROOT
            try:
                score.REF_ROOT = td / "ref"
                rc = score.main_full(td / "pred", threshold=0.05,
                                     report_path=td / "report.md")
            finally:
                score.REF_ROOT = old_root
            self.assertEqual(rc, 1)
            txt = (td / "report.md").read_text()
            self.assertIn("FAIL", txt)


class TestDiscoverCombos(unittest.TestCase):
    def test_count_and_filter(self):
        combos = discover_combos()
        # 62 total dirs under /root/ppocr_reference; -2 strip__ = 60
        self.assertEqual(len(combos), 60)
        for c in combos:
            name = c[0]
            self.assertIn("__", name)
            self.assertFalse(name.startswith("strip__"))
        # regex filter
        only_v4 = discover_combos(r"^PP-OCRv4_")
        self.assertGreater(len(only_v4), 0)
        for n, det, rec in only_v4:
            self.assertTrue(n.startswith("PP-OCRv4_"))


class TestConfigSchema(unittest.TestCase):
    """Spot-check that emitted configs/<name>.json matches CONTRACT schema."""

    def test_det_schema(self):
        with open(CONFIGS_DIR / "PP-OCRv4_mobile_det.json") as f:
            d = json.load(f)
        self.assertEqual(d["name"], "PP-OCRv4_mobile_det")
        self.assertEqual(d["type"], "det")
        self.assertTrue(d["file"].endswith(".mnn"))
        self.assertEqual(len(d["sha256"]), 64)
        self.assertIsInstance(d["bytes"], int)
        self.assertIn("det", d)
        for k in ("thresh", "box_thresh", "unclip_ratio", "max_candidates", "resize"):
            self.assertIn(k, d["det"])
        r = d["det"]["resize"]
        for k in ("mode", "limit_side_len", "resize_long", "stride", "max_side_limit"):
            self.assertIn(k, r)
        self.assertIn(r["mode"], ("limit_min", "resize_long"))

    def test_rec_schema(self):
        with open(CONFIGS_DIR / "PP-OCRv4_mobile_rec.json") as f:
            d = json.load(f)
        self.assertEqual(d["type"], "rec")
        self.assertIn("rec", d)
        self.assertEqual(d["rec"]["shape"], [3, 48, 320])
        self.assertTrue(d["rec"]["use_space"])
        self.assertIsInstance(d["rec"]["dict"], list)
        self.assertGreater(len(d["rec"]["dict"]), 0)

    def test_cls_schema(self):
        with open(CONFIGS_DIR / f"{CLS_NAME}.json") as f:
            d = json.load(f)
        self.assertEqual(d["type"], "cls")
        self.assertIn("cls", d)
        self.assertEqual(d["cls"]["shape"], [3, 80, 160])
        self.assertEqual(d["cls"]["labels"], ["0_degree", "180_degree"])


def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
