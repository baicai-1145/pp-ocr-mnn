#!/usr/bin/env python3
"""Unit tests for score.py matched_line_cer (decision-maker spec, no slack).

Spec:
  1. Greedy 1-1 matching: at each step pick the pred/base pair with the
     MINIMAL normalized Levenshtein (lev / max(len) or len(b)), tie-break by
     index order; each line can be used once.
  2. CER = [ sum(lev(p,b) over matched pairs)
           + sum(len(b) for unmatched base)
           + sum(len(p) for unmatched pred) ] / sum(len(b) over all base lines)
     - unmatched base = missed detection (full length charged)
     - unmatched pred = extra detection (full length charged)
  3. join-CER is reported alongside as reference only.
"""
import sys, unittest
sys.path.insert(0, "/root/pp-ocr-mnn/tools")
from score import matched_line_cer, levenshtein


class TestMatchedLineCER(unittest.TestCase):
    def test_perfect(self):
        mlc, jc = matched_line_cer(["ABC", "DE"], ["ABC", "DE"])
        self.assertEqual(mlc, 0.0)

    def test_single_char_error(self):
        mlc, _ = matched_line_cer(["ABX"], ["ABY"])   # lev=1, len=3
        self.assertAlmostEqual(mlc, 1/3)

    def test_missing_line_full_charge(self):
        # 2 base lines, 1 pred. pred matches 'ABC' exactly;
        # missing 'DEF' charges full len 3. denom = 3+3=6.
        mlc, _ = matched_line_cer(["ABC"], ["ABC", "DEF"])
        self.assertAlmostEqual(mlc, 3/6)

    def test_extra_line_charged_full(self):
        # extra pred 'XY' has no partner -> add len 2 to numerator,
        # denominator unchanged (base total only).
        mlc, _ = matched_line_cer(["ABC", "XY"], ["ABC"])
        self.assertAlmostEqual(mlc, 2/3)

    def test_user_example_3base_2pred(self):
        # base = [ABC(3), DEF(3), GHI(3)], pred = [ABC, DxF]
        # match ABC->ABC lev 0; DxF vs DEF lev 1; DEF and GHI unmatched?
        # greedy picks min first: ABC/ABC = 0. Next best: DxF/DEF norm = 1/3;
        # DxF/GHI = 3/3; so pair DxF-DEF. Unmatched base GHI charges 3.
        # numerator = 0 + 1 + 3 = 4; denominator = 9.
        mlc, _ = matched_line_cer(["ABC", "DxF"], ["ABC", "DEF", "GHI"])
        self.assertAlmostEqual(mlc, 4/9)

    def test_empty_pred_all_base_missed(self):
        mlc, _ = matched_line_cer([], ["ABC", "DEF"])
        self.assertAlmostEqual(mlc, 1.0)

    def test_empty_base_extra_pred_capped(self):
        # no base -> denominator 0 => conventionally 0 if no pred else 1.0
        mlc, _ = matched_line_cer(["XYZ"], [])
        self.assertEqual(mlc, 1.0)

    def test_greedy_prefers_minimal_pair(self):
        # decoy near-match must not steal the exact match:
        # base=[ABC, ABD], pred=[ABC]. ABC matches itself (0). ABD unmatched (3).
        # If greedy wrongly paired ABD first, numer would differ.
        mlc, _ = matched_line_cer(["ABC"], ["ABC", "ABD"])
        self.assertAlmostEqual(mlc, 3/6)

    def test_duplicate_lines_one_to_one(self):
        # two identical base lines, one pred: only one can match.
        mlc, _ = matched_line_cer(["AA"], ["AA", "AA"])
        self.assertAlmostEqual(mlc, 2/4)

    def test_both_empty(self):
        mlc, _ = matched_line_cer([], [])
        self.assertEqual(mlc, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
