# PP-OCR CER report

Threshold: CER ≤ 0.05

## Main matrix (7×7) — lang-averaged CER

Rows = `det`, cols = `rec`. Cell = mean CER across the 16 languages. Status: PASS / FAIL.

| det \\ rec | PP-OCRv4_mobile_rec | PP-OCRv4_server_rec | PP-OCRv5_mobile_rec | PP-OCRv5_server_rec | PP-OCRv6_tiny_rec | PP-OCRv6_small_rec | PP-OCRv6_medium_rec |
|---|---|---|---|---|---|---|---|
| PP-OCRv4_mobile_det | 0.4254 FAIL | 0.3365 FAIL | 0.3691 FAIL | 0.3234 FAIL | 0.3105 FAIL | 0.3116 FAIL | 0.3621 FAIL |
| PP-OCRv4_server_det | 0.2472 FAIL | 0.2351 FAIL | 0.9313 FAIL | 0.8084 FAIL | 0.2623 FAIL | 0.2313 FAIL | 0.2480 FAIL |
| PP-OCRv5_mobile_det | 0.3564 FAIL | 0.3304 FAIL | 0.3546 FAIL | 0.2689 FAIL | 0.2919 FAIL | 0.3229 FAIL | 0.3442 FAIL |
| PP-OCRv5_server_det | 0.3396 FAIL | 0.3621 FAIL | 0.2934 FAIL | 0.3312 FAIL | 0.3724 FAIL | 0.3441 FAIL | 0.3604 FAIL |
| PP-OCRv6_tiny_det | 0.2357 FAIL | 0.2314 FAIL | 0.2478 FAIL | 0.2198 FAIL | 0.2414 FAIL | 0.2115 FAIL | 0.2159 FAIL |
| PP-OCRv6_small_det | 0.3020 FAIL | 0.2360 FAIL | 0.2362 FAIL | 0.2392 FAIL | 0.2769 FAIL | 0.2818 FAIL | 0.2953 FAIL |
| PP-OCRv6_medium_det | 0.2122 FAIL | 0.1971 FAIL | 0.2065 FAIL | 0.1989 FAIL | 0.2006 FAIL | 0.1897 FAIL | 0.2057 FAIL |

**Main matrix (49 cells):** PASS=0  FAIL=49  N/A=0

## Lang-rec block (per-cell CER)

| combo | langs scored | mean CER | status |
|---|---|---|---|
| PP-OCRv4_mobile_det__en_PP-OCRv4_mobile_rec | 1 | N/A | N/A |
| PP-OCRv5_mobile_det__arabic_PP-OCRv5_mobile_rec | 2 | N/A | N/A |
| PP-OCRv5_mobile_det__cyrillic_PP-OCRv5_mobile_rec | 2 | 1.0000 | FAIL |
| PP-OCRv5_mobile_det__devanagari_PP-OCRv5_mobile_rec | 2 | 0.0000 | PASS |
| PP-OCRv5_mobile_det__el_PP-OCRv5_mobile_rec | 2 | N/A | N/A |
| PP-OCRv5_mobile_det__en_PP-OCRv5_mobile_rec | 1 | N/A | N/A |
| PP-OCRv5_mobile_det__eslav_PP-OCRv5_mobile_rec | 2 | 1.0000 | FAIL |
| PP-OCRv5_mobile_det__korean_PP-OCRv5_mobile_rec | 2 | N/A | N/A |
| PP-OCRv5_mobile_det__latin_PP-OCRv5_mobile_rec | 2 | N/A | N/A |
| PP-OCRv5_mobile_det__th_PP-OCRv5_mobile_rec | 2 | 1.0000 | FAIL |

## Doc-rec block (PP-OCRv4_server_rec_doc)

| combo | langs scored | mean CER | status |
|---|---|---|---|
| PP-OCRv4_server_det__PP-OCRv4_server_rec_doc | 5 | 0.0000 | PASS |

## Schema warnings

- PP-OCRv4_server_det__PP-OCRv4_mobile_rec: 9 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv4_server_det__PP-OCRv4_server_rec: 9 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv4_server_det__PP-OCRv5_mobile_rec: 9 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv4_server_det__PP-OCRv5_server_rec: 9 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv4_server_det__PP-OCRv6_medium_rec: 9 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv4_server_det__PP-OCRv6_small_rec: 9 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv4_server_det__PP-OCRv6_tiny_rec: 9 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv5_mobile_det__PP-OCRv4_mobile_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv5_mobile_det__PP-OCRv4_server_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv5_mobile_det__PP-OCRv5_mobile_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv5_mobile_det__PP-OCRv5_server_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv5_mobile_det__PP-OCRv6_medium_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv5_mobile_det__PP-OCRv6_small_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv5_mobile_det__PP-OCRv6_tiny_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_medium_det__PP-OCRv4_mobile_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_medium_det__PP-OCRv4_server_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_medium_det__PP-OCRv5_mobile_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_medium_det__PP-OCRv5_server_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_medium_det__PP-OCRv6_medium_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_medium_det__PP-OCRv6_small_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_medium_det__PP-OCRv6_tiny_rec: 27 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_small_det__PP-OCRv4_mobile_rec: 26 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_small_det__PP-OCRv4_server_rec: 26 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_small_det__PP-OCRv5_mobile_rec: 26 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_small_det__PP-OCRv5_server_rec: 26 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_small_det__PP-OCRv6_medium_rec: 26 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_small_det__PP-OCRv6_small_rec: 26 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_small_det__PP-OCRv6_tiny_rec: 26 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_tiny_det__PP-OCRv4_mobile_rec: 25 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_tiny_det__PP-OCRv4_server_rec: 25 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_tiny_det__PP-OCRv5_mobile_rec: 25 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_tiny_det__PP-OCRv5_server_rec: 25 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_tiny_det__PP-OCRv6_medium_rec: 25 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_tiny_det__PP-OCRv6_small_rec: 25 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)
- PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec: 25 invalid-baseline + 0 missing-pred (backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)

---
**Total:** PASS=2  FAIL=52  N/A=6  (cells=60)
