# PP-OCR CER report

Threshold: CER ≤ 0.05

## Main matrix (7×7) — lang-averaged CER

Rows = `det`, cols = `rec`. Cell = mean CER across the 16 languages. Status: PASS / FAIL.

| det \\ rec | PP-OCRv4_mobile_rec | PP-OCRv4_server_rec | PP-OCRv5_mobile_rec | PP-OCRv5_server_rec | PP-OCRv6_tiny_rec | PP-OCRv6_small_rec | PP-OCRv6_medium_rec |
|---|---|---|---|---|---|---|---|
| PP-OCRv4_mobile_det | 0.4254 FAIL | 0.3365 FAIL | 0.3691 FAIL | 0.3234 FAIL | 0.3105 FAIL | 0.3116 FAIL | 0.3621 FAIL |
| PP-OCRv4_server_det | 0.2729 FAIL | 0.2563 FAIL | 0.9875 FAIL | 0.8647 FAIL | 0.2871 FAIL | 0.2711 FAIL | 0.2813 FAIL |
| PP-OCRv5_mobile_det | 0.3236 FAIL | 0.3066 FAIL | 0.3194 FAIL | 0.2991 FAIL | 0.3350 FAIL | 0.3136 FAIL | 0.3308 FAIL |
| PP-OCRv5_server_det | 0.3396 FAIL | 0.3621 FAIL | 0.2934 FAIL | 0.3312 FAIL | 0.3724 FAIL | 0.3441 FAIL | 0.3604 FAIL |
| PP-OCRv6_tiny_det | 0.2895 FAIL | 0.2746 FAIL | 0.2918 FAIL | 0.2697 FAIL | 0.2881 FAIL | 0.2632 FAIL | 0.2670 FAIL |
| PP-OCRv6_small_det | 0.3103 FAIL | 0.2865 FAIL | 0.2843 FAIL | 0.2929 FAIL | 0.2985 FAIL | 0.3046 FAIL | 0.2843 FAIL |
| PP-OCRv6_medium_det | 0.2854 FAIL | 0.2526 FAIL | 0.2597 FAIL | 0.2537 FAIL | 0.2572 FAIL | 0.2506 FAIL | 0.2729 FAIL |

**Main matrix (49 cells):** PASS=0  FAIL=49  N/A=0

## Lang-rec block (per-cell CER)

| combo | langs scored | mean CER | status |
|---|---|---|---|
| PP-OCRv4_mobile_det__en_PP-OCRv4_mobile_rec | 1 | 0.1722 | FAIL |
| PP-OCRv5_mobile_det__arabic_PP-OCRv5_mobile_rec | 2 | 0.3367 | FAIL |
| PP-OCRv5_mobile_det__cyrillic_PP-OCRv5_mobile_rec | 2 | 0.3658 | FAIL |
| PP-OCRv5_mobile_det__devanagari_PP-OCRv5_mobile_rec | 2 | 0.2303 | FAIL |
| PP-OCRv5_mobile_det__el_PP-OCRv5_mobile_rec | 2 | 0.2072 | FAIL |
| PP-OCRv5_mobile_det__en_PP-OCRv5_mobile_rec | 1 | 0.1612 | FAIL |
| PP-OCRv5_mobile_det__eslav_PP-OCRv5_mobile_rec | 2 | 0.4310 | FAIL |
| PP-OCRv5_mobile_det__korean_PP-OCRv5_mobile_rec | 2 | 0.2258 | FAIL |
| PP-OCRv5_mobile_det__latin_PP-OCRv5_mobile_rec | 2 | 0.2237 | FAIL |
| PP-OCRv5_mobile_det__th_PP-OCRv5_mobile_rec | 2 | 0.3021 | FAIL |

## Doc-rec block (PP-OCRv4_server_rec_doc)

| combo | langs scored | mean CER | status |
|---|---|---|---|
| PP-OCRv4_server_det__PP-OCRv4_server_rec_doc | 5 | 0.9800 | FAIL |

---
**Total:** PASS=0  FAIL=60  N/A=0  (cells=60)
