# PP-OCR CER report

Threshold: CER ≤ 0.05

## Main matrix (7×7) — lang-averaged CER

Rows = `det`, cols = `rec`. Cell = mean CER across the 16 languages. Status: PASS / FAIL.

| det \\ rec | PP-OCRv4_mobile_rec | PP-OCRv4_server_rec | PP-OCRv5_mobile_rec | PP-OCRv5_server_rec | PP-OCRv6_tiny_rec | PP-OCRv6_small_rec | PP-OCRv6_medium_rec |
|---|---|---|---|---|---|---|---|
| PP-OCRv4_mobile_det | 0.4096 FAIL | 0.3186 FAIL | 0.3541 FAIL | 0.3080 FAIL | 0.2901 FAIL | 0.2883 FAIL | 0.3380 FAIL |
| PP-OCRv4_server_det | 0.2586 FAIL | 0.2407 FAIL | 0.2517 FAIL | 0.2424 FAIL | 0.2746 FAIL | 0.2399 FAIL | 0.2510 FAIL |
| PP-OCRv5_mobile_det | 0.3139 FAIL | 0.2928 FAIL | 0.3118 FAIL | 0.2876 FAIL | 0.3197 FAIL | 0.2927 FAIL | 0.3056 FAIL |
| PP-OCRv5_server_det | 0.3251 FAIL | 0.3460 FAIL | 0.2783 FAIL | 0.3143 FAIL | 0.3524 FAIL | 0.3253 FAIL | 0.3321 FAIL |
| PP-OCRv6_tiny_det | 0.2818 FAIL | 0.2521 FAIL | 0.2824 FAIL | 0.2464 FAIL | 0.2722 FAIL | 0.2494 FAIL | 0.2457 FAIL |
| PP-OCRv6_small_det | 0.2975 FAIL | 0.2721 FAIL | 0.2696 FAIL | 0.2783 FAIL | 0.2821 FAIL | 0.2856 FAIL | 0.2597 FAIL |
| PP-OCRv6_medium_det | 0.2659 FAIL | 0.2321 FAIL | 0.2393 FAIL | 0.2327 FAIL | 0.2351 FAIL | 0.2261 FAIL | 0.2471 FAIL |

**Main matrix (49 cells):** PASS=0  FAIL=49  N/A=0

## Lang-rec block (per-cell CER)

| combo | langs scored | mean CER | status |
|---|---|---|---|
| PP-OCRv4_mobile_det__en_PP-OCRv4_mobile_rec | 1 | 0.1682 | FAIL |
| PP-OCRv5_mobile_det__arabic_PP-OCRv5_mobile_rec | 2 | 0.3344 | FAIL |
| PP-OCRv5_mobile_det__cyrillic_PP-OCRv5_mobile_rec | 2 | 0.3499 | FAIL |
| PP-OCRv5_mobile_det__devanagari_PP-OCRv5_mobile_rec | 2 | 0.2299 | FAIL |
| PP-OCRv5_mobile_det__el_PP-OCRv5_mobile_rec | 2 | 0.2074 | FAIL |
| PP-OCRv5_mobile_det__en_PP-OCRv5_mobile_rec | 1 | 0.1605 | FAIL |
| PP-OCRv5_mobile_det__eslav_PP-OCRv5_mobile_rec | 2 | 0.3934 | FAIL |
| PP-OCRv5_mobile_det__korean_PP-OCRv5_mobile_rec | 2 | 0.1986 | FAIL |
| PP-OCRv5_mobile_det__latin_PP-OCRv5_mobile_rec | 2 | 0.2166 | FAIL |
| PP-OCRv5_mobile_det__th_PP-OCRv5_mobile_rec | 2 | 0.2990 | FAIL |

## Doc-rec block (PP-OCRv4_server_rec_doc)

| combo | langs scored | mean CER | status |
|---|---|---|---|
| PP-OCRv4_server_det__PP-OCRv4_server_rec_doc | 5 | 0.1973 | FAIL |

---
**Total:** PASS=0  FAIL=60  N/A=0  (cells=60)
