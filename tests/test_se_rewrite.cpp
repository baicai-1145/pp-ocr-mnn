// test_se_rewrite — SE-pool rewrite variant policy (task-7).
//
// Pure-header test: no MNN, no models, no filesystem. It pins the parts of
// the dispatch decision that are easy to regress:
//   * only the four measured-good dets are allowlisted;
//   * the models we deliberately excluded (v4_server / v6_medium /
//     v5_server) stay excluded, because the rewrite regresses or is a no-op
//     on them;
//   * rec / cls models are never eligible (only det has the SE pooling);
//   * the variant file name is exactly "<name>.red.mnn", which is what
//     tools/rewrite_se_pool.py + the deployer produce.
#include <cstdio>
#include <string>

#include "ppocr/se_rewrite.h"

static int g_failures = 0;

static void expect(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

int main() {
  using ppocr::det_has_se_rewrite_variant;
  using ppocr::se_rewrite_variant_name;

  // --- allowlist: the four models with a measured Metal win ---------------
  const char* kExpected[] = {
      "PP-OCRv6_tiny_det",
      "PP-OCRv6_small_det",
      "PP-OCRv5_mobile_det",
      "PP-OCRv4_mobile_det",
  };
  for (const char* n : kExpected) {
    expect(det_has_se_rewrite_variant(n),
           std::string("expected allowlisted: ") + n);
  }

  // --- exclusions: regressed or no-op, must never be selected ------------
  const char* kExcluded[] = {
      "PP-OCRv4_server_det",   // 1141 -> 3541 ms under Metal, pooling 0.36%
      "PP-OCRv6_medium_det",   // no global Pooling3D ops (rewrite is a no-op)
      "PP-OCRv5_server_det",   // no global Pooling3D ops
      "PP-OCRv4_mobile_seal_det",
      "PP-OCRv4_server_seal_det",
  };
  for (const char* n : kExcluded) {
    expect(!det_has_se_rewrite_variant(n),
           std::string("expected excluded: ") + n);
  }

  // --- rec / cls carry no SE-block global pooling ------------------------
  const char* kNonDet[] = {
      "PP-OCRv6_tiny_rec",
      "PP-OCRv5_mobile_rec",
      "PP-OCRv4_mobile_rec",
      "PP-OCRv6_medium_rec",
      "cyrillic_PP-OCRv5_mobile_rec",
      "arabic_PP-OCRv5_mobile_rec",
      "PP-LCNet_x1_0_textline_ori",
  };
  for (const char* n : kNonDet) {
    expect(!det_has_se_rewrite_variant(n),
           std::string("expected non-det excluded: ") + n);
  }

  // --- empty / near-miss names are never eligible ------------------------
  expect(!det_has_se_rewrite_variant(""), "empty name must be excluded");
  expect(!det_has_se_rewrite_variant("PP-OCRv6_tiny_det.red"),
         "suffixed name must be excluded");
  expect(!det_has_se_rewrite_variant("PP-OCRv6_tiny_det "),
         "trailing space must be excluded (exact match only)");
  expect(!det_has_se_rewrite_variant("pp-ocrv6_tiny_det"),
         "matching is case-sensitive (exact model names only)");

  // --- variant file naming ----------------------------------------------
  expect(se_rewrite_variant_name("PP-OCRv5_mobile_det") ==
             "PP-OCRv5_mobile_det.red.mnn",
         "variant name for v5 mobile det");
  expect(se_rewrite_variant_name("PP-OCRv6_tiny_det") ==
             "PP-OCRv6_tiny_det.red.mnn",
         "variant name for v6 tiny det");

  if (g_failures == 0) {
    std::printf("test_se_rewrite: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "test_se_rewrite: %d failure(s)\n", g_failures);
  return 1;
}
