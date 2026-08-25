// pp-ocr-mnn — CTC greedy decoder implementation.
// Owner: post. No OpenCV; pure C++17. No platform ifdefs.
#include "ppocr/postprocess/ctc_decode.h"

#include <cstdint>
#include <stdexcept>

namespace ppocr {

RecOut ctc_decode(const float* logits, int timesteps, int num_classes,
                  const RecConfig& cfg) {
  RecOut out;
  if (!logits || timesteps <= 0 || num_classes <= 1) return out;

  // Char table: ["blank"] + dict + (use_space ? [" "] : []).
  // Index 0 is "blank" (literal sentinel; not in the dict).
  int blank = 0;

  // Sanity: if num_classes doesn't match expected length, fall back to whatever
  // shape the model emitted; we still use blank=0.
  int expected = 1 + static_cast<int>(cfg.dict.size()) + (cfg.use_space ? 1 : 0);
  if (expected != num_classes) {
    // Mismatch — log nothing (we are library code), just decode with the model's
    // classes. blank still 0.
  }

  int prev = -1;
  double prob_sum = 0.0;
  int prob_count = 0;
  std::string text;
  text.reserve(timesteps);

  for (int t = 0; t < timesteps; ++t) {
    const float* row = logits + static_cast<ptrdiff_t>(t) * num_classes;
    int best = 0;
    float best_p = row[0];
    for (int c = 1; c < num_classes; ++c) {
      if (row[c] > best_p) {
        best_p = row[c];
        best = c;
      }
    }
    if (best == blank) {
      prev = blank;
      continue;
    }
    if (best != prev) {
      // Map index -> char.
      // idx 0 = blank; idx 1..dict.size() = dict[0..]; last (if use_space) = " ".
      const std::string* s = nullptr;
      if (best == 0) {
        s = nullptr;  // blank — already filtered
      } else if (best >= 1 && best - 1 < static_cast<int>(cfg.dict.size())) {
        s = &cfg.dict[best - 1];
      } else if (cfg.use_space &&
                 best == 1 + static_cast<int>(cfg.dict.size())) {
        static const std::string sp = " ";
        s = &sp;
      } else {
        // Out-of-range: skip. This guards against registry/dict desync.
        prev = best;
        continue;
      }
      if (s) {
        text.append(*s);
        prob_sum += best_p;
        ++prob_count;
      }
    }
    // If best == prev, it's a repeated non-blank char — collapse (do not emit,
    // do not include in score). This matches the official CTC greedy.
    prev = best;
  }

  out.text = std::move(text);
  if (prob_count > 0) {
    out.score = static_cast<float>(prob_sum / prob_count);
  } else {
    out.score = 0.0f;
  }
  return out;
}

}  // namespace ppocr
