// pp-ocr-mnn — CTC greedy decoder (postprocess module)
// Owner: post.
#ifndef PPOCR_POSTPROCESS_CTC_DECODE_H_
#define PPOCR_POSTPROCESS_CTC_DECODE_H_

#include <string>

#include "ppocr/config.h"

namespace ppocr {

struct RecOut {
  std::string text;  // UTF-8
  float score = 0;
};

// Greedy CTC decode. Char table is ["blank"] + dict + (use_space ? [" "] : []).
// blank index = 0. Collapses repeats and drops blanks. score = mean of
// per-timestep probabilities of the EMITTED (non-collapsed, non-blank) chars.
// If no char is emitted, score = 0.
RecOut ctc_decode(const float* logits, int timesteps, int num_classes,
                  const RecConfig& cfg);

}  // namespace ppocr

#endif  // PPOCR_POSTPROCESS_CTC_DECODE_H_
