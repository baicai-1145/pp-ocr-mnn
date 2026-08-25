// pp-ocr-mnn — CTC decode (owner post)
//
// Decodes a (T, num_classes) tensor into a UTF-8 string. Alphabet is
// ["blank"] + dict + (use_space ? [" "] : []); blank index 0. Repeats
// are collapsed and blanks dropped. Score = mean probability of emitted
// characters (official PaddleOCR CTCLabelDecode / BaseRecLabelDecode
// semantics, see ppocr/postprocess/rec_postprocess.py).
//
// Real implementation lives on branch ws/post. This header freezes the
// ABI so ws/m1 can compile against the same signature.
#ifndef PPOCR_POSTPROCESS_CTC_DECODE_H_
#define PPOCR_POSTPROCESS_CTC_DECODE_H_

#include <string>
#include "ppocr/config.h"

namespace ppocr {

struct RecOut {
  std::string text; // UTF-8
  float score = 0;
};

RecOut ctc_decode(const float* logits, int timesteps, int num_classes,
                  const RecConfig& cfg);

} // namespace ppocr
#endif // PPOCR_POSTPROCESS_CTC_DECODE_H_
