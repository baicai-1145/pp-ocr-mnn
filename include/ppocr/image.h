// pp-ocr-mnn — minimal Image declaration (postprocess module temporary shim).
// Owner: m1 (decision-maker will merge if m1 ships a fuller version).
// Contract: 8-bit BGR, c==3, contiguous row-major data (w*h*3).
#ifndef PPOCR_IMAGE_H_
#define PPOCR_IMAGE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace ppocr {

struct Image {
  int w = 0;
  int h = 0;
  int c = 0;
  std::vector<uint8_t> data;
};

Image load_image(const std::string& path);
bool save_image(const std::string& path, const Image& img);

}  // namespace ppocr

#endif  // PPOCR_IMAGE_H_
