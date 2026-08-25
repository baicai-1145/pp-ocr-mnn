// pp-ocr-mnn — image I/O implementation
//
// Decodes via stb_image and re-emits in BGR interleaved layout (8-bit per
// channel). Grayscale and RGBA inputs are normalized to BGR; alpha is
// composited over a black background when present (PaddleOCR's loaders
// do the same). Encoding is via stb_image_write.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ppocr/image.h"

#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

namespace ppocr {

Image load_image(const std::string& path) {
  Image img;
  int w = 0, h = 0, c = 0;
  // Force 8-bit output regardless of source depth.
  uint8_t* px = stbi_load(path.c_str(), &w, &h, &c, 0);
  if (!px) {
    // stb sets an error string we can surface to the caller; the public
    // ABI converts empty data into a PPOCR_ERR_IO status.
    return img;
  }
  img.w = w;
  img.h = h;
  img.c = 3;
  img.data.assign(static_cast<size_t>(w) * h * 3, 0);

  if (c == 3) {
    // stb returns RGB; swap to BGR in place.
    for (int i = 0; i < w * h; ++i) {
      img.data[3 * i + 0] = px[3 * i + 2];
      img.data[3 * i + 1] = px[3 * i + 1];
      img.data[3 * i + 2] = px[3 * i + 0];
    }
  } else if (c == 1) {
    // Broadcast gray into BGR.
    for (int i = 0; i < w * h; ++i) {
      uint8_t g = px[i];
      img.data[3 * i + 0] = g;
      img.data[3 * i + 1] = g;
      img.data[3 * i + 2] = g;
    }
  } else if (c == 4) {
    // RGBA -> BGR with alpha composited over black. PaddleOCR's pymupdf
    // loader flattens the same way, so we keep parity.
    for (int i = 0; i < w * h; ++i) {
      uint8_t r = px[4 * i + 0];
      uint8_t g = px[4 * i + 1];
      uint8_t b = px[4 * i + 2];
      uint8_t a = px[4 * i + 3];
      // straight alpha; matches premultiplied=0 in cv::cvtColor BGR.
      img.data[3 * i + 0] = static_cast<uint8_t>((b * a) / 255);
      img.data[3 * i + 1] = static_cast<uint8_t>((g * a) / 255);
      img.data[3 * i + 2] = static_cast<uint8_t>((r * a) / 255);
    }
  } else {
    // Unexpected channel count — discard and report empty.
    img.data.clear();
    img.w = img.h = img.c = 0;
  }

  stbi_image_free(px);
  return img;
}

bool save_image(const std::string& path, const Image& img) {
  if (img.c != 3 || img.data.empty() || img.w <= 0 || img.h <= 0) {
    return false;
  }
  // stb expects RGB order for its writers; convert from our BGR layout.
  std::vector<uint8_t> rgb(img.data.size());
  const size_t n = static_cast<size_t>(img.w) * img.h;
  for (size_t i = 0; i < n; ++i) {
    rgb[3 * i + 0] = img.data[3 * i + 2];
    rgb[3 * i + 1] = img.data[3 * i + 1];
    rgb[3 * i + 2] = img.data[3 * i + 0];
  }

  // Choose codec from the file extension. Anything we don't recognize
  // falls through to PNG (lossless, no extra params).
  const std::string ext = [&] {
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return std::string{};
    std::string e = path.substr(dot + 1);
    for (auto& ch : e) ch = static_cast<char>(std::tolower(ch));
    return e;
  }();

  if (ext == "jpg" || ext == "jpeg") {
    return stbi_write_jpg(path.c_str(), img.w, img.h, 3, rgb.data(), 90) != 0;
  }
  if (ext == "bmp") {
    return stbi_write_bmp(path.c_str(), img.w, img.h, 3, rgb.data()) != 0;
  }
  if (ext == "tga") {
    return stbi_write_tga(path.c_str(), img.w, img.h, 3, rgb.data()) != 0;
  }
  return stbi_write_png(path.c_str(), img.w, img.h, 3, rgb.data(),
                        img.w * 3) != 0;
}

} // namespace ppocr
