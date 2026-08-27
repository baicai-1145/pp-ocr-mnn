// pp-ocr-mnn — image I/O (owner m1)
//
// Plain-old data container for an 8-bit BGR image (channel order B,G,R).
// All preprocessing in ppocr reads from this layout; stb_image handles decoding
// and stb_image_write handles encoding. No platform ifdefs.
#ifndef PPOCR_IMAGE_H_
#define PPOCR_IMAGE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace ppocr {

struct Image {
  int w = 0;        // width  in pixels
  int h = 0;        // height in pixels
  int c = 0;        // channels (always 3 after load; this codebase is BGR-only)
  std::vector<uint8_t> data; // w * h * 3 bytes, row-major, BGR order

  // M3-PERF6: optional non-owning view over external storage. When set,
  // `data` stays empty and pipeline readers use bytes() instead of
  // data.data(). Writers (rot90 etc.) only ever build fresh owning
  // Images, so views never alias mutable state.
  const uint8_t* ext = nullptr;
  const uint8_t* bytes() const { return ext ? ext : data.data(); }
  bool empty() const { return !ext && data.empty(); }
};

// Decode an image file (jpg/png/bmp/gif first frame) into BGR8.
// - Grayscale (c==1) and RGBA (c==4) inputs are converted to BGR (c=3).
// - Returns an Image with empty data on any failure (stb can't open / decode).
Image load_image(const std::string& path);

// Encode an Image to disk. Extension chooses codec:
//   .png -> PNG (lossless), .jpg/.jpeg -> JPEG quality 90, .bmp -> BMP.
// Returns true on success. data must be w*h*3 bytes; c==3 expected.
bool save_image(const std::string& path, const Image& img);

} // namespace ppocr
#endif // PPOCR_IMAGE_H_
