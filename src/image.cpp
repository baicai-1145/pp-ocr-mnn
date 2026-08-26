// pp-ocr-mnn — image I/O implementation
//
// JPEG decodes via the system libjpeg (-ljpeg, libjpeg-turbo on every
// mainstream distro). This is REQUIRED for bit-exact parity with the
// baseline pipeline: cv2.imread (libjpeg-turbo) and stb_image produce
// pixel differences of up to ±3 on ~13% of pixels for some jpegs,
// which is enough to flip det boxes at the DB threshold boundary
// (see tools/M2_DECODE_ALIGN.md).
// All other formats fall back to stb_image. Grayscale and RGBA inputs
// are normalized to BGR; alpha is composited over a black background
// when present (PaddleOCR's loaders do the same).
// Encoding is via stb_image_write.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ppocr/image.h"

#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

#if !defined(PPDECODE_HAVE_LIBJPEG)
#define PPDECODE_HAVE_LIBJPEG 1
#endif

#if PPDECODE_HAVE_LIBJPEG
#include <jpeglib.h>
#include <csetjmp>
namespace {
struct pp_jpeg_err {
  jpeg_error_mgr pub;
  jmp_buf jb;
};
void pp_jpeg_error_exit(j_common_ptr ci) {
  pp_jpeg_err* e = reinterpret_cast<pp_jpeg_err*>(ci->err);
  longjmp(e->jb, 1);
}
// Decode a baseline/progressive JPEG to interleaved BGR8 via libjpeg.
// Returns false on any decode error; output layout matches cv2.imread.
bool decode_jpeg_bgr(const std::string& path, ppocr::Image& img) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> buf(static_cast<size_t>(sz));
  if (fread(buf.data(), 1, buf.size(), f) != buf.size()) { fclose(f); return false; }
  fclose(f);

  jpeg_decompress_struct cinfo{};
  pp_jpeg_err jerr{};
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = pp_jpeg_error_exit;
  if (setjmp(jerr.jb)) {
    jpeg_destroy_decompress(&cinfo);
    return false;
  }
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, buf.data(), static_cast<unsigned long>(buf.size()));
  if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
    jpeg_destroy_decompress(&cinfo);
    return false;
  }
  // cv2.imread reads 3-channel BGR for both color and gray jpegs (gray is
  // broadcast), so request JCS_EXT_BGR unconditionally and let libjpeg's
  // color converter handle grayscale / YCbCr / CMYK sources.
  cinfo.out_color_space = JCS_EXT_BGR;
  jpeg_start_decompress(&cinfo);
  const int w = static_cast<int>(cinfo.output_width);
  const int h = static_cast<int>(cinfo.output_height);
  if (w <= 0 || h <= 0 || cinfo.output_components != 3) {
    jpeg_destroy_decompress(&cinfo);
    return false;
  }
  img.w = w; img.h = h; img.c = 3;
  img.data.assign(static_cast<size_t>(w) * h * 3, 0);
  const size_t rb = static_cast<size_t>(w) * 3;
  while (cinfo.output_scanline < cinfo.output_height) {
    unsigned char* rp[1] = { img.data.data() + cinfo.output_scanline * rb };
    if (jpeg_read_scanlines(&cinfo, rp, 1) != 1) {
      jpeg_destroy_decompress(&cinfo);
      return false;
    }
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return true;
}
} // namespace
#endif // PPDECODE_HAVE_LIBJPEG

namespace ppocr {

Image load_image(const std::string& path) {
  Image img;
#if PPDECODE_HAVE_LIBJPEG
  {
    // Cheap extension sniff: JPEG goes through libjpeg; everything else
    // (png/webp/bmp/tga/...) stays on stb. Unknown extensions also go to
    // stb, which sniffs magic bytes rather than trusting the name.
    auto dot = path.find_last_of('.');
    std::string ext = dot == std::string::npos ? std::string{} : path.substr(dot + 1);
    for (auto& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (ext == "jpg" || ext == "jpeg") {
      if (decode_jpeg_bgr(path, img)) return img;
      // Fall through: retry with stb so corrupt-jpeg handling matches the
      // old behaviour (empty Image -> PPOCR_ERR_IO at the ABI layer).
      img = Image{};
    }
  }
#endif
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
