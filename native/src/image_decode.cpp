// Image decoding for the vision front end: see lfm/image_decode.hpp for what it reproduces and why.
#include <lfm/image_decode.hpp>

#include <algorithm>
#include <array>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string_view>

extern "C" {
#include <jpeglib.h>
}

#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "../third_party/stb/stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace lfm::image {
namespace {

using Bytes = std::vector<std::uint8_t>;

[[noreturn]] void fail(const char* code, const std::string& message) { throw DecodeError(code, message); }

bool starts_with(const std::uint8_t* p, std::size_t n, std::string_view prefix) {
  return n >= prefix.size() && std::memcmp(p, prefix.data(), prefix.size()) == 0;
}

void check_pixels(std::uint64_t w, std::uint64_t h, const Limits& limits) {
  if (w == 0 || h == 0) fail("corrupt_image", "image has a zero dimension");
  const std::uint64_t limit = std::min(limits.max_pixels, pillow_bomb_pixels);
  if (w * h > limit) {
    fail("image_too_large", "image has " + std::to_string(w * h) + " pixels, more than the limit of " +
                                std::to_string(limit));
  }
}

// ---------------------------------------------------------------- orientation (Pillow's getexif + exif_transpose)

// The first value of tag 0x0112 in IFD0 of a TIFF block, or nullopt (Pillow's Exif.load on "Exif\0\0" + TIFF).
std::optional<int> tiff_orientation(const std::uint8_t* p, std::size_t n) {
  if (starts_with(p, n, std::string_view("Exif\0\0", 6))) {
    p += 6;
    n -= 6;
  }
  if (n < 8) return std::nullopt;
  bool big;
  if (p[0] == 'I' && p[1] == 'I' && p[2] == 42 && p[3] == 0) big = false;
  else if (p[0] == 'M' && p[1] == 'M' && p[2] == 0 && p[3] == 42) big = true;
  else return std::nullopt;
  auto u16 = [&](std::size_t o) -> unsigned {
    return big ? (unsigned(p[o]) << 8 | p[o + 1]) : (unsigned(p[o + 1]) << 8 | p[o]);
  };
  auto u32 = [&](std::size_t o) -> std::uint32_t {
    return big ? (std::uint32_t(p[o]) << 24 | std::uint32_t(p[o + 1]) << 16 | std::uint32_t(p[o + 2]) << 8 | p[o + 3])
               : (std::uint32_t(p[o + 3]) << 24 | std::uint32_t(p[o + 2]) << 16 | std::uint32_t(p[o + 1]) << 8 | p[o]);
  };
  const std::uint32_t ifd = u32(4);
  if (ifd < 8 || ifd > n || n - ifd < 2) return std::nullopt;
  const unsigned count = u16(ifd);
  for (unsigned i = 0; i < count; ++i) {
    const std::size_t e = ifd + 2 + std::size_t(i) * 12;
    if (e + 12 > n) return std::nullopt;
    if (u16(e) != 0x0112) continue;
    const unsigned type = u16(e + 2);
    const std::uint32_t values = u32(e + 4);
    if (values < 1) return std::nullopt;
    switch (type) {
      case 1: case 7: return int(p[e + 8]);      // BYTE, UNDEFINED
      case 3: return int(u16(e + 8));           // SHORT
      case 4: return int(u32(e + 8));           // LONG
      default: return std::nullopt;
    }
  }
  return std::nullopt;
}

// Pillow's XMP fallback: the first match of tiff:Orientation(="|>)([0-9]).
std::optional<int> xmp_orientation(const std::uint8_t* p, std::size_t n) {
  static constexpr std::string_view key = "tiff:Orientation";
  for (std::size_t i = 0; i + key.size() + 2 < n + 1; ++i) {
    if (std::memcmp(p + i, key.data(), key.size()) != 0) continue;
    std::size_t j = i + key.size();
    if (j + 1 < n && p[j] == '=' && p[j + 1] == '"') j += 2;
    else if (j < n && p[j] == '>') j += 1;
    else continue;
    if (j < n && p[j] >= '0' && p[j] <= '9') return p[j] - '0';
  }
  return std::nullopt;
}

struct Metadata {
  Bytes exif;          // Pillow's info["exif"] (JPEG: the APP1 segments concatenated; PNG: "Exif\0\0" + eXIf)
  bool has_exif = false;
  Bytes raw_profile;   // PNG "Raw profile type exif", decoded from hex
  bool has_raw_profile = false;
  Bytes xmp;           // JPEG APP1 XMP or PNG iTXt XML:com.adobe.xmp
  bool has_xmp = false;
};

int orientation_of(const Metadata& m) {
  std::optional<int> o;
  if (m.has_exif) o = tiff_orientation(m.exif.data(), m.exif.size());
  else if (m.has_raw_profile) o = tiff_orientation(m.raw_profile.data(), m.raw_profile.size());
  if (!o && m.has_xmp) o = xmp_orientation(m.xmp.data(), m.xmp.size());
  return o.value_or(1);
}

// Pillow's Image.transpose for exif_transpose's methods, on an RGB image.
Rgb transpose(const Rgb& in, int orientation) {
  if (orientation < 2 || orientation > 8) return in;
  const int w = in.width, h = in.height;
  const bool swap = orientation >= 5;
  Rgb out;
  out.width = swap ? h : w;
  out.height = swap ? w : h;
  out.pixels.resize(in.pixels.size());
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      int sx = 0, sy = 0;
      switch (orientation) {
        case 2: sx = w - 1 - x; sy = y; break;              // FLIP_LEFT_RIGHT
        case 3: sx = w - 1 - x; sy = h - 1 - y; break;      // ROTATE_180
        case 4: sx = x; sy = h - 1 - y; break;              // FLIP_TOP_BOTTOM
        case 5: sx = y; sy = x; break;                      // TRANSPOSE
        case 6: sx = y; sy = h - 1 - x; break;              // ROTATE_270 (clockwise 90)
        case 7: sx = w - 1 - y; sy = h - 1 - x; break;      // TRANSVERSE
        case 8: sx = w - 1 - y; sy = x; break;              // ROTATE_90 (counter-clockwise)
      }
      std::memcpy(&out.pixels[(std::size_t(y) * out.width + x) * 3], &in.pixels[(std::size_t(sy) * w + sx) * 3], 3);
    }
  }
  return out;
}

// ---------------------------------------------------------------- JPEG (as Pillow's JpegDecode.c drives libjpeg)

struct JpegError {
  jpeg_error_mgr pub;
  std::jmp_buf jump;
};

void jpeg_error_exit(j_common_ptr cinfo) {
  std::longjmp(reinterpret_cast<JpegError*>(cinfo->err)->jump, 1);
}
void jpeg_silent(j_common_ptr) {}

// A source that never invents data: when the bytes run out, it suspends (Pillow's buffer source, given the whole
// file at once), so a truncated image is seen as one rather than padded with a fake EOI.
struct SuspendingSource {
  jpeg_source_mgr pub;
  std::size_t skip;
};
void src_init(j_decompress_ptr) {}
boolean src_fill(j_decompress_ptr) { return FALSE; }
void src_skip(j_decompress_ptr cinfo, long n) {
  auto* s = reinterpret_cast<SuspendingSource*>(cinfo->src);
  if (n <= 0) return;
  if (std::size_t(n) > s->pub.bytes_in_buffer) {
    s->skip = std::size_t(n) - s->pub.bytes_in_buffer;
    s->pub.next_input_byte += s->pub.bytes_in_buffer;
    s->pub.bytes_in_buffer = 0;
  } else {
    s->pub.next_input_byte += n;
    s->pub.bytes_in_buffer -= std::size_t(n);
  }
}
void src_term(j_decompress_ptr) {}

// Pillow's JPEG plugin reads the markers itself before decoding: APP1 Exif and XMP, and SOF's precision and
// component count (it refuses other than 8 bits and 1, 3 or 4 components).
void jpeg_markers(const std::uint8_t* p, std::size_t n, Metadata& meta) {
  std::size_t i = 2;
  while (i + 4 <= n) {
    if (p[i] != 0xFF) { ++i; continue; }
    const std::uint8_t m = p[i + 1];
    if (m == 0xFF) { ++i; continue; }
    if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7) || m == 0x01) { i += 2; continue; }
    if (m == 0xDA || m == 0xD9) break;  // start of scan: Pillow's header parse stops here
    const std::size_t len = std::size_t(p[i + 2]) << 8 | p[i + 3];
    if (len < 2 || i + 2 + len > n) break;
    const std::uint8_t* s = p + i + 4;
    const std::size_t sn = len - 2;
    if (m == 0xE1) {
      if (starts_with(s, sn, std::string_view("Exif\0\0", 6))) {
        if (!meta.has_exif) {
          meta.exif.assign(s, s + sn);
          meta.has_exif = true;
        } else {
          meta.exif.insert(meta.exif.end(), s + 6, s + sn);
        }
      } else if (starts_with(s, sn, std::string_view("http://ns.adobe.com/xap/1.0/\0", 29))) {
        meta.xmp.assign(s + 29, s + sn);
        meta.has_xmp = true;
      }
    }
    i += 2 + len;
  }
}

Rgb decode_jpeg(const std::uint8_t* data, std::size_t size, const Limits& limits, Metadata& meta) {
  jpeg_markers(data, size, meta);
  jpeg_decompress_struct cinfo{};
  JpegError err{};
  cinfo.err = jpeg_std_error(&err.pub);
  err.pub.error_exit = jpeg_error_exit;
  err.pub.output_message = jpeg_silent;
  SuspendingSource src{};
  Rgb out;
  std::vector<std::uint8_t> row;
  const char* refusal = nullptr;  // set before a longjmp-free return path
  std::string refusal_message;
  if (setjmp(err.jump)) {
    char message[JMSG_LENGTH_MAX];
    err.pub.format_message(reinterpret_cast<j_common_ptr>(&cinfo), message);
    jpeg_destroy_decompress(&cinfo);
    fail("corrupt_image", std::string("JPEG: ") + message);
  }
  jpeg_create_decompress(&cinfo);
  src.pub.init_source = src_init;
  src.pub.fill_input_buffer = src_fill;
  src.pub.skip_input_data = src_skip;
  src.pub.resync_to_restart = jpeg_resync_to_restart;
  src.pub.term_source = src_term;
  src.pub.next_input_byte = data;
  src.pub.bytes_in_buffer = size;
  cinfo.src = &src.pub;

  int ok;
  do {
    ok = jpeg_read_header(&cinfo, FALSE);
  } while (ok == JPEG_HEADER_TABLES_ONLY);
  if (ok == JPEG_SUSPENDED) {
    refusal = "truncated_image";
    refusal_message = "JPEG: the file ends before its image header";
  } else if (cinfo.data_precision != 8) {
    refusal = "unsupported_jpeg";
    refusal_message = "JPEG: " + std::to_string(cinfo.data_precision) + "-bit samples (only 8-bit are supported)";
  } else if (cinfo.num_components != 1 && cinfo.num_components != 3 && cinfo.num_components != 4) {
    refusal = "unsupported_jpeg";
    refusal_message = "JPEG: " + std::to_string(cinfo.num_components) + " components (1, 3 or 4 are supported)";
  }
  if (!refusal) {
    try {
      check_pixels(cinfo.image_width, cinfo.image_height, limits);
    } catch (const DecodeError& e) {
      jpeg_destroy_decompress(&cinfo);
      throw;
    }
    const int comps = cinfo.num_components;
    cinfo.out_color_space = comps == 1 ? JCS_GRAYSCALE : comps == 3 ? JCS_RGB : JCS_CMYK;
    if (!jpeg_start_decompress(&cinfo)) {
      refusal = "truncated_image";
      refusal_message = "JPEG: the file ends before its image data";
    } else {
      const int w = int(cinfo.output_width), h = int(cinfo.output_height), oc = cinfo.output_components;
      out.width = w;
      out.height = h;
      out.pixels.resize(std::size_t(w) * h * 3);
      row.resize(std::size_t(w) * oc);
      for (int y = 0; y < h; ++y) {
        JSAMPROW rp = row.data();
        if (jpeg_read_scanlines(&cinfo, &rp, 1) != 1) {
          refusal = "truncated_image";
          refusal_message = "JPEG: the file ends after " + std::to_string(y) + " of " + std::to_string(h) + " rows";
          break;
        }
        std::uint8_t* o = &out.pixels[std::size_t(y) * w * 3];
        if (oc == 1) {
          for (int x = 0; x < w; ++x) o[3 * x] = o[3 * x + 1] = o[3 * x + 2] = row[x];
        } else if (oc == 3) {
          std::memcpy(o, row.data(), std::size_t(w) * 3);
        } else {  // "CMYK;I" (every byte inverted), then Pillow's cmyk2rgb
          for (int x = 0; x < w; ++x) {
            const int k = 255 - row[4 * x + 3];  // the inverted byte Pillow holds
            const int nk = 255 - k;
            for (int c = 0; c < 3; ++c) {
              const int v = 255 - row[4 * x + c];
              const int tmp = v * nk + 128;
              const int r = nk - (((tmp >> 8) + tmp) >> 8);
              o[3 * x + c] = std::uint8_t(r <= 0 ? 0 : r < 256 ? r : 255);
            }
          }
        }
      }
      // Pillow accepts a file whose rows are all there even if jpeg_finish_decompress suspends (no EOI).
      if (!refusal) jpeg_finish_decompress(&cinfo);
    }
  }
  jpeg_destroy_decompress(&cinfo);
  if (refusal) fail(refusal, refusal_message);
  return out;
}

// ---------------------------------------------------------------- PNG

std::uint32_t be32(const std::uint8_t* p) {
  return std::uint32_t(p[0]) << 24 | std::uint32_t(p[1]) << 16 | std::uint32_t(p[2]) << 8 | p[3];
}

int hex_value(std::uint8_t c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// Pillow: "".join(text.split("\n")[3:]) then bytes.fromhex (which skips whitespace between pairs).
std::optional<Bytes> raw_profile_bytes(const std::string& text) {
  std::size_t pos = 0;
  for (int k = 0; k < 3; ++k) {
    pos = text.find('\n', pos);
    if (pos == std::string::npos) return Bytes{};
    ++pos;
  }
  Bytes out;
  int hi = -1;
  for (std::size_t i = pos; i < text.size(); ++i) {
    const auto c = std::uint8_t(text[i]);
    if (c == '\n' || c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f') {
      if (hi >= 0) return std::nullopt;
      continue;
    }
    const int v = hex_value(c);
    if (v < 0) return std::nullopt;
    if (hi < 0) hi = v;
    else { out.push_back(std::uint8_t(hi << 4 | v)); hi = -1; }
  }
  if (hi >= 0) return std::nullopt;
  return out;
}

struct PngHeader {
  std::uint32_t width = 0, height = 0;
  int depth = 0, color = 0;
};

PngHeader png_chunks(const std::uint8_t* p, std::size_t n, Metadata& meta) {
  PngHeader h;
  std::size_t i = 8;
  bool first = true;
  while (i + 12 <= n) {
    const std::uint32_t len = be32(p + i);
    const std::string type(reinterpret_cast<const char*>(p + i + 4), 4);
    if (len > n - i - 12) break;
    const std::uint8_t* d = p + i + 8;
    if (first && type == "CgBI") fail("unsupported_png", "PNG: Apple CgBI (\"iPhone\") PNGs are not supported");
    if (type == "IHDR" && len >= 13) {
      h.width = be32(d);
      h.height = be32(d + 4);
      h.depth = d[8];
      h.color = d[9];
    } else if (type == "eXIf") {
      meta.exif.assign({'E', 'x', 'i', 'f', 0, 0});
      meta.exif.insert(meta.exif.end(), d, d + len);
      meta.has_exif = true;
    } else if (type == "iTXt") {
      // keyword \0 compression_flag compression_method language \0 translated \0 text
      const std::uint8_t* end = d + len;
      const std::uint8_t* k = std::find(d, end, 0);
      const std::string keyword(reinterpret_cast<const char*>(d), std::size_t(k - d));
      if (k + 3 <= end && k[1] == 0 && (keyword == "XML:com.adobe.xmp" || keyword == "Raw profile type exif")) {
        const std::uint8_t* lang = k + 3;
        const std::uint8_t* lend = std::find(lang, end, 0);
        const std::uint8_t* tr = lend < end ? lend + 1 : end;
        const std::uint8_t* trend = std::find(tr, end, 0);
        const std::uint8_t* text = trend < end ? trend + 1 : end;
        if (keyword == "XML:com.adobe.xmp") {
          meta.xmp.assign(text, end);
          meta.has_xmp = true;
        } else if (auto b = raw_profile_bytes(std::string(text, end))) {
          meta.raw_profile = *b;
          meta.has_raw_profile = true;
        }
      }
    } else if (type == "tEXt") {
      const std::uint8_t* end = d + len;
      const std::uint8_t* k = std::find(d, end, 0);
      if (k < end && std::string(reinterpret_cast<const char*>(d), std::size_t(k - d)) == "Raw profile type exif") {
        if (auto b = raw_profile_bytes(std::string(k + 1, end))) {
          meta.raw_profile = *b;
          meta.has_raw_profile = true;
        }
      }
    }
    first = false;
    if (type == "IEND") break;
    i += 12 + len;
  }
  return h;
}

Rgb decode_png(const std::uint8_t* data, std::size_t size, const Limits& limits, Metadata& meta) {
  if (size > std::size_t(INT32_MAX)) fail("image_too_large", "PNG: file over 2 GiB");
  const PngHeader h = png_chunks(data, size, meta);
  check_pixels(h.width, h.height, limits);
  int w = 0, ht = 0, n = 0;
  Rgb out;
  if (h.depth == 16 && h.color == 0) {  // Pillow: I;16, then I16_RGB
    stbi_us* g = stbi_load_16_from_memory(data, int(size), &w, &ht, &n, 1);
    if (!g) fail("corrupt_image", std::string("PNG: ") + stbi_failure_reason());
    out.width = w;
    out.height = ht;
    out.pixels.resize(std::size_t(w) * ht * 3);
    for (std::size_t i = 0; i < std::size_t(w) * ht; ++i) {
      const std::uint8_t v = (g[i] >> 8) == 0 ? std::uint8_t(g[i] & 0xFF) : 255;
      out.pixels[3 * i] = out.pixels[3 * i + 1] = out.pixels[3 * i + 2] = v;
    }
    stbi_image_free(g);
    return out;
  }
  // Always RGBA: stb expands gray to (g, g, g), as Pillow's L to RGB does, and the alpha is dropped. (The pinned
  // stb refuses req_comp 0, a regression upstream fixed after it; see third_party/README.md.)
  stbi_uc* px = stbi_load_from_memory(data, int(size), &w, &ht, &n, 4);
  if (!px) fail("corrupt_image", std::string("PNG: ") + stbi_failure_reason());
  out.width = w;
  out.height = ht;
  out.pixels.resize(std::size_t(w) * ht * 3);
  for (std::size_t i = 0; i < std::size_t(w) * ht; ++i) {
    out.pixels[3 * i] = px[4 * i];
    out.pixels[3 * i + 1] = px[4 * i + 1];
    out.pixels[3 * i + 2] = px[4 * i + 2];
  }
  stbi_image_free(px);
  return out;
}

Metadata sniff_metadata(const std::uint8_t* data, std::size_t size) {
  Metadata meta;
  if (starts_with(data, size, "\xFF\xD8\xFF")) jpeg_markers(data, size, meta);
  else if (starts_with(data, size, "\x89PNG\r\n\x1A\n")) png_chunks(data, size, meta);
  return meta;
}

}  // namespace

Rgb load_rgb(const std::uint8_t* data, std::size_t size, const Limits& limits) {
  Metadata meta;
  Rgb img;
  if (starts_with(data, size, "\xFF\xD8\xFF")) img = decode_jpeg(data, size, limits, meta);
  else if (starts_with(data, size, "\x89PNG\r\n\x1A\n")) img = decode_png(data, size, limits, meta);
  else fail("unsupported_format", "not a JPEG or PNG image");
  return transpose(img, orientation_of(meta));
}

int orientation(const std::uint8_t* data, std::size_t size) { return orientation_of(sniff_metadata(data, size)); }

}  // namespace lfm::image
