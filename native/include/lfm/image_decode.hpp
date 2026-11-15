#pragma once
// Image decoding for the vision front end (validation/images-v1 rule 1): the RGB pixels HF's
// `transformers.image_utils.load_image` gives the model, which is Pillow 13.0.0's `Image.open`, then
// `ImageOps.exif_transpose`, then `convert("RGB")`. Behaviour read from Pillow's source and checked against it image by
// image (tools/hash_lfm3_image_front.py, the native check `lfm3-image-decode-check`):
//   JPEG (vendored libjpeg-turbo 3.2.0, as Thor's Pillow bundles): libjpeg's defaults (islow DCT, fancy upsampling).
//     1 component is L, 3 are RGB (libjpeg converts), 4 are CMYK read as Adobe-inverted ("CMYK;I": every byte
//     inverted), then Pillow's cmyk2rgb. Only 8-bit samples. A file that ends early is refused, as Pillow refuses it
//     ("image file is truncated"); libjpeg's other warnings are not errors, as in Pillow.
//   PNG (vendored stb_image, PNG only): L, LA, RGB, RGBA, palette (with or without tRNS: transparency dropped),
//     1/2/4-bit gray scaled to 0..255, 16-bit RGB and RGBA and LA reduced to their high byte, and 16-bit gray as Pillow
//     does it (opened as I;16, then I16_RGB: a value under 256 is kept, anything larger becomes 255).
//   Orientation: EXIF's tag 0x0112 (JPEG APP1 "Exif\0\0" segments concatenated; PNG eXIf, or a tEXt/zTXt/iTXt
//     "Raw profile type exif"), and only when EXIF has none, the first `tiff:Orientation="d"` or
//     `tiff:Orientation>d` in XMP (JPEG APP1 "http://ns.adobe.com/xap/1.0/\0"; PNG iTXt "XML:com.adobe.xmp").
//     2..8 transpose as Pillow's exif_transpose; any other value does nothing.
//   To RGB: L and 1-bit replicate, alpha is dropped (not composited), CMYK: nk = 255 - K, each channel
//     nk - (C*nk + 128 + ((C*nk + 128) >> 8)) >> 8, clipped to 0..255.
// Refused (DecodeError, with a code): anything not JPEG or PNG, a corrupt or truncated file, JPEG with other than 1, 3
// or 4 components or other than 8-bit samples, an Apple CgBI PNG, and images over Pillow's decompression-bomb limit
// (2 x 89,478,485 pixels) or over the engine's own limit, whichever is lower.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm::image {

struct Rgb {
  int width = 0, height = 0;
  std::vector<std::uint8_t> pixels;  // height rows of width x 3 bytes, R G B
};

struct DecodeError : std::runtime_error {
  std::string code;  // unsupported_format, corrupt_image, truncated_image, unsupported_jpeg, unsupported_png,
                     // image_too_large
  DecodeError(std::string c, const std::string& message) : std::runtime_error(message), code(std::move(c)) {}
};

// Pillow's Image.MAX_IMAGE_PIXELS (89,478,485); Image.open refuses more than twice that.
inline constexpr std::uint64_t pillow_bomb_pixels = 2ull * 89478485ull;

struct Limits {
  std::uint64_t max_pixels = pillow_bomb_pixels;  // the engine may set a lower limit
};

// The pixels HF's load_image gives, from the file's bytes. Throws DecodeError.
Rgb load_rgb(const std::uint8_t* data, std::size_t size, const Limits& limits = {});

// The orientation Pillow's exif_transpose would read (1 if none or unparseable), exposed for tests.
int orientation(const std::uint8_t* data, std::size_t size);

}  // namespace lfm::image
