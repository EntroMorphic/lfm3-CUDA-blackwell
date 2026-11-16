#pragma once
// validation/image-queue-v1 part 2: the in-process server's handler thread decodes and preprocesses an image part, and
// leaves the pixel values here under the part's base64 text for as long as its request lives; the engine-owner thread
// (WorkerCore::parse_speech) finds them there instead of decoding the same bytes again. A miss (an out-of-process
// worker, a front end that had the image's token count cached) decodes, as before. Exact by construction: the same
// bytes, the same decoder and preprocessing, the same pixel values.
#include <lfm/image_preprocess.hpp>
#include <lfm/leased_map.hpp>

namespace lfm::image {

inline LeasedMap<PixelValues>& shared_pixel_handoff() {
  static LeasedMap<PixelValues> handoff;
  return handoff;
}

}  // namespace lfm::image
