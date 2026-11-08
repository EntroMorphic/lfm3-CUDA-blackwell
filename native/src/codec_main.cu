// liquid-codec: the native codec decoder on a file of codes (#2 step 2, A2). Codes are raw little-endian int32,
// frame-major [frames][12], each in [0, 2016); the output is raw FP32 samples at 24 kHz.
//   liquid-codec CODEC_DIR CODES.i32 OUT.f32 [CHUNK_FRAMES]
// CHUNK_FRAMES (default 0: the whole sequence in one call) streams the codes through the decoder that many frames at
// a time. LFM_CODEC_DUMP=DIR writes the first call's intermediates to DIR. Prints one JSON line to stdout.
#include <lfm/codec.cuh>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) try {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: liquid-codec CODEC_DIR CODES.i32 OUT.f32 [CHUNK_FRAMES]\n";
    return 2;
  }
  std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error(std::string("cannot read ") + argv[2]);
  const auto bytes = size_t(in.tellg());
  if (!bytes || bytes % (lfm::codec::kCodebooks * sizeof(int32_t)))
    throw std::runtime_error("codes file is not a whole number of 12-code frames");
  std::vector<int32_t> codes(bytes / sizeof(int32_t));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(codes.data()), std::streamsize(bytes));
  const int frames = int(codes.size() / lfm::codec::kCodebooks);
  char* end = nullptr;
  const long chunk_arg = argc == 5 ? std::strtol(argv[4], &end, 10) : 0;
  if (argc == 5 && (end == argv[4] || *end || chunk_arg < 0)) throw std::runtime_error("CHUNK_FRAMES must be >= 0");
  const int chunk = chunk_arg ? int(std::min<long>(chunk_arg, frames)) : frames;

  lfm::codec::Decoder decoder(argv[1], chunk);
  if (const char* d = std::getenv("LFM_CODEC_DUMP")) decoder.dump_dir = d;
  std::vector<float> audio;
  audio.reserve(size_t(frames) * lfm::codec::kSamplesPerFrame);
  lfm::codec::check(cudaDeviceSynchronize(), "codec setup");
  const auto start = std::chrono::steady_clock::now();
  for (int f = 0; f < frames; f += chunk) {
    const int n = std::min(chunk, frames - f);
    decoder.decode(codes.data() + size_t(f) * lfm::codec::kCodebooks, n, f + n >= frames, audio);
  }
  lfm::codec::check(cudaDeviceSynchronize(), "codec decode");
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  if (audio.size() != size_t(frames) * lfm::codec::kSamplesPerFrame) throw std::runtime_error("codec: wrong sample count");
  std::ofstream out(argv[3], std::ios::binary);
  out.write(reinterpret_cast<const char*>(audio.data()), std::streamsize(audio.size() * sizeof(float)));
  if (!out) throw std::runtime_error(std::string("cannot write ") + argv[3]);
  std::cout << "{\"frames\": " << frames << ", \"chunk_frames\": " << chunk << ", \"samples\": " << audio.size()
            << ", \"decode_ms\": " << ms << ", \"audio_seconds\": " << audio.size() / 24000.0
            << ", \"device_bytes\": " << decoder.device_bytes() << "}\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "liquid-codec: " << e.what() << "\n";
  return 1;
}
