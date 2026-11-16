// Host tool for the audio-input parsers and front end (tests/test_audio_input_host.py): one JSON line per input.
//   audio-input-tool wav FILE...      -> {"ok":true,"rate":R,"samples":N,"sum":S} or {"ok":false,"error":"..."}
//   audio-input-tool b64 FILE...      -> {"ok":true,"hex":"..."} or {"ok":false,"error":"..."} (FILE holds the text)
//   audio-input-tool mel WAV OUT.f32  -> writes lfm::audio_features::log_mel of the clip ([128][frames] float32) and
//                                        prints {"frames":T,"tokens":N}
//   audio-input-tool mel-threads WAV  -> log_mel at 1 to 12 threads and from 8 callers at once, each byte-identical to
//                                        the 1-thread result (validation/audio-frontend-v1); {"ok":true,"frames":T}
#include <lfm/audio_features.hpp>
#include <lfm/base64.hpp>
#include <lfm/wav.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>
#include <vector>

namespace {
std::string slurp(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error(std::string("cannot read ") + path);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
std::string escape(const std::string& s) {
  std::string o;
  for (char c : s) o += (c == '"' || c == '\\') ? std::string("\\") + c : (unsigned char)c < 32 ? std::string(" ") : std::string(1, c);
  return o;
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 3) throw std::runtime_error("usage: audio-input-tool wav|b64 FILE... | mel WAV OUT.f32 | mel-threads WAV");
  const std::string mode = argv[1];
  if (mode == "mel-threads") {
    const auto a = lfm::wav::read(argv[2]);
    const auto want = lfm::audio_features::log_mel(a.samples, 1);
    for (int t = 2; t <= 12; ++t)
      if (lfm::audio_features::log_mel(a.samples, t) != want) throw std::runtime_error("the thread count changed the features");
    std::vector<std::vector<float>> got(8);
    std::vector<std::thread> callers;
    for (size_t i = 0; i < got.size(); ++i) callers.emplace_back([&a, &got, i] { got[i] = lfm::audio_features::log_mel(a.samples); });
    for (auto& c : callers) c.join();
    for (const auto& g : got)
      if (g != want) throw std::runtime_error("concurrent calls changed the features");
    std::cout << "{\"ok\":true,\"frames\":" << lfm::audio_features::frames(a.samples.size()) << "}\n";
    return 0;
  }
  if (mode == "exact-threads") {  // amendment 2: F1 and F2 the same floats at 1 to 12 threads and from 8 callers
    using lfm::audio_features::Transform;
    const auto a = lfm::wav::read(argv[2]);
    const auto want = lfm::audio_features::log_mel(a.samples, 1, Transform::kExactF2);
    for (int t = 1; t <= 12; ++t)
      if (lfm::audio_features::log_mel(a.samples, t, Transform::kExactF1) != want ||
          lfm::audio_features::log_mel(a.samples, t, Transform::kExactF2) != want)
        throw std::runtime_error("F1, F2 or the thread count changed the features");
    std::vector<std::vector<float>> got(8);
    std::vector<std::thread> callers;
    for (size_t i = 0; i < got.size(); ++i)
      callers.emplace_back([&a, &got, i] { got[i] = lfm::audio_features::log_mel(a.samples, 4, i % 2 ? Transform::kExactF1 : Transform::kExactF2); });
    for (auto& c : callers) c.join();
    for (const auto& g : got)
      if (g != want) throw std::runtime_error("concurrent calls changed the features");
    std::cout << "{\"ok\":true,\"frames\":" << lfm::audio_features::frames(a.samples.size()) << "}\n";
    return 0;
  }
  if (mode == "mel") {
    const auto a = lfm::wav::read(argv[2]);
    const auto mel = lfm::audio_features::log_mel(a.samples);
    std::ofstream f(argv[3], std::ios::binary);
    f.write(reinterpret_cast<const char*>(mel.data()), std::streamsize(mel.size() * sizeof(float)));
    std::cout << "{\"frames\":" << lfm::audio_features::frames(a.samples.size())
              << ",\"tokens\":" << lfm::audio_features::tokens(a.samples.size()) << "}\n";
    return f ? 0 : 1;
  }
  for (int i = 2; i < argc; ++i) {
    try {
      if (mode == "wav") {
        const auto a = lfm::wav::parse(slurp(argv[i]));
        double sum = 0;
        for (float v : a.samples) sum += v;
        std::printf("{\"ok\":true,\"rate\":%d,\"samples\":%zu,\"sum\":%.9g}\n", a.rate, a.samples.size(), sum);
      } else if (mode == "b64") {
        const auto bytes = lfm::base64::decode(slurp(argv[i]));
        std::string hex;
        for (unsigned char c : bytes) { char h[3]; std::snprintf(h, 3, "%02x", c); hex += h; }
        std::cout << "{\"ok\":true,\"hex\":\"" << hex << "\"}\n";
      } else {
        throw std::runtime_error("unknown mode");
      }
    } catch (const std::exception& e) {
      std::cout << "{\"ok\":false,\"error\":\"" << escape(e.what()) << "\"}\n";
    }
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << "audio-input-tool: " << e.what() << "\n";
  return 1;
}
