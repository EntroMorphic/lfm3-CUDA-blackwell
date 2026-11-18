// audio-parse-bench: validation/audio-frontend-v1, quiet window 16 (diagnostic, no rule). On one thread, the work the
// engine thread does for an audio clip at admission before it finds the clip's features (WorkerCore::parse_speech):
//   - parse: lfm::Json::parse of an engine request line carrying the clip as base64, as the in-process server sends it;
//   - decode: lfm3_speech::clip_from_base64 (base64 decode, WAV parse, samples to float) and the clip's token count;
//   - lookup: the feature cache's find by the clip's base64 text (the cache holding that clip).
// Each: one untimed run, then REPEAT timed runs; prints the medians as JSON. Compiled as the engine compiles it.
// Usage: audio-parse-bench WAV [REPEAT]
#include <lfm/audio_features.hpp>
#include <lfm/base64.hpp>
#include <lfm/model_config.hpp>
#include <lfm/speech_run.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
template <class F>
double timed(int repeat, F&& f) {
  f();  // untimed
  std::vector<double> t;
  for (int i = 0; i < repeat; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  return median(t);
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2 || argc > 3) throw std::runtime_error("usage: audio-parse-bench WAV [REPEAT]");
  const int repeat = argc == 3 ? std::stoi(argv[2]) : 10;
  std::ifstream f(argv[1], std::ios::binary);
  const std::string wav((std::istreambuf_iterator<char>(f)), {});
  if (wav.empty()) throw std::runtime_error("cannot read the WAV");
  const std::string b64 = lfm::base64::encode(wav);
  // the engine request as Service frames it for op speech_admit (the fields beside the clip are small)
  const std::string line = "{\"op\":\"speech_admit\",\"id\":1,\"mode\":\"text\",\"max_new_tokens\":32,\"seed\":0,"
                           "\"chunks\":[[1,2,3]],\"audio\":[\"" + b64 + "\"]}";
  size_t sink = 0;
  const double parse = timed(repeat, [&] {
    const auto r = lfm::Json::parse(line);
    sink += r.at("audio").array()[0].string().size();
  });
  const auto request = lfm::Json::parse(line);
  const std::string& clip = request.at("audio").array()[0].string();
  const double decode = timed(repeat, [&] {
    const auto samples = lfm3_speech::clip_from_base64(clip);
    sink += lfm::audio_features::tokens(samples.size());
  });
  auto& cache = lfm::audio_features::shared_feature_cache();
  const auto lease = cache.put(clip, std::make_shared<const std::vector<float>>(1, 0.0f));
  const double lookup = timed(repeat, [&] { sink += cache.find(clip) ? 1 : 0; });
  std::printf("{\"wav_bytes\":%zu,\"base64_bytes\":%zu,\"line_bytes\":%zu,\"repeat\":%d,\"parse_seconds\":%.6f,"
              "\"decode_seconds\":%.6f,\"lookup_seconds\":%.6f,\"total_seconds\":%.6f,\"sink\":%zu}\n",
              wav.size(), b64.size(), line.size(), repeat, parse, decode, lookup, parse + decode + lookup, sink);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "audio-parse-bench: %s\n", e.what());
  return 1;
}
