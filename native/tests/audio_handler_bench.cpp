// audio-handler-bench: validation/ttft-rest-v1 stage 0 (a measurement, no rule). On one thread, each step the in-process
// server's handler takes for a /speech request carrying one audio clip, before the engine (Service: strict_json,
// parse_messages, encode_speech's decode for the token count, compute_audio_features' copy, decode and log_mel, the
// feature cache's put), timed one at a time with the server's own functions on the request body
// tools/measure_lfm3_audio_ttft.py sends. Each: one untimed run, then REPEAT timed runs; the medians as JSON.
// Usage: audio-handler-bench WAV [REPEAT]
#include <lfm/audio_features.hpp>
#include <lfm/base64.hpp>
#include <lfm/lfm3_http.hpp>
#include <lfm/lfm3_text.hpp>
#include <lfm/speech_run.hpp>
#include <lfm/wav.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <stdexcept>
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
  if (argc < 2 || argc > 3) throw std::runtime_error("usage: audio-handler-bench WAV [REPEAT]");
  const int repeat = argc == 3 ? std::stoi(argv[2]) : 10;
  std::ifstream f(argv[1], std::ios::binary);
  const std::string wav((std::istreambuf_iterator<char>(f)), {});
  if (wav.empty()) throw std::runtime_error("cannot read the WAV");
  const std::string b64 = lfm::base64::encode(wav);
  // the body measure_lfm3_audio_ttft.py sends (json.dumps: the same bytes up to key order and spacing)
  const std::string body = "{\"messages\": [{\"role\": \"system\", \"content\": \"Perform ASR in English.\"}, {\"role\": "
                           "\"user\", \"content\": [{\"type\": \"audio\", \"audio\": \"" + b64 +
                           "\"}]}], \"mode\": \"text\", \"max_new_tokens\": 32, \"seed\": 0, \"stream\": true}";
  size_t sink = 0;
  const double utf8 = timed(repeat, [&] { sink += lfm::lfm3_text::valid_utf8(body) ? 1 : 0; });
  const double strict = timed(repeat, [&] { sink += lfm::http::strict_json(body).at("mode").string().size(); });
  const double json_parse = timed(repeat, [&] { sink += lfm::Json::parse(body).at("mode").string().size(); });
  // validation/ttft-rest-v2 stage 0: strict_json's other parts, one at a time. The depth scan is a copy of
  // strict_json's loop, kept identical.
  const double depth_scan = timed(repeat, [&] {
    bool quoted = false, escaped = false;
    int depth = 0;
    for (const char c : body) {
      if (quoted) {
        if (escaped) escaped = false;
        else if (c == '\\') escaped = true;
        else if (c == '"') quoted = false;
      } else if (c == '"') quoted = true;
      else if (c == '[' || c == '{') { if (++depth > 64) throw std::invalid_argument("JSON nesting exceeds limit"); }
      else if (c == ']' || c == '}') --depth;
    }
    sink += size_t(depth + quoted + escaped);  // the loop's state, so it is not optimized away
  });
  const double body_copy = timed(repeat, [&] { sink += std::string(std::string_view(body)).size(); });
  const lfm::Json tree = lfm::Json::parse(body);
  const double tree_copy = timed(repeat, [&] { const lfm::Json copy = tree; sink += copy.at("mode").string().size(); });
  const auto request = lfm::http::strict_json(body);
  const double messages = timed(repeat, [&] {
    sink += lfm::lfm3_text::parse_messages(request.at("messages"), true, true).size();
  });
  const auto parsed = lfm::lfm3_text::parse_messages(request.at("messages"), true, true);
  const std::string* clip = nullptr;
  for (const auto& m : parsed)
    for (const auto& part : m.parts)
      if (part.audio) clip = &part.data;
  if (!clip) throw std::runtime_error("no audio part");
  const double base64 = timed(repeat, [&] { sink += lfm::base64::decode(*clip).size(); });
  const std::string bytes = lfm::base64::decode(*clip);
  const double wav_parse = timed(repeat, [&] { sink += lfm::wav::parse(bytes).samples.size(); });
  const double decode = timed(repeat, [&] { sink += lfm3_speech::clip_from_base64(*clip).size(); });
  const double copy = timed(repeat, [&] { sink += std::string(*clip).size(); });
  const auto pcm = lfm3_speech::clip_from_base64(*clip);
  const double mel = timed(repeat, [&] { sink += lfm::audio_features::log_mel(pcm).size(); });
  const auto features = std::make_shared<const std::vector<float>>(lfm::audio_features::log_mel(pcm));
  const double put = timed(repeat, [&] {
    lfm::audio_features::FeatureCache cache;
    sink += cache.put(*clip, features, pcm.size()) ? 1 : 0;
  });
  // the handler's own sequence: strict_json, parse_messages, decode (token count), copy + decode (features), log_mel, put
  const double handler = strict + messages + decode + copy + decode + mel + put;
  std::printf("{\"wav_bytes\":%zu,\"base64_bytes\":%zu,\"body_bytes\":%zu,\"repeat\":%d,\"valid_utf8\":%.6f,"
              "\"strict_json\":%.6f,\"json_parse\":%.6f,\"depth_scan\":%.6f,\"body_copy\":%.6f,\"tree_copy\":%.6f,\"parse_messages\":%.6f,\"base64_decode\":%.6f,\"wav_parse\":%.6f,"
              "\"clip_from_base64\":%.6f,\"clip_copy\":%.6f,\"log_mel\":%.6f,\"cache_put\":%.6f,"
              "\"handler_sequence\":%.6f,\"handler_without_log_mel\":%.6f,\"sink\":%zu}\n",
              wav.size(), b64.size(), body.size(), repeat, utf8, strict, json_parse, depth_scan, body_copy, tree_copy, messages, base64, wav_parse, decode,
              copy, mel, put, handler, handler - mel, sink);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "audio-handler-bench: %s\n", e.what());
  return 1;
}
