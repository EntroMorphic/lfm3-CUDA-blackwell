// Host tool for the audio-input parsers and front end (tests/test_audio_input_host.py): one JSON line per input.
//   audio-input-tool wav FILE...      -> {"ok":true,"rate":R,"samples":N,"sum":S} or {"ok":false,"error":"..."}
//   audio-input-tool b64 FILE...      -> {"ok":true,"hex":"..."} or {"ok":false,"error":"..."} (FILE holds the text)
//   audio-input-tool mel WAV OUT.f32  -> writes lfm::audio_features::log_mel of the clip ([128][frames] float32) and
//                                        prints {"frames":T,"tokens":N,"threads":K} (K: log_mel's thread count, LFM3_MEL_THREADS)
//   audio-input-tool mel-threads WAV  -> log_mel at 1 to 12 threads and from 8 callers at once, each byte-identical to
//                                        the 1-thread result (validation/audio-frontend-v1); {"ok":true,"frames":T}
#include <lfm/audio_features.hpp>
#include <lfm/base64.hpp>
#include <lfm/wav.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <atomic>
#include <optional>
#include <cstring>
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
  if (mode == "cache-handles") {  // validation/audio-handles-v1: the feature cache's handles and their lifetime
    using lfm::audio_features::FeatureCache;
    auto need = [](bool ok, const char* what) { if (!ok) throw std::runtime_error(std::string("cache-handles: ") + what); };
    FeatureCache cache;
    auto feats = [](float v) { return std::make_shared<const std::vector<float>>(4, v); };
    std::optional<FeatureCache::Lease> la(cache.put("clip-a", feats(1), 100)), lb(cache.put("clip-b", feats(2), 200));
    const uint64_t ha = la->handle(), hb = lb->handle();
    need(ha && hb && ha != hb, "distinct nonzero handles");
    const auto ra = cache.resolve(ha);
    need(ra && ra->samples == 100 && (*ra->features)[0] == 1 && cache.clip_of(ha) == std::string("clip-a"),
         "a handle names its own clip, features and sample count");
    auto la2 = cache.lease_if_present("clip-a");  // the same clip in another request: the same entry and handle
    need(la2 && la2->handle() == ha && cache.size() == 2, "the same clip shares an entry");
    la.reset();
    need(cache.resolve(ha).has_value(), "a handle lives while any lease holds its entry");
    la2.reset();
    need(!cache.resolve(ha) && !cache.clip_of(ha) && cache.size() == 1, "a handle outliving its leases is refused");
    const auto la3 = cache.put("clip-a", feats(1), 100);
    need(la3.handle() != ha && la3.handle() != hb, "handles are never reused");
    need(!cache.resolve(0) && !cache.resolve(~uint64_t(0)), "unknown handles are refused");
    // concurrent requests: put, lease, resolve and release from 8 threads over a few shared clips
    std::vector<std::thread> threads;
    std::atomic<int> wrong{0};
    for (int t = 0; t < 8; ++t)
      threads.emplace_back([&, t] {
        for (int i = 0; i < 2000; ++i) {
          const std::string key = "shared-" + std::to_string((t + i) % 3);
          const auto l = cache.put(key, feats(float((t + i) % 3)), size_t((t + i) % 3 + 1));
          const auto r = cache.resolve(l.handle());
          if (!r || r->samples != size_t((t + i) % 3 + 1) || cache.clip_of(l.handle()) != key) ++wrong;
        }
      });
    for (auto& t : threads) t.join();
    need(wrong == 0, "concurrent handles name their own clips");
    need(cache.size() == 2, "every shared entry released (clip-a and clip-b still held)");
    std::cout << "{\"ok\":true}\n";
    return 0;
  }
  if (mode == "exact-threads") {  // amendment 2 and mel-tiered-v1: F1, F2 and F3 the same floats (bytes) at 1 to 12
                                   // threads and from 8 callers; F3's decisions per tier reported (one run, 1 thread)
    using lfm::audio_features::Transform;
    const auto a = lfm::wav::read(argv[2]);
    const auto want = lfm::audio_features::log_mel(a.samples, 1, Transform::kExactF2);
    const auto same = [&want](const std::vector<float>& v) {
      return v.size() == want.size() && std::memcmp(v.data(), want.data(), v.size() * sizeof(float)) == 0;
    };
    auto& st = lfm::audio_features::exact_stats();
    const unsigned long long f0 = st.fast, s0 = st.second, q0 = st.quad, u0 = st.undecided;
    if (!same(lfm::audio_features::log_mel(a.samples, 1, Transform::kExactF3)))
      throw std::runtime_error("F3 differs from F2");
    const unsigned long long fast = st.fast - f0, second = st.second - s0, quad = st.quad - q0, undecided = st.undecided - u0;
    for (int t = 1; t <= 12; ++t)
      for (const auto tr : {Transform::kExactF1, Transform::kExactF2, Transform::kExactF3})
        if (!same(lfm::audio_features::log_mel(a.samples, t, tr)))
          throw std::runtime_error("F1, F2, F3 or the thread count changed the features");
    std::vector<std::vector<float>> got(9);
    std::vector<std::thread> callers;
    for (size_t i = 0; i < got.size(); ++i)
      callers.emplace_back([&a, &got, i] {
        const Transform tr[] = {Transform::kExactF1, Transform::kExactF2, Transform::kExactF3};
        got[i] = lfm::audio_features::log_mel(a.samples, 4, tr[i % 3]);
      });
    for (auto& c : callers) c.join();
    for (const auto& g : got)
      if (!same(g)) throw std::runtime_error("concurrent calls changed the features");
    std::cout << "{\"ok\":true,\"frames\":" << lfm::audio_features::frames(a.samples.size()) << ",\"f3\":{\"fast\":" << fast
              << ",\"second\":" << second << ",\"quad\":" << quad << ",\"undecided\":" << undecided << "}}\n";
    return 0;
  }
  if (mode == "mel") {
    const auto a = lfm::wav::read(argv[2]);
    const auto mel = lfm::audio_features::log_mel(a.samples);
    std::ofstream f(argv[3], std::ios::binary);
    f.write(reinterpret_cast<const char*>(mel.data()), std::streamsize(mel.size() * sizeof(float)));
    std::cout << "{\"frames\":" << lfm::audio_features::frames(a.samples.size())
              << ",\"tokens\":" << lfm::audio_features::tokens(a.samples.size())
              << ",\"threads\":" << lfm::audio_features::mel_threads() << "}\n";
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
