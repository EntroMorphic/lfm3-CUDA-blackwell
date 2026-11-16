// audio-mel-check: validation/audio-frontend-v1 rule 1. lfm::audio_features::log_mel on a list of clips, host only,
// either writing each clip's features or comparing them byte for byte with reference files.
//   audio-mel-check [--threads N] [--repeat R] (--write DIR | --compare DIR) LIST OUT.json
//   LIST: one clip per line, "NAME WAV" or "NAME WAV REF" (REF: an explicit reference file, e.g. a recorded engine
//   dump; otherwise DIR/NAME.mel.f32). --write writes DIR/NAME.mel.f32. --compare checks every clip against its
//   reference: equal size and equal bytes, else the first differing float and the largest absolute difference.
//   --repeat R: log_mel runs R times per clip and every run must equal the first (the tool fails otherwise).
//   --threads N: log_mel's thread count (the threaded front end only; the original takes none and N must be 1).
// Built from the original header it produces the reference (the bytes the engine has shipped since audio-in-v1).
// OUT.json: counts, the differing clips, and the clips' FNV-1a hashes in list order.
#include <lfm/audio_features.hpp>
#include <lfm/wav.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<float> features(const std::vector<float>& samples, int threads) {
#ifdef LFM_AUDIO_FEATURES_THREADED
  return lfm::audio_features::log_mel(samples, threads);
#else
  if (threads != 1) throw std::runtime_error("this build's log_mel is single-threaded: --threads 1 only");
  return lfm::audio_features::log_mel(samples);
#endif
}
uint64_t fnv1a(const std::vector<float>& f) {
  uint64_t h = 1469598103934665603ull;
  const auto* p = reinterpret_cast<const unsigned char*>(f.data());
  for (size_t i = 0; i < f.size() * sizeof(float); ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
std::vector<float> read_floats(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot read " + path);
  const auto n = size_t(f.tellg());
  if (n % 4) throw std::runtime_error(path + ": not float32");
  std::vector<float> v(n / 4);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n));
  return v;
}
}  // namespace

int main(int argc, char** argv) try {
  int threads = 1, repeat = 1;
  std::string mode, dir;
  std::vector<std::string> rest;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--threads") threads = std::stoi(value());
    else if (a == "--repeat") repeat = std::stoi(value());
    else if (a == "--write" || a == "--compare") mode = a, dir = value();
    else rest.push_back(a);
  }
  if (mode.empty() || rest.size() != 2 || threads < 1 || threads > 64 || repeat < 1 || repeat > 100)
    throw std::runtime_error(
        "usage: audio-mel-check [--threads N] [--repeat R] (--write DIR | --compare DIR) LIST OUT.json");
  std::ifstream list(rest[0]);
  if (!list) throw std::runtime_error("cannot read " + rest[0]);
  size_t clips = 0, identical = 0;
  std::ostringstream differ, hashes;
  std::string line;
  while (std::getline(list, line)) {
    std::istringstream ls(line);
    std::string name, wav, ref;
    if (!(ls >> name >> wav)) continue;
    ls >> ref;
    if (ref.empty()) ref = dir + "/" + name + ".mel.f32";
    const auto audio = lfm::wav::read(wav);
    if (audio.rate != lfm::audio_features::kRate) throw std::runtime_error(wav + ": not 16 kHz");
    const auto mel = features(audio.samples, threads);
    for (int r = 1; r < repeat; ++r)
      if (features(audio.samples, threads) != mel) throw std::runtime_error(name + ": log_mel differs between runs");
    const uint64_t h = fnv1a(mel);
    char hs[17];
    std::snprintf(hs, sizeof hs, "%016llx", (unsigned long long)h);
    hashes << (clips ? "," : "") << "\"" << name << "\":\"" << hs << "\"";
    ++clips;
    if (mode == "--write") {
      std::ofstream o(ref, std::ios::binary);
      o.write(reinterpret_cast<const char*>(mel.data()), std::streamsize(mel.size() * sizeof(float)));
      if (!o) throw std::runtime_error("cannot write " + ref);
      ++identical;
      continue;
    }
    const auto want = read_floats(ref);
    if (want.size() == mel.size() && std::memcmp(want.data(), mel.data(), mel.size() * sizeof(float)) == 0) {
      ++identical;
      continue;
    }
    size_t first = 0, count = 0;
    double worst = 0;
    bool found = false;
    for (size_t i = 0; i < std::min(want.size(), mel.size()); ++i)
      if (std::memcmp(&want[i], &mel[i], sizeof(float)) != 0) {
        if (!found) first = i, found = true;
        ++count;
        worst = std::max(worst, std::fabs(double(want[i]) - double(mel[i])));
      }
    differ << (differ.tellp() > 0 ? "," : "") << "{\"name\":\"" << name << "\",\"size\":" << mel.size()
           << ",\"reference_size\":" << want.size() << ",\"first\":" << first << ",\"count\":" << count
           << ",\"max_abs\":" << worst << "}";
    std::fprintf(stderr, "DIFFERS %s: %zu of %zu floats, first at %zu, max |d| %g\n", name.c_str(), count, mel.size(),
                 first, worst);
  }
  std::ofstream out(rest[1]);
  out << "{\"mode\":\"" << mode.substr(2) << "\",\"threads\":" << threads << ",\"repeat\":" << repeat
      << ",\"clips\":" << clips << ",\"identical\":" << identical << ",\"differ\":[" << differ.str()
      << "],\"fnv1a\":{" << hashes.str() << "}}\n";
  std::fprintf(stderr, "%s: %zu clips, %zu identical, threads %d\n", mode.c_str() + 2, clips, identical, threads);
  return (out && identical == clips) ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "audio-mel-check: %s\n", e.what());
  return 2;
}
