// audio-mel-audit: validation/audio-frontend-v1 amendment 1, rule 1's audit. For each clip, log_mel by the direct sums
// (the current path, kDft) and by the FFT (kFft); for every float that differs, a reference value of that output in
// IEEE quad precision from the same inputs: the windowed float32 samples and the float32 mel filters as the engine
// has them, the DFT with exact quad twiddles (cos and sin of 2 pi n k / 400 in quad), the power, the mel sum, log10,
// then (max(., -12) + 4) / 4, all in quad. A difference passes if the FFT's float is at least as close to the
// reference as the current path's. One JSON line per differing float, then a summary; exits 1 if any fails.
// Usage: audio-mel-audit LIST OUT.jsonl   (LIST: "NAME WAV" per line, as audio-mel-check's)
// Quad: __float128 with libquadmath (x86-64: g++ ... -lquadmath), or long double where it is IEEE quad (aarch64).
#include <lfm/audio_features.hpp>
#include <lfm/wav.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(__x86_64__)
#include <quadmath.h>
using quad = __float128;
inline quad qcos(quad x) { return cosq(x); }
inline quad qsin(quad x) { return sinq(x); }
inline quad qlog10(quad x) { return log10q(x); }
inline quad qabs(quad x) { return fabsq(x); }
inline const quad kPi = M_PIq;
inline std::string str(quad x) {
  char b[64];
  quadmath_snprintf(b, sizeof b, "%.36Qe", x);
  return b;
}
#else
static_assert(sizeof(long double) == 16 && __LDBL_MANT_DIG__ == 113, "needs IEEE quad long double");
using quad = long double;
inline quad qcos(quad x) { return std::cos(x); }
inline quad qsin(quad x) { return std::sin(x); }
inline quad qlog10(quad x) { return std::log10(x); }
inline quad qabs(quad x) { return std::fabs(x); }
inline const quad kPi = 3.141592653589793238462643383279502884L;
inline std::string str(quad x) {
  char b[64];
  std::snprintf(b, sizeof b, "%.36Le", x);
  return b;
}
#endif

namespace af = lfm::audio_features;

// all 128 outputs of frame t of the padded clip x, in quad
std::vector<quad> reference_frame(const std::vector<float>& x, size_t t) {
  const auto& w = af::detail::window();
  const auto& filters = af::mel_filters();
  std::vector<quad> frame(af::kFft), power(af::kBins), out(af::kMels);
  for (int n = 0; n < af::kFft; ++n) frame[n] = quad(double(x[t * af::kHop + n] * w[n]));  // the engine's input, exactly
  for (int k = 0; k < af::kBins; ++k) {
    quad re = 0, im = 0;
    for (int n = 0; n < af::kFft; ++n) {
      const quad a = 2 * kPi * quad((n * k) % af::kFft) / quad(af::kFft);
      re += frame[n] * qcos(a);
      im -= frame[n] * qsin(a);
    }
    power[k] = re * re + im * im;
  }
  for (int m = 0; m < af::kMels; ++m) {
    quad s = 0;
    for (int k = 0; k < af::kBins; ++k) s += quad(filters[size_t(k) * af::kMels + m]) * power[k];
    quad l = qlog10(s);
    if (l < quad(af::kLogFloor)) l = quad(af::kLogFloor);
    out[m] = (l + quad(af::kOffset)) / quad(af::kDivisor);
  }
  return out;
}

int main(int argc, char** argv) try {
  if (argc != 3) throw std::runtime_error("usage: audio-mel-audit LIST OUT.jsonl");
  std::ifstream list(argv[1]);
  std::ofstream out(argv[2]);
  if (!list || !out) throw std::runtime_error("cannot open the list or the output");
  size_t clips = 0, clips_differ = 0, floats = 0, compared = 0, pass = 0;
  std::string line;
  while (std::getline(list, line)) {
    std::istringstream ls(line);
    std::string name, wav;
    if (!(ls >> name >> wav)) continue;
    const auto audio = lfm::wav::read(wav);
    const size_t T = af::frames(audio.samples.size());
    const auto dft = af::log_mel(audio.samples, af::kMelThreads, af::Transform::kDft);
    const auto fft = af::log_mel(audio.samples, af::kMelThreads, af::Transform::kFft);
    ++clips;
    floats += dft.size();
    if (std::memcmp(dft.data(), fft.data(), dft.size() * sizeof(float)) == 0) continue;
    ++clips_differ;
    std::vector<float> x(af::padded_length(audio.samples.size()), 0.0f);
    std::copy(audio.samples.begin(), audio.samples.end(), x.begin());
    std::map<size_t, std::vector<quad>> frames;
    for (size_t i = 0; i < dft.size(); ++i) {
      if (std::memcmp(&dft[i], &fft[i], sizeof(float)) == 0) continue;
      const size_t m = i / T, t = i % T;
      auto it = frames.find(t);
      if (it == frames.end()) it = frames.emplace(t, reference_frame(x, t)).first;
      const quad ref = it->second[m];
      const quad e_dft = qabs(quad(dft[i]) - ref), e_fft = qabs(quad(fft[i]) - ref);
      const quad mid = (quad(dft[i]) + quad(fft[i])) / 2;
      const bool ok = e_fft <= e_dft;
      ++compared;
      pass += ok;
      char d[32], f[32];
      std::snprintf(d, sizeof d, "%.9e", double(dft[i]));
      std::snprintf(f, sizeof f, "%.9e", double(fft[i]));
      out << "{\"clip\":\"" << name << "\",\"mel\":" << m << ",\"frame\":" << t << ",\"current\":" << d
          << ",\"fft\":" << f << ",\"reference\":\"" << str(ref) << "\",\"current_error\":\"" << str(e_dft)
          << "\",\"fft_error\":\"" << str(e_fft) << "\",\"reference_minus_midpoint\":\"" << str(ref - mid)
          << "\",\"fft_at_least_as_close\":" << (ok ? "true" : "false") << "}\n";
      std::fprintf(stderr, "%s mel %zu frame %zu: current %.9e fft %.9e ref %s: %s\n", name.c_str(), m, t,
                   double(dft[i]), double(fft[i]), str(ref).c_str(), ok ? "FFT at least as close" : "FFT FARTHER");
    }
  }
  std::fprintf(stderr, "%zu clips (%zu floats): %zu clips differ, %zu floats differ, %zu pass the audit, %zu fail\n",
               clips, floats, clips_differ, compared, pass, compared - pass);
  out << "{\"summary\":{\"clips\":" << clips << ",\"floats\":" << floats << ",\"clips_differ\":" << clips_differ
      << ",\"floats_differ\":" << compared << ",\"pass\":" << pass << ",\"fail\":" << compared - pass << "}}\n";
  return pass == compared ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "audio-mel-audit: %s\n", e.what());
  return 2;
}
