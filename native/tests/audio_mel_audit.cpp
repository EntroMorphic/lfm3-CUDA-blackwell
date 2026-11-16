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

#include <array>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>
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

// ---- amendment 2's rule 1: every output against an independent quad reference ----
// The reference: a mixed-radix FFT in quad written here (any radices; twiddles from this platform's quad cos and sin
// of 2 pi t / 400), the power, the mel sums over each filter's nonzero taps, the quad log10, the clamp and
// (+ 4) / 4; then the float32 rounding decided under a generous bound (1e6 quad ulps of A0 on every bin), or reported
// undecided.
void quad_fft(const std::vector<quad>& re_in, const std::vector<quad>& im_in, size_t off, size_t stride, int n,
              quad* re, quad* im, const std::vector<quad>& C, const std::vector<quad>& S) {
  if (n == 1) {
    re[0] = re_in[off], im[0] = im_in[off];
    return;
  }
  int r = 2;
  while (n % r) ++r;
  const int m = n / r;
  for (int j = 0; j < r; ++j) quad_fft(re_in, im_in, off + j * stride, stride * r, m, re + j * m, im + j * m, C, S);
  std::vector<quad> tr(r), ti(r);
  const int unit = af::kFft / n;
  for (int k = 0; k < m; ++k) {
    for (int j = 0; j < r; ++j) {
      const int t = (j * k * unit) % af::kFft;
      const quad a = re[j * m + k], b = im[j * m + k];
      tr[j] = a * C[t] + b * S[t], ti[j] = b * C[t] - a * S[t];
    }
    for (int q = 0; q < r; ++q) {
      quad sr = 0, si = 0;
      for (int j = 0; j < r; ++j) {
        const int t = (j * q * m * unit) % af::kFft;  // W_r^(j q) = W^(j q n / r * 400 / n)
        sr += tr[j] * C[t] + ti[j] * S[t];
        si += ti[j] * C[t] - tr[j] * S[t];
      }
      re[q * m + k] = sr, im[q * m + k] = si;
    }
  }
}
struct Reference {
  float value;
  bool decided;
  std::string exact;  // the quad value
};
std::vector<Reference> reference_clip(const std::vector<float>& samples) {
  static const std::vector<quad> C = [] {
    std::vector<quad> v(af::kFft);
    for (int t = 0; t < af::kFft; ++t) v[t] = qcos(2 * kPi * quad(t) / quad(af::kFft));
    return v;
  }();
  static const std::vector<quad> S = [] {
    std::vector<quad> v(af::kFft);
    for (int t = 0; t < af::kFft; ++t) v[t] = qsin(2 * kPi * quad(t) / quad(af::kFft));
    return v;
  }();
  const auto& w = af::detail::window();
  const auto& filters = af::mel_filters();
  const size_t T = af::frames(samples.size());
  std::vector<float> x(af::padded_length(samples.size()), 0.0f);
  std::copy(samples.begin(), samples.end(), x.begin());
  std::vector<Reference> out(size_t(af::kMels) * T);
  std::vector<quad> fr(af::kFft), fi(af::kFft, 0), re(af::kFft), im(af::kFft), power(af::kBins);
  const quad uq = quad(1) / quad(1ull << 56) / quad(1ull << 57);  // 2^-113
  for (size_t t = 0; t < T; ++t) {
    quad a0 = 0;
    for (int n = 0; n < af::kFft; ++n) {
      fr[n] = quad(double(x[t * af::kHop + n] * w[n]));
      a0 = std::max(a0, qabs(fr[n]));
    }
    quad_fft(fr, fi, 0, 1, af::kFft, re.data(), im.data(), C, S);
    const quad e = 1e6 * uq * a0;
    for (int k = 0; k < af::kBins; ++k) power[k] = re[k] * re[k] + im[k] * im[k];
    for (int m = 0; m < af::kMels; ++m) {
      quad s = 0, d = 0;
      int taps = 0;
      for (int k = 0; k < af::kBins; ++k) {
        const float f = filters[size_t(k) * af::kMels + m];
        if (f == 0.0f) continue;
        s += quad(f) * power[k];
        d += quad(f) * (e * (2 * (qabs(re[k]) + qabs(im[k])) + 2 * e) + 8 * uq * power[k]);
        ++taps;
      }
      d += quad(taps + 2) * 4 * uq * s;
      Reference& r = out[size_t(m) * T + t];
      if (s == 0 && d == 0) {
        r = {-2.0f, true, "-2"};
        continue;
      }
      if (s + d < quad(1e-12) * (1 - quad(1e-30))) {  // the exact sum below 10^-12: clamped
        r = {-2.0f, true, "-2 (clamped)"};
        continue;
      }
      if (s <= d) {
        r = {0.0f, false, "sum not resolved"};
        continue;
      }
      const quad L = qlog10(s), eL = d / ((s - d) * quad(2.3)) + 64 * uq * qabs(L) + quad(1e-60);
      if (L + eL < quad(af::kLogFloor)) {
        r = {-2.0f, true, "-2 (clamped)"};
        continue;
      }
      const quad Lc = L < quad(af::kLogFloor) ? quad(af::kLogFloor) : L;
      const quad v = (Lc + quad(af::kOffset)) / quad(af::kDivisor);
      const quad ev = (eL + 8 * uq * qabs(Lc + quad(af::kOffset))) / quad(af::kDivisor) + quad(1e-60);
      const float c0 = float(double(v));
      r = {c0, false, str(v)};
      for (const float c : {c0, std::nextafter(c0, -INFINITY), std::nextafter(c0, INFINITY)}) {
        const quad lo = (quad(c) + quad(std::nextafter(c, -INFINITY))) / 2, hi = (quad(c) + quad(std::nextafter(c, INFINITY))) / 2;
        if (v - ev > lo && v + ev < hi) {
          r.value = c, r.decided = true;
          break;
        }
      }
    }
  }
  return out;
}
// audio-mel-audit exact f1|f2 THREADS LIST OUT.jsonl: LIST lines "NAME WAV REF" (REF: today's bytes for that clip).
int exact_main(int argc, char** argv) {
  if (argc != 6) throw std::runtime_error("usage: audio-mel-audit exact f1|f2 THREADS LIST OUT.jsonl");
  const std::string which = argv[2];
  const auto transform = which == "f1" ? af::Transform::kExactF1 : af::Transform::kExactF2;
  if (which != "f1" && which != "f2") throw std::runtime_error("exact: f1 or f2");
  const int workers = std::stoi(argv[3]);
  std::vector<std::array<std::string, 3>> clips;
  {
    std::ifstream list(argv[4]);
    std::string line;
    while (std::getline(list, line)) {
      std::istringstream ls(line);
      std::array<std::string, 3> c;
      if (ls >> c[0] >> c[1] >> c[2]) clips.push_back(c);
    }
  }
  std::ofstream out(argv[5]);
  std::mutex lock;
  std::atomic<size_t> next{0};
  size_t floats = 0, wrong = 0, undecided = 0, changed = 0, changed_closer = 0;
  auto work = [&] {
    for (size_t i; (i = next++) < clips.size();) {
      const auto& c = clips[i];
      const auto audio = lfm::wav::read(c[1]);
      const size_t T = af::frames(audio.samples.size());
      const auto fast = af::log_mel(audio.samples, 1, transform);
      const auto ref = reference_clip(audio.samples);
      std::vector<float> today(fast.size());
      {
        std::ifstream f(c[2], std::ios::binary);
        f.read(reinterpret_cast<char*>(today.data()), std::streamsize(today.size() * sizeof(float)));
        if (!f || f.peek() != EOF) throw std::runtime_error(c[2] + ": not this clip's features");
      }
      std::ostringstream lines;
      size_t w = 0, u = 0, ch = 0, cl = 0;
      for (size_t j = 0; j < fast.size(); ++j) {
        const auto& r = ref[j];
        const bool same_as_today = std::memcmp(&fast[j], &today[j], 4) == 0;
        if (!r.decided) ++u;
        const bool ok = r.decided && std::memcmp(&fast[j], &r.value, 4) == 0;
        if (r.decided && !ok) ++w;
        if (!same_as_today) {
          ++ch;
          const bool closer = r.decided && ok;  // correctly rounded: the nearest float, so strictly closer than today's
          cl += closer;
        }
        if (!ok || !same_as_today) {
          char a[32], b[32];
          std::snprintf(a, sizeof a, "%.9e", double(fast[j]));
          std::snprintf(b, sizeof b, "%.9e", double(today[j]));
          lines << "{\"clip\":\"" << c[0] << "\",\"mel\":" << j / T << ",\"frame\":" << j % T << ",\"new\":" << a
                << ",\"today\":" << b << ",\"reference\":\"" << r.exact << "\",\"decided\":" << (r.decided ? "true" : "false")
                << ",\"new_is_reference_rounded\":" << (ok ? "true" : "false") << "}\n";
        }
      }
      std::lock_guard g(lock);
      out << lines.str();
      floats += fast.size(), wrong += w, undecided += u, changed += ch, changed_closer += cl;
    }
  };
  std::vector<std::thread> pool;
  for (int k = 0; k < workers; ++k) pool.emplace_back(work);
  for (auto& t : pool) t.join();
  const auto& st = af::exact_stats();
  out << "{\"summary\":{\"transform\":\"" << which << "\",\"clips\":" << clips.size() << ",\"floats\":" << floats
      << ",\"not_the_reference\":" << wrong << ",\"reference_undecided\":" << undecided << ",\"changed_from_today\":" << changed
      << ",\"changed_and_correct\":" << changed_closer << ",\"fast\":" << st.fast.load() << ",\"quad\":" << st.quad.load()
      << ",\"quad_undecided\":" << st.undecided.load() << "}}\n";
  std::fprintf(stderr, "%s: %zu clips, %zu floats: %zu not the reference's rounding, %zu undecided by the reference; "
               "%zu changed from today (%zu of them the reference's rounding); fast %llu, quad %llu, quad undecided %llu\n",
               which.c_str(), clips.size(), floats, wrong, undecided, changed, changed_closer,
               (unsigned long long)st.fast.load(), (unsigned long long)st.quad.load(), (unsigned long long)st.undecided.load());
  return wrong == 0 && undecided == 0 && changed == changed_closer ? 0 : 1;
}

int main(int argc, char** argv) try {
  if (argc > 1 && std::string(argv[1]) == "exact") return exact_main(argc, argv);
  if (argc != 3) throw std::runtime_error("usage: audio-mel-audit LIST OUT.jsonl | exact f1|f2 THREADS LIST OUT.jsonl");
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
