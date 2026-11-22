#pragma once
// Audio input (#3 audio in): the LFM3-Omni log-mel front end (feature_extraction_lfm3_omni.py, Lfm3OmniFeatureExtractor
// with the checkpoint's processor_config.json), on the host.
//
// 16 kHz mono samples, right-padded with zeros to n_fft + k * hop; frames of 400 samples every 160 (center=False),
// a periodic Hann window, the power spectrum of each frame's 201 bins, 128 Slaney mel filters over 0-8000 Hz
// (transformers.audio_utils.mel_filter_bank, norm="slaney", mel_scale="slaney"), log10 clamped below at -12, then
// (x + 4) / 4. The spectrum and the mel products are computed in double here (HF: float32 FFT and matmul), so the
// features agree with HF's to float rounding; the encoder reads them rounded to BF16.
// Layout as HF's input_features for one clip: [128 mel][frames]. Host-only.
//
// Exactness (validation/audio-frontend-v1): the bytes are those this front end has produced since audio-in-v1, when
// nvcc's host pass (GCC 13, -O3) fused its double arithmetic: the window as 0.5 - 0.5 cos in one rounding, each
// DFT sum by fused multiply-adds in ascending n, the power as fma(re, re, im * im), and each mel sum by fused
// multiply-adds in ascending bin. Those fusions are written here as std::fma, so the result no longer depends on
// how a compiler contracts. Bins and frames are independent: the DFT runs across bins in vector lanes and across
// frames in threads, each sum keeping its own order, so the bytes do not depend on the thread count either.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

// audio-mel-check and the tests use log_mel's thread count when this is defined
#define LFM_AUDIO_FEATURES_THREADED 1

namespace lfm::audio_features {

inline constexpr int kRate = 16000, kFft = 400, kHop = 160, kBins = kFft / 2 + 1, kMels = 128, kStride = 8;
// Host copies of one clip's stages (the parity check against HF): [128][T] features, the downsampler's [N][768], the
// encoder's [N][768] and the projected [N][2048], BF16 values as float.
struct Dump {
  std::vector<float> mel, down, encoded, features;
};

inline constexpr double kLogFloor = -12.0, kOffset = 4.0, kDivisor = 4.0;

// The padded length and frame count of a clip of n samples (Lfm3OmniFeatureExtractor.pad_length / num_frames).
inline size_t padded_length(size_t n) { return n < size_t(kFft) ? size_t(kFft) : n + (kHop - (n - kFft) % kHop) % kHop; }
inline size_t frames(size_t n) { return 1 + (padded_length(n) - kFft) / kHop; }
// Encoder tokens (placeholders) for a clip of n samples: one per 8 frames, rounded up.
inline size_t tokens(size_t n) { return (frames(n) + kStride - 1) / kStride; }

namespace detail {
inline double hz_to_mel(double f) {
  const double logstep = 27.0 / std::log(6.4);
  return f >= 1000.0 ? 15.0 + std::log(f / 1000.0) * logstep : 3.0 * f / 200.0;
}
inline double mel_to_hz(double m) {
  const double logstep = std::log(6.4) / 27.0;
  return m >= 15.0 ? 1000.0 * std::exp(logstep * (m - 15.0)) : 200.0 * m / 3.0;
}
// numpy.linspace(start, stop, num): start + i * step, the last element set to stop
inline std::vector<double> linspace(double start, double stop, int num) {
  std::vector<double> y(num);
  const double step = (stop - start) / (num - 1);
  for (int i = 0; i < num; ++i) y[i] = std::fma(double(i), step, start);  // fused, as shipped
  y[num - 1] = stop;
  return y;
}
}  // namespace detail

// [bin][mel], as mel_filter_bank returns it (float64), then as HF uses it (cast to float32)
inline const std::vector<float>& mel_filters() {
  static const std::vector<float> filters = [] {
    const auto mel = detail::linspace(detail::hz_to_mel(0.0), detail::hz_to_mel(kRate / 2), kMels + 2);
    std::vector<double> hz(mel.size());
    for (size_t i = 0; i < mel.size(); ++i) hz[i] = detail::mel_to_hz(mel[i]);
    const auto fft = detail::linspace(0.0, kRate / 2, kBins);
    std::vector<float> f(size_t(kBins) * kMels);
    for (int b = 0; b < kBins; ++b)
      for (int m = 0; m < kMels; ++m) {
        const double down = -(hz[m] - fft[b]) / (hz[m + 1] - hz[m]);
        const double up = (hz[m + 2] - fft[b]) / (hz[m + 2] - hz[m + 1]);
        const double tri = std::max(0.0, std::min(down, up));
        f[size_t(b) * kMels + m] = float(tri * (2.0 / (hz[m + 2] - hz[m])));
      }
    return f;
  }();
  return filters;
}

// Threads one clip's features use by default (validation/audio-frontend-v1: fixed before rule 2's measurement, kept by
// it).
inline constexpr int kMelThreads = 4;
// The process's thread count for log_mel: kMelThreads, or LFM3_MEL_THREADS (an integer 1..64, read once; anything else
// is refused). A diagnostic switch (audio-frontend-v1, quiet window 16: the stall with 1 thread against 4); the
// features do not depend on it (rule 1).
inline int mel_threads() {
  static const int n = [] {
    const char* v = std::getenv("LFM3_MEL_THREADS");
    if (!v || !*v) return kMelThreads;
    const std::string s = v;
    const bool digits = s.size() <= 2 && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; });
    const int k = digits ? std::stoi(s) : 0;
    if (k < 1 || k > 64) throw std::runtime_error("LFM3_MEL_THREADS: an integer 1..64");
    return k;
  }();
  return n;
}
// How the features are computed (validation/audio-frontend-v1):
// - kDft: today's direct sums in double (rule 1's bytes; the fallback);
// - kFft: amendment 1's double FFT, rounded as computed (failed amended rule 1; kept as the audited candidate);
// - kExactF1, kExactF2: amendment 2, every output correctly rounded from the exact mathematics. The fast path is the
//   double FFT (F1) or the compensated double-double FFT (F2) under a written error bound, and the output is
//   recomputed in quad precision wherever that bound cannot decide its rounding. F1 and F2 give the same floats.
// - kExactF3 (validation/mel-tiered-v1): F1's double FFT, and F2's double-double FFT only for a frame with an output
//   F1's bound cannot decide; quad last. The same floats as F1 and F2.
enum class Transform { kDft, kFft, kExactF1, kExactF2, kExactF3 };
// The process's default: F3, the correctly rounded front end by tiered precision (validation/mel-tiered-v1; F2 before
// it: audio-frontend-v1 amendment 2). LFM3_AUDIO_FEATURES (read once) chooses explicitly: dft (audio-in-v1's bytes,
// the fallback), fft, exact-f1, exact-f2, exact-f3.
inline Transform default_transform() {
  static const Transform t = [] {
    const char* v = std::getenv("LFM3_AUDIO_FEATURES");
    const std::string s = v ? v : "";
    if (s.empty() || s == "exact-f3") return Transform::kExactF3;
    if (s == "exact-f2") return Transform::kExactF2;
    if (s == "dft") return Transform::kDft;
    if (s == "fft") return Transform::kFft;
    if (s == "exact-f1") return Transform::kExactF1;
    throw std::runtime_error("LFM3_AUDIO_FEATURES: dft, fft, exact-f1, exact-f2 or exact-f3");
  }();
  return t;
}
// Counts of the correctly rounded front end's outputs, process-wide: decided on the fast path, decided by F3's second
// tier (the frame's double-double FFT), recomputed in quad, and left undecided even in quad (rounded from the quad value
// and reported; expected never).
struct ExactStats {
  std::atomic<uint64_t> fast{0}, second{0}, quad{0}, undecided{0};
  std::atomic<uint64_t> frames{0}, second_frames{0};  // frames computed; frames F3's second tier recomputed
};
inline ExactStats& exact_stats() {
  static ExactStats s;
  return s;
}

namespace detail {
// torch.hann_window(400) (periodic), in float32
inline const std::vector<float>& window() {
  static const std::vector<float> w = [] {
    std::vector<float> v(kFft);
    for (int n = 0; n < kFft; ++n) v[n] = float(std::fma(-0.5, std::cos(2.0 * M_PI * n / kFft), 0.5));
    return v;
  }();
  return w;
}
// The DFT's basis in the order the sums use it: [n][kBasisStride], bin k's value cos(2 pi (n k mod 400) / 400) (and
// sin), the same doubles as the original's table lookups; bins past kBins are zeros (unused lanes).
inline constexpr int kBasisStride = 204;
struct Basis {
  std::vector<double> cos, sin;
};
inline const Basis& basis() {
  static const Basis b = [] {
    std::vector<double> c(kFft), s(kFft);
    for (int i = 0; i < kFft; ++i) c[i] = std::cos(2.0 * M_PI * i / kFft);
    for (int i = 0; i < kFft; ++i) s[i] = std::sin(2.0 * M_PI * i / kFft);
    Basis t{std::vector<double>(size_t(kFft) * kBasisStride, 0.0), std::vector<double>(size_t(kFft) * kBasisStride, 0.0)};
    for (int n = 0; n < kFft; ++n)
      for (int k = 0; k < kBins; ++k) {
        const int i = (n * k) % kFft;
        t.cos[size_t(n) * kBasisStride + k] = c[i];
        t.sin[size_t(n) * kBasisStride + k] = s[i];
      }
    return t;
  }();
  return b;
}
// ---- the FFT (validation/audio-frontend-v1 amendment 1): the same transform as the direct sums, in double, by mixed
// radix (400 = 4 * 4 * 5 * 5, decimation in time). Its twiddles are the DFT's own table doubles: W^t = cos t - i sin t,
// t in 400ths of a turn. Every multiply-add is an explicit std::fma (and every other product feeds no addition), so
// the result does not depend on how a compiler contracts. It differs from the direct sums by about 1e-16 relative.
struct Complex {
  double re, im;
};
inline const std::vector<double>& turn_cos() {
  static const std::vector<double> c = [] {
    std::vector<double> v(kFft);
    for (int i = 0; i < kFft; ++i) v[i] = std::cos(2.0 * M_PI * i / kFft);
    return v;
  }();
  return c;
}
inline const std::vector<double>& turn_sin() {
  static const std::vector<double> s = [] {
    std::vector<double> v(kFft);
    for (int i = 0; i < kFft; ++i) v[i] = std::sin(2.0 * M_PI * i / kFft);
    return v;
  }();
  return s;
}
// a * W^t, W = exp(-2 pi i / 400); the tables may be passed in (fft() hoists them; the arithmetic is the same)
inline Complex twiddle(Complex a, int t, const double* cos_table = turn_cos().data(),
                       const double* sin_table = turn_sin().data()) {
  const double c = cos_table[size_t(t)], s = sin_table[size_t(t)];
  // (re + i im)(c - i s) = (re c + im s) + i (im c - re s)
  return {std::fma(a.re, c, a.im * s), std::fma(a.im, c, -(a.re * s))};
}
// out[0..n) = DFT_n of in[0], in[stride], ..., in[(n - 1) stride]; n = 400 / (the product of the radices before it)
inline void fft(const Complex* in, int stride, int n, Complex* out) {
  if (n == 1) {
    out[0] = in[0];
    return;
  }
  const int r = n % 4 == 0 ? 4 : 5;  // 400 -> 4, 100 -> 4, 25 -> 5, 5 -> 5
  const int m = n / r;
  for (int j = 0; j < r; ++j) fft(in + size_t(j) * stride, stride * r, m, out + size_t(j) * m);
  const int unit = kFft / n;  // W_n = W^unit
  const double* C = turn_cos().data();
  const double* S = turn_sin().data();
  Complex t[5];
  for (int k = 0; k < m; ++k) {
    for (int j = 0; j < r; ++j) t[j] = j == 0 || k == 0 ? out[size_t(j) * m + k] : twiddle(out[size_t(j) * m + k], j * k * unit, C, S);
    if (r == 4) {  // multiplications by -i are exact
      const Complex a0{t[0].re + t[2].re, t[0].im + t[2].im}, a1{t[0].re - t[2].re, t[0].im - t[2].im};
      const Complex b0{t[1].re + t[3].re, t[1].im + t[3].im}, b1{t[1].re - t[3].re, t[1].im - t[3].im};
      out[k] = {a0.re + b0.re, a0.im + b0.im};
      out[size_t(m) + k] = {a1.re + b1.im, a1.im - b1.re};
      out[size_t(2) * m + k] = {a0.re - b0.re, a0.im - b0.im};
      out[size_t(3) * m + k] = {a1.re - b1.im, a1.im + b1.re};
    } else {  // radix 5: X_q = sum_j t_j W5^(j q), W5 = W^80, as fused complex multiply-adds in ascending j
      for (int q = 0; q < 5; ++q) {
        double re = t[0].re, im = t[0].im;
        for (int j = 1; j < 5; ++j) {
          const int e = (j * q % 5) * (kFft / 5);
          const double c = C[size_t(e)], s = S[size_t(e)];
          re = std::fma(t[j].re, c, std::fma(t[j].im, s, re));
          im = std::fma(t[j].im, c, std::fma(-t[j].re, s, im));
        }
        out[size_t(q) * m + k] = {re, im};
      }
    }
  }
}
// ---- amendment 2: correctly rounded features ----
// Quad precision: IEEE binary128, aarch64's long double or the compilers' __float128 elsewhere. Only + - * / and
// conversions are used (no libquadmath): log and the twiddles' cos and sin come from series below.
#if LDBL_MANT_DIG == 113
using quad = long double;
#else
using quad = __float128;
#endif
inline constexpr double kU = 0x1p-53;    // double's unit roundoff
inline constexpr double kUdd = 0x1p-101; // the double-double operations' bound used here (their own: <= 2^-104 x 4)
inline const quad& quad_pi() {  // pi as three doubles (to 2^-161), summed in quad
  static const quad p = quad(0x1.921fb54442d18p+1) + quad(0x1.1a62633145c07p-53) + quad(-0x1.f1976b7ed8fbcp-109);
  return p;
}
// cos and sin of x, |x| <= pi / 4, by Taylor series to 40 terms (the next term < 2^-150)
inline void quad_cos_sin(quad x, quad& c, quad& s) {
  const quad x2 = x * x;
  quad term = 1, cs = 1, sn = x, ts = x;
  for (int n = 1; n <= 20; ++n) {
    term = -term * x2 / quad((2 * n - 1) * (2 * n));
    cs += term;
    ts = -ts * x2 / quad((2 * n) * (2 * n + 1));
    sn += ts;
  }
  c = cs, s = sn;
}
// The twiddles cos and sin of 2 pi i / 400 in quad (error a few quad ulps: the series, pi to 2^-161, and the octant
// symmetries, which are exact).
struct QuadTurn {
  std::vector<quad> cos, sin;
};
inline const QuadTurn& quad_turn() {
  static const QuadTurn t = [] {
    std::vector<quad> C(51), S(51);  // the first octant: i = 0..50 hundredths of a quarter turn
    for (int j = 0; j <= 50; ++j) quad_cos_sin(quad_pi() * quad(j) / quad(200), C[size_t(j)], S[size_t(j)]);
    QuadTurn r{std::vector<quad>(kFft), std::vector<quad>(kFft)};
    for (int i = 0; i < kFft; ++i) {
      const int q = i / 100, j = i % 100;
      const quad c = j <= 50 ? C[size_t(j)] : S[size_t(100 - j)], s = j <= 50 ? S[size_t(j)] : C[size_t(100 - j)];
      r.cos[size_t(i)] = q == 0 ? c : q == 1 ? -s : q == 2 ? -c : s;
      r.sin[size_t(i)] = q == 0 ? s : q == 1 ? c : q == 2 ? -s : -c;
    }
    return r;
  }();
  return t;
}
// 2 atanh(z) = log((1 + z) / (1 - z)), |z| <= 0.172, to 60 terms (the next term < 2^-150)
inline quad quad_log_series(quad z) {
  const quad z2 = z * z;
  quad sum = 0, p = z;
  for (int k = 0; k < 60; ++k) {
    sum += p / quad(2 * k + 1);
    p *= z2;
  }
  return 2 * sum;
}
inline const quad& quad_ln2() {
  static const quad v = [] {  // 2 atanh(1/3), to 80 terms ((1/9)^80 < 2^-250)
    const quad z = quad(1) / quad(3), z2 = z * z;
    quad sum = 0, p = z;
    for (int k = 0; k < 80; ++k) sum += p / quad(2 * k + 1), p *= z2;
    return 2 * sum;
  }();
  return v;
}
inline const quad& quad_ln10() {  // 3 ln 2 + log(1.25), log(1.25) = 2 atanh(1/9)
  static const quad v = 3 * quad_ln2() + quad_log_series(quad(1) / quad(9));
  return v;
}
// log10 of x > 0 in quad: x = 2^e m, m in [1/sqrt 2, sqrt 2), log m by the series (relative error a few quad ulps)
inline quad quad_log10(quad x) {
  int e = 0;
  (void)std::frexp(double(x), &e);  // the exponent, to within one: corrected below
  quad m = x;
  for (int k = e; k > 0; k -= 60) m /= quad(std::ldexp(1.0, std::min(k, 60)));  // exact powers of two
  for (int k = e; k < 0; k += 60) m *= quad(std::ldexp(1.0, std::min(-k, 60)));
  const quad r2 = quad(0x1.6a09e667f3bcdp+0) + quad(-0x1.bdd3413b26456p-54);  // sqrt 2 to 2^-107: the cut only
  while (m >= r2) m /= 2, ++e;
  while (m * r2 < 1) m *= 2, --e;
  return (quad(e) * quad_ln2() + quad_log_series((m - 1) / (m + 1))) / quad_ln10();
}

// Double-double numbers (error-free transformations: TwoSum and the fma TwoProduct; QD's "sloppy" add and fma mul).
struct DD {
  double hi, lo;
};
inline DD quick_two_sum(double a, double b) {
  const double s = a + b;
  return {s, b - (s - a)};
}
inline DD two_sum(double a, double b) {
  const double s = a + b, bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}
inline DD operator+(DD a, DD b) {
  DD s = two_sum(a.hi, b.hi);
  return quick_two_sum(s.hi, s.lo + (a.lo + b.lo));
}
inline DD operator-(DD a) { return {-a.hi, -a.lo}; }
inline DD operator-(DD a, DD b) { return a + -b; }
inline DD operator*(DD a, DD b) {
  const double p = a.hi * b.hi, e = std::fma(a.hi, b.hi, -p);
  return quick_two_sum(p, std::fma(a.hi, b.lo, std::fma(a.lo, b.hi, e)));
}
struct ComplexDD {
  DD re, im;
};
struct DDTurn {
  std::vector<DD> cos, sin;
};
inline const DDTurn& dd_turn() {  // the quad twiddles rounded to double-double (to 2^-106 relative)
  static const DDTurn t = [] {
    const auto& q = quad_turn();
    DDTurn r{std::vector<DD>(kFft), std::vector<DD>(kFft)};
    for (int i = 0; i < kFft; ++i) {
      const double ch = double(q.cos[size_t(i)]), sh = double(q.sin[size_t(i)]);
      r.cos[size_t(i)] = {ch, double(q.cos[size_t(i)] - quad(ch))};
      r.sin[size_t(i)] = {sh, double(q.sin[size_t(i)] - quad(sh))};
    }
    return r;
  }();
  return t;
}
// fft() in double-double: the same radices and order
inline void fft_dd(const ComplexDD* in, int stride, int n, ComplexDD* out) {
  if (n == 1) {
    out[0] = in[0];
    return;
  }
  const int r = n % 4 == 0 ? 4 : 5;
  const int m = n / r;
  for (int j = 0; j < r; ++j) fft_dd(in + size_t(j) * stride, stride * r, m, out + size_t(j) * m);
  const int unit = kFft / n;
  const auto& W = dd_turn();
  ComplexDD t[5];
  for (int k = 0; k < m; ++k) {
    for (int j = 0; j < r; ++j) {
      const ComplexDD a = out[size_t(j) * m + k];
      if (j == 0 || k == 0) {
        t[j] = a;
      } else {  // a (c - i s)
        const DD c = W.cos[size_t(j * k * unit)], s = W.sin[size_t(j * k * unit)];
        t[j] = {a.re * c + a.im * s, a.im * c - a.re * s};
      }
    }
    if (r == 4) {
      const ComplexDD a0{t[0].re + t[2].re, t[0].im + t[2].im}, a1{t[0].re - t[2].re, t[0].im - t[2].im};
      const ComplexDD b0{t[1].re + t[3].re, t[1].im + t[3].im}, b1{t[1].re - t[3].re, t[1].im - t[3].im};
      out[k] = {a0.re + b0.re, a0.im + b0.im};
      out[size_t(m) + k] = {a1.re + b1.im, a1.im - b1.re};
      out[size_t(2) * m + k] = {a0.re - b0.re, a0.im - b0.im};
      out[size_t(3) * m + k] = {a1.re - b1.im, a1.im + b1.re};
    } else {
      for (int q = 0; q < 5; ++q) {
        DD re = t[0].re, im = t[0].im;
        for (int j = 1; j < 5; ++j) {
          const int e = (j * q % 5) * (kFft / 5);
          const DD c = W.cos[size_t(e)], s = W.sin[size_t(e)];
          re = re + (t[j].re * c + t[j].im * s);
          im = im + (t[j].im * c - t[j].re * s);
        }
        out[size_t(q) * m + k] = {re, im};
      }
    }
  }
}
// The written bound (amendment 2) on every bin's components' error after the FFT, from A0 = max |x[n]|: the stages
// n = 5, 25, 100, 400 under the per-operation bounds with unit roundoff `u` and twiddle error `dw`.
inline double fft_error_bound(double a0, double u, double dw, double add_const = 2.5, double r4_const = 8.1,
                              double r5_const = 80.0) {
  double e = 0, a = a0;
  const double g = 1 + 4 * u;
  auto twiddle = [&] { e = 1.415 * e + a * (2 * dw + add_const * u), a = 1.415 * a * g; };
  auto radix4 = [&] { e = 4 * e + r4_const * u * a, a = 4 * a * g; };
  auto radix5 = [&] { e = 7 * e + a * (8 * dw + r5_const * u), a = 9 * a * g; };
  radix5();
  twiddle(), radix5();
  twiddle(), radix4();
  twiddle(), radix4();
  return e * (1 + 1e-9);
}
// The double table twiddles' largest error, |turn_cos/turn_sin - exact| over the 400 entries, measured against the
// quad twiddles (a few quad ulps, covered by 2^-200) (validation/mel-tiered-v1 amendment 1)
inline double table_twiddle_error() {
  static const double d = [] {
    const auto& q = quad_turn();
    quad m = 0;
    for (int i = 0; i < kFft; ++i)
      for (const quad diff : {quad(turn_cos()[size_t(i)]) - q.cos[size_t(i)], quad(turn_sin()[size_t(i)]) - q.sin[size_t(i)]})
        m = std::max(m, diff < 0 ? -diff : diff);
    return double(m + quad(0x1p-200)) * (1 + 0x1p-50);  // rounded up past the conversion
  }();
  return d;
}
// The 2-norm bound (validation/mel-tiered-v1 amendment 1) on the double FFT's error, every bin's complex error at most
// it, from n0 >= ||x||_2: levels n = 5, 25, 100, 400 (radices 5, 5, 4, 4); E' = sqrt(r) (E + kappa V) + b_r (1 + kappa) V
// with V = sqrt(n_prev) n0 + E; the first level has no twiddles.
inline double fft_error_bound_l2(double n0) {
  const double u = kU, dw = table_twiddle_error();
  const auto gamma = [u](int k) { return k * u / (1 - k * u); };
  const double kappa = 2 * (dw + u * (1 + dw) + u * (1 + dw) * (1 + u));
  const double b4 = 8 * gamma(2), b5 = 10 * (gamma(8) * (1 + dw) + dw);
  double E = b5 * n0;  // level 1: n = 5 from length-1 copies, no twiddles
  const struct { int r, n_prev; } levels[] = {{5, 5}, {4, 25}, {4, 100}};
  for (const auto& l : levels) {
    const double V = std::sqrt(double(l.n_prev)) * n0 + E;
    E = std::sqrt(double(l.r)) * (E + kappa * V) + (l.r == 4 ? b4 : b5) * (1 + kappa) * V;
  }
  return E * (1 + 1e-9);
}
// Each mel filter's nonzero taps, ascending: the exact sums (zero taps add exactly nothing)
struct MelTaps {
  std::vector<int> k0, k1;  // [k0, k1) per mel
};
inline const MelTaps& mel_taps() {
  static const MelTaps t = [] {
    const auto& f = mel_filters();
    MelTaps r{std::vector<int>(kMels), std::vector<int>(kMels)};
    for (int m = 0; m < kMels; ++m) {
      int lo = kBins, hi = 0;
      for (int k = 0; k < kBins; ++k)
        if (f[size_t(k) * kMels + m] != 0.0f) lo = std::min(lo, k), hi = std::max(hi, k + 1);
      r.k0[size_t(m)] = lo < hi ? lo : 0, r.k1[size_t(m)] = lo < hi ? hi : 0;
    }
    return r;
  }();
  return t;
}
// v's float32 rounding when every value within e of v rounds alike: true and `out` set; otherwise false
inline bool round_if_decided(double v, double e, float& out) {
  const float f = float(v);
  const double lo = (double(f) + double(std::nextafter(f, -INFINITY))) / 2;  // exact: adjacent floats, in double
  const double hi = (double(f) + double(std::nextafter(f, INFINITY))) / 2;
  const double ee = e * (1 + 1e-9) + 4 * kU * std::fabs(v) + 0x1p-1060;  // the comparisons' own roundings
  if (v - ee > lo && v + ee < hi) {
    out = f;
    return true;
  }
  return false;
}
// The output for one mel of one frame from its double sum m with error bound d (amendment 2), or false when the
// bound cannot decide its rounding. libm's log10 is taken within 2 ulp (glibc's documented bound), used as 8u.
inline bool output_from_mel_double(double m, double d, float& out) {
  if (m == 0 && d == 0) {  // silence: the exact sum is 0, log10 is -inf, the clamp gives -12
    out = -2.0f;
    return true;
  }
  if ((m + d) * (1 + 1e-9) < 1e-12 * (1 - 1e-9)) {  // the exact sum is below 10^-12: clamped
    out = -2.0f;
    return true;
  }
  if (m <= d) return false;
  const double L = std::log10(m);
  const double eL = d / ((m - d) * 2.302585092994045) * (1 + 1e-9) + 8 * kU * std::fabs(L);
  if (L + eL < kLogFloor) {
    out = -2.0f;
    return true;
  }
  const double Lc = std::max(L, kLogFloor);
  const double v = (Lc + kOffset) / kDivisor;
  const double e = (eL + 1.01 * kU * std::fabs(Lc + kOffset)) / kDivisor;
  return round_if_decided(v, e, out);
}
// The quad recompute of one output: frame `x` (the windowed samples), mel `mel`. Returns the float and whether quad
// decided it (counted when not).
inline float quad_output(const double* x, int mel) {
  const auto& W = quad_turn();
  const auto& taps = mel_taps();
  const auto& f = mel_filters();
  const quad uq = quad(0x1p-112);  // quad's unit roundoff, doubled
  quad l1 = 0;
  for (int n = 0; n < kFft; ++n) l1 += quad(std::fabs(x[n]));
  const quad ex = quad(1000) * uq * l1;  // each bin's component error: 400-term sums, twiddles to a few ulps
  quad m = 0, d = 0;
  int taps_n = 0;
  for (int k = taps.k0[size_t(mel)]; k < taps.k1[size_t(mel)]; ++k) {
    const quad fk = quad(f[size_t(k) * kMels + mel]);
    if (fk == 0) continue;
    quad re = 0, im = 0;
    for (int n = 0; n < kFft; ++n) {
      const int i = (n * k) % kFft;
      re += quad(x[n]) * W.cos[size_t(i)];
      im -= quad(x[n]) * W.sin[size_t(i)];
    }
    const quad p = re * re + im * im;
    const quad ar = re < 0 ? -re : re, ai = im < 0 ? -im : im;
    m += fk * p;
    d += fk * (ex * (2 * (ar + ai) + 2 * ex) + 4 * uq * p);
    ++taps_n;
  }
  d += quad(taps_n + 1) * 2 * uq * m;
  float out = -2.0f;
  bool decided = false;
  if (m == 0 && d == 0) {
    decided = true;
  } else if (m + d < quad(1e-12) * (1 - quad(0x1p-40))) {  // the exact sum is below 10^-12: clamped
    decided = true;
  } else if (m > d) {
    const quad L = quad_log10(m);
    const quad aL = L < 0 ? -L : L;
    const quad eL = d / ((m - d) * quad(2.302585092994045)) * 2 + 64 * uq * aL + quad(0x1p-200);
    if (L + eL < quad(kLogFloor)) {
      decided = true;
    } else {
      const quad Lc = L < quad(kLogFloor) ? quad(kLogFloor) : L;
      const quad v = (Lc + quad(kOffset)) / quad(kDivisor);
      const quad e = (eL + 4 * uq * (Lc + quad(kOffset) < 0 ? -(Lc + quad(kOffset)) : Lc + quad(kOffset))) /
                         quad(kDivisor) + quad(0x1p-200);
      out = float(double(v));  // a candidate: checked against quad midpoints below
      for (const float cand : {out, std::nextafter(out, -INFINITY), std::nextafter(out, INFINITY)}) {
        const quad lo = (quad(cand) + quad(std::nextafter(cand, -INFINITY))) / 2;
        const quad hi = (quad(cand) + quad(std::nextafter(cand, INFINITY))) / 2;
        if (v - e > lo && v + e < hi) {
          out = cand, decided = true;
          break;
        }
      }
      if (!decided) {  // round the quad value itself: the nearest of the three candidates
        quad best = -1;
        for (const float cand : {out, std::nextafter(out, -INFINITY), std::nextafter(out, INFINITY)}) {
          const quad dist = quad(cand) > v ? quad(cand) - v : v - quad(cand);
          if (best < 0 || dist < best) best = dist, out = cand;
        }
      }
    }
  }
  if (decided) exact_stats().quad.fetch_add(1, std::memory_order_relaxed);
  else exact_stats().undecided.fetch_add(1, std::memory_order_relaxed);
  return out;
}
// One frame's spectrum under a tier's bound (amendment 2): each bin's power (rounded to double) and a bound dpower on
// |power - exact|. F1: the double FFT (`compensated` false); F2: the double-double FFT. `l2` (F3's first tier,
// mel-tiered-v1 amendment 1): the double FFT's bound is the smaller of amendment 2's and the 2-norm bound.
inline void frame_power(const double* frame, double a0, bool compensated, double* power, double* dpower,
                        bool l2 = false) {
  // the bound's twiddle errors: the double table within 24 u of exact (amendment 2); the double-double table, from
  // the quad series, within 2^-104
  const double dw = compensated ? 0x1p-104 : 24 * kU;
  const double u = compensated ? kUdd : kU;
  double e = compensated ? fft_error_bound(a0, u, dw, 4.0, 10.0, 100.0) : fft_error_bound(a0, u, dw);
  if (l2 && !compensated) {
    double ss = 0;
    for (int n = 0; n < kFft; ++n) ss = std::fma(frame[n], frame[n], ss);
    e = std::min(e, fft_error_bound_l2(std::sqrt(ss) * (1 + 1e-9)));
  }
  double re[kBins], im[kBins];
  if (compensated) {
    ComplexDD din[kFft], dout[kFft];
    for (int n = 0; n < kFft; ++n) din[n] = {{frame[n], 0.0}, {0.0, 0.0}};
    fft_dd(din, 1, kFft, dout);
    for (int k = 0; k < kBins; ++k) {
      const DD p = dout[k].re * dout[k].re + dout[k].im * dout[k].im;
      power[k] = p.hi + p.lo;  // rounded to double
      re[k] = dout[k].re.hi, im[k] = dout[k].im.hi;
    }
  } else {
    Complex zin[kFft], zout[kFft];
    for (int n = 0; n < kFft; ++n) zin[n] = {frame[n], 0.0};
    fft(zin, 1, kFft, zout);
    for (int k = 0; k < kBins; ++k) {
      re[k] = zout[k].re, im[k] = zout[k].im;
      power[k] = std::fma(re[k], re[k], im[k] * im[k]);
    }
  }
  for (int k = 0; k < kBins; ++k)  // |P^ - P| (amendment 2), with margin for the double-double rounding
    dpower[k] = (e * (2 * (std::fabs(re[k]) + std::fabs(im[k])) + 4 * e) + 2.02 * kU * power[k]) * (1 + 1e-9);
}
// Mel `m`'s output from a frame's power and bound: true and `out` set when the bound decides its rounding
inline bool mel_output(const double* power, const double* dpower, int m, float& out,
                       const std::vector<float>& filters = mel_filters(), const MelTaps& taps = mel_taps()) {
  double sum = 0, bound = 0;
  int n_taps = 0;
  for (int k = taps.k0[size_t(m)]; k < taps.k1[size_t(m)]; ++k) {
    const double fk = double(filters[size_t(k) * kMels + m]);
    if (fk == 0) continue;
    sum = std::fma(fk, power[k], sum);
    bound = std::fma(fk, dpower[k], bound);
    ++n_taps;
  }
  bound = bound * (1 + 1e-9) + (n_taps + 1) * 1.01 * kU * sum;
  return output_from_mel_double(sum, bound, out);
}
// Frames [t0, t1) of the padded clip x into out, correctly rounded. F1: the double FFT, quad where its bound cannot
// decide; F2: the double-double FFT, likewise. F3 (validation/mel-tiered-v1, `tiered`): the double FFT; where its
// bound cannot decide an output, the double-double FFT of that frame (computed once per frame) under F2's bound; quad
// last. Every output is the correctly rounded value of the same exact mathematics, so all three give the same floats.
inline void exact_frames(const float* x, size_t T, size_t t0, size_t t1, float* out, bool compensated,
                         bool tiered = false) {
  const auto& w = window();
  const auto& filters = mel_filters();
  const auto& taps = mel_taps();
  double frame[kFft], power[kBins], dpower[kBins], power_dd[kBins], dpower_dd[kBins];
  uint64_t fast = 0, second = 0, second_frames = 0;
  for (size_t t = t0; t < t1; ++t) {
    double a0 = 0;
    for (int n = 0; n < kFft; ++n) {
      frame[n] = double(x[t * kHop + n] * w[n]);
      a0 = std::max(a0, std::fabs(frame[n]));
    }
    frame_power(frame, a0, compensated && !tiered, power, dpower, tiered);
    bool have_dd = false;
    for (int m = 0; m < kMels; ++m) {
      float v;
      if (mel_output(power, dpower, m, v, filters, taps)) {
        ++fast;
      } else if (tiered) {
        if (!have_dd) frame_power(frame, a0, true, power_dd, dpower_dd), have_dd = true, ++second_frames;
        if (mel_output(power_dd, dpower_dd, m, v, filters, taps)) ++second;
        else v = quad_output(frame, m);
      } else {
        v = quad_output(frame, m);
      }
      out[size_t(m) * T + t] = v;
    }
  }
  exact_stats().fast.fetch_add(fast, std::memory_order_relaxed);
  exact_stats().second.fetch_add(second, std::memory_order_relaxed);
  exact_stats().frames.fetch_add(t1 - t0, std::memory_order_relaxed);
  exact_stats().second_frames.fetch_add(second_frames, std::memory_order_relaxed);
}
// Frames [t0, t1) of the padded clip x into out ([128][T]).
// kDft: kBlock frames share one pass over the basis; for each (frame, bin) the sum runs over n in ascending order, as
// the original's. kFft: each frame by fft(). Then, the same for both: the power, the mel sums and the log.
inline void log_mel_frames(const float* x, size_t T, size_t t0, size_t t1, float* out, Transform transform) {
  if (transform == Transform::kExactF1 || transform == Transform::kExactF2 || transform == Transform::kExactF3)
    return exact_frames(x, T, t0, t1, out, transform == Transform::kExactF2, transform == Transform::kExactF3);
  constexpr int kBlock = 4;
  const auto& w = window();
  const auto& b = basis();
  const auto& filters = mel_filters();
  alignas(64) double frame[kBlock][kFft], re[kBlock][kBasisStride], im[kBlock][kBasisStride], mel[kMels];
  Complex zin[kFft], zout[kFft];
  for (size_t tb = t0; tb < t1; tb += kBlock) {
    const int nb = int(std::min<size_t>(kBlock, t1 - tb));
    for (int j = 0; j < nb; ++j)
      for (int n = 0; n < kFft; ++n) frame[j][n] = double(x[(tb + j) * kHop + n] * w[n]);  // the windowed float32 sample
    if (transform == Transform::kFft) {
      for (int j = 0; j < nb; ++j) {
        for (int n = 0; n < kFft; ++n) zin[n] = {frame[j][n], 0.0};
        fft(zin, 1, kFft, zout);
        for (int k = 0; k < kBins; ++k) re[j][k] = zout[k].re, im[j][k] = zout[k].im;
      }
    } else {
      for (int j = 0; j < nb; ++j)
        for (int k = 0; k < kBasisStride; ++k) re[j][k] = im[j][k] = 0.0;
      for (int n = 0; n < kFft; ++n) {
        const double* c = b.cos.data() + size_t(n) * kBasisStride;
        const double* s = b.sin.data() + size_t(n) * kBasisStride;
        for (int j = 0; j < nb; ++j) {
          const double f = frame[j][n], g = -f;
          double* r = re[j];
          double* q = im[j];
          for (int k = 0; k < kBasisStride; ++k) {
            r[k] = std::fma(f, c[k], r[k]);  // re += f cos, one rounding
            q[k] = std::fma(g, s[k], q[k]);  // im -= f sin, one rounding
          }
        }
      }
    }
    for (int j = 0; j < nb; ++j) {
      for (int m = 0; m < kMels; ++m) mel[m] = 0.0;
      for (int k = 0; k < kBins; ++k) {
        const double power = std::fma(re[j][k], re[j][k], im[j][k] * im[j][k]);
        const float* fk = filters.data() + size_t(k) * kMels;
        for (int m = 0; m < kMels; ++m) mel[m] = std::fma(double(fk[m]), power, mel[m]);
      }
      const size_t t = tb + j;
      for (int m = 0; m < kMels; ++m)
        out[size_t(m) * T + t] = float((std::max(std::log10(mel[m]), kLogFloor) + kOffset) / kDivisor);
    }
  }
}
}  // namespace detail

// The clip's features, [128][frames(n)] (row-major: mel-major, as HF's input_features[0]), computed on up to
// `threads` threads (the calling thread included; at least 64 frames each). The result does not depend on `threads`.
inline std::vector<float> log_mel(const std::vector<float>& samples, int threads = mel_threads(),
                                  Transform transform = default_transform()) {
  if (samples.empty()) throw std::runtime_error("audio features: empty clip");
  if (threads < 1) throw std::runtime_error("audio features: threads < 1");
  const size_t T = frames(samples.size());
  std::vector<float> x(padded_length(samples.size()), 0.0f);
  std::copy(samples.begin(), samples.end(), x.begin());
  // every table built before any worker starts
  (void)detail::window(), (void)detail::basis(), (void)detail::turn_cos(), (void)detail::turn_sin(), (void)mel_filters();
  if (transform == Transform::kExactF1 || transform == Transform::kExactF2 || transform == Transform::kExactF3)
    (void)detail::quad_turn(), (void)detail::dd_turn(), (void)detail::mel_taps(), (void)detail::quad_ln10();
  if (transform == Transform::kExactF3) (void)detail::table_twiddle_error();
  std::vector<float> out(size_t(kMels) * T);
  const size_t n = std::max<size_t>(1, std::min<size_t>(size_t(threads), T / 64));
  const size_t per = (T + n - 1) / n;
  std::vector<std::thread> workers;
  workers.reserve(n - 1);  // before any thread starts: a throw here leaves nothing running
  for (size_t i = 1; i < n; ++i) {
    const size_t a = std::min(T, i * per), e = std::min(T, a + per);
    if (a >= e) continue;
    try {
      workers.emplace_back(detail::log_mel_frames, x.data(), T, a, e, out.data(), transform);
    } catch (...) {  // no thread to be had (EAGAIN, or bad_alloc: red-team finding 4): this range here, the same bytes
      detail::log_mel_frames(x.data(), T, a, e, out.data(), transform);
    }
  }
  detail::log_mel_frames(x.data(), T, 0, std::min(T, per), out.data(), transform);
  for (auto& w : workers) w.join();
  return out;
}

// Features computed ahead of the engine (validation/audio-frontend-v1). The in-process server computes a clip's
// features on the request's handler thread and leaves them here, keyed by the clip as the engine request carries it
// (its base64 text, byte for byte); the engine takes them instead of computing them on the thread that steps every
// row. A Lease holds an entry until the request ends (the same clip in two requests: one entry, two leases). A clip
// the engine does not find here (the out-of-process worker, lfm3-speech, a refused or expired request) is computed
// by the engine as before: log_mel is a function of the clip, so a hit and a miss give the same bytes.
//
// Handles (validation/audio-handles-v1): each entry has a handle, a number assigned when the entry is created and never
// reused in the process. The in-process server sends the engine a clip's handle instead of its base64 when the entry
// has the clip's features and sample count; the engine resolves it (`resolve`). A handle resolves only while some
// lease holds its entry: the sending request's lease lasts until the request ends, so a handle cannot outlive the
// request that sent it, or name another clip.
class FeatureCache {
  struct Entry {
    std::shared_ptr<const std::vector<float>> features;
    size_t samples = 0;    // the clip's sample count (set with the features)
    uint64_t handle = 0;   // assigned at creation, never reused
    int leases = 0;
  };
  using Map = std::map<std::string, Entry, std::less<>>;

 public:
  using Features = std::shared_ptr<const std::vector<float>>;
  // An entry held for one request. It keeps the entry's map position (stable until erased), not a copy of its key:
  // the key, a clip's base64 text, can be megabytes (audio-frontend-v1 red-team finding 2).
  class Lease {
   public:
    Lease() = default;
    Lease(Lease&& o) noexcept : cache_(o.cache_), it_(o.it_) { o.cache_ = nullptr; }
    Lease& operator=(Lease&& o) noexcept {
      if (this != &o) release(), cache_ = o.cache_, it_ = o.it_, o.cache_ = nullptr;
      return *this;
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease() { release(); }
    explicit operator bool() const noexcept { return cache_ != nullptr; }
    // the entry's handle (immutable while the entry exists, which this lease guarantees)
    uint64_t handle() const noexcept { return cache_ ? it_->second.handle : 0; }

   private:
    friend class FeatureCache;
    Lease(FeatureCache* cache, Map::iterator it) noexcept : cache_(cache), it_(it) {}
    void release() noexcept {
      if (!cache_) return;
      std::lock_guard g(cache_->lock_);
      if (--it_->second.leases == 0) {
        cache_->by_handle_.erase(it_->second.handle);
        cache_->entries_.erase(it_);
      }
      cache_ = nullptr;
    }
    FeatureCache* cache_ = nullptr;
    Map::iterator it_{};
  };
  // Leaves `features` (of a clip of `samples` samples) under `key` (kept as they are when the key already has features:
  // the same clip, the same bytes). The map position and the handle's index come first, so a failed allocation leaves
  // no lease counted and no half-made entry (red-team finding 3).
  Lease put(std::string_view key, Features features, size_t samples) {
    std::lock_guard g(lock_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      it = entries_.emplace(std::string(key), Entry{}).first;
      try {
        it->second.handle = next_handle_;
        by_handle_.emplace(next_handle_, it);
        ++next_handle_;
      } catch (...) {
        entries_.erase(it);
        throw;
      }
    }
    if (!it->second.features) it->second.features = std::move(features), it->second.samples = samples;
    ++it->second.leases;
    return Lease(this, it);
  }
  // What a handle names while a lease holds its entry: the clip's features and sample count. Nothing otherwise (an
  // unknown or expired handle, or an entry without features).
  struct Resolved {
    Features features;
    size_t samples = 0;
  };
  std::optional<Resolved> resolve(uint64_t handle) const {
    std::lock_guard g(lock_);
    const auto h = by_handle_.find(handle);
    if (h == by_handle_.end() || !h->second->second.features) return std::nullopt;
    return Resolved{h->second->second.features, h->second->second.samples};
  }
  // The clip (its base64 text) a live handle names. For tests (the HTTP harness checks a handle's features against
  // log_mel of its clip); the engine never needs it.
  std::optional<std::string> clip_of(uint64_t handle) const {
    std::lock_guard g(lock_);
    const auto h = by_handle_.find(handle);
    if (h == by_handle_.end()) return std::nullopt;
    return h->second->first;
  }
  // A lease on the entry for `key` when its features are already there (the same clip in flight in another request,
  // or twice in one): nothing to compute. Otherwise nothing.
  std::optional<Lease> lease_if_present(std::string_view key) {
    std::lock_guard g(lock_);
    const auto it = entries_.find(key);
    if (it == entries_.end() || !it->second.features) return std::nullopt;
    ++it->second.leases;
    return Lease(this, it);
  }
  Features find(std::string_view key) const {
    std::lock_guard g(lock_);
    const auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : it->second.features;
  }
  size_t size() const {
    std::lock_guard g(lock_);
    return entries_.size();
  }

 private:
  mutable std::mutex lock_;
  Map entries_;
  std::map<uint64_t, Map::iterator> by_handle_;
  uint64_t next_handle_ = 1;
};
// The process's cache: the in-process server's Service puts, its WorkerCore finds.
inline FeatureCache& shared_feature_cache() {
  static FeatureCache cache;
  return cache;
}

}  // namespace lfm::audio_features
