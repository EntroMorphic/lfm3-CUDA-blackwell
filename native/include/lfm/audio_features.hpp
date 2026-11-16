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
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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

// Threads one clip's features use by default (validation/audio-frontend-v1: fixed before rule 2's measurement).
inline constexpr int kMelThreads = 4;
// The spectrum by the direct sums (kDft: rule 1's bytes) or the FFT (amendment 1). The default stays kDft until the
// FFT passes amended rule 1.
enum class Transform { kDft, kFft };
inline constexpr Transform kDefaultTransform = Transform::kDft;

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
// a * W^t, W = exp(-2 pi i / 400)
inline Complex twiddle(Complex a, int t) {
  const double c = turn_cos()[size_t(t)], s = turn_sin()[size_t(t)];
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
  Complex t[5];
  for (int k = 0; k < m; ++k) {
    for (int j = 0; j < r; ++j) t[j] = j == 0 || k == 0 ? out[size_t(j) * m + k] : twiddle(out[size_t(j) * m + k], j * k * unit);
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
          const double c = turn_cos()[size_t(e)], s = turn_sin()[size_t(e)];
          re = std::fma(t[j].re, c, std::fma(t[j].im, s, re));
          im = std::fma(t[j].im, c, std::fma(-t[j].re, s, im));
        }
        out[size_t(q) * m + k] = {re, im};
      }
    }
  }
}
// Frames [t0, t1) of the padded clip x into out ([128][T]).
// kDft: kBlock frames share one pass over the basis; for each (frame, bin) the sum runs over n in ascending order, as
// the original's. kFft: each frame by fft(). Then, the same for both: the power, the mel sums and the log.
inline void log_mel_frames(const float* x, size_t T, size_t t0, size_t t1, float* out, Transform transform) {
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
inline std::vector<float> log_mel(const std::vector<float>& samples, int threads = kMelThreads,
                                  Transform transform = kDefaultTransform) {
  if (samples.empty()) throw std::runtime_error("audio features: empty clip");
  if (threads < 1) throw std::runtime_error("audio features: threads < 1");
  const size_t T = frames(samples.size());
  std::vector<float> x(padded_length(samples.size()), 0.0f);
  std::copy(samples.begin(), samples.end(), x.begin());
  (void)detail::window(), (void)detail::basis(), (void)detail::turn_cos(), (void)detail::turn_sin(), (void)mel_filters();  // built before any worker starts
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
    } catch (const std::system_error&) {  // no thread to be had: this range on the calling thread (same bytes)
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
class FeatureCache {
 public:
  using Features = std::shared_ptr<const std::vector<float>>;
  class Lease {
   public:
    Lease() = default;
    Lease(Lease&& o) noexcept : cache_(o.cache_), key_(std::move(o.key_)) { o.cache_ = nullptr; }
    Lease& operator=(Lease&& o) noexcept {
      if (this != &o) release(), cache_ = o.cache_, key_ = std::move(o.key_), o.cache_ = nullptr;
      return *this;
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease() { release(); }

   private:
    friend class FeatureCache;
    Lease(FeatureCache* cache, std::string key) : cache_(cache), key_(std::move(key)) {}
    void release() noexcept {
      if (!cache_) return;
      std::lock_guard g(cache_->lock_);
      const auto it = cache_->entries_.find(key_);
      if (it != cache_->entries_.end() && --it->second.leases == 0) cache_->entries_.erase(it);
      cache_ = nullptr;
    }
    FeatureCache* cache_ = nullptr;
    std::string key_;
  };
  // Leaves `features` under `key` (kept as they are when the key is already there: the same clip, the same bytes).
  Lease put(const std::string& key, Features features) {
    std::lock_guard g(lock_);
    auto& e = entries_[key];
    if (!e.features) e.features = std::move(features);
    ++e.leases;
    return Lease(this, key);
  }
  Features find(const std::string& key) const {
    std::lock_guard g(lock_);
    const auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : it->second.features;
  }
  size_t size() const {
    std::lock_guard g(lock_);
    return entries_.size();
  }

 private:
  struct Entry {
    Features features;
    int leases = 0;
  };
  mutable std::mutex lock_;
  std::map<std::string, Entry> entries_;
};
// The process's cache: the in-process server's Service puts, its WorkerCore finds.
inline FeatureCache& shared_feature_cache() {
  static FeatureCache cache;
  return cache;
}

}  // namespace lfm::audio_features
