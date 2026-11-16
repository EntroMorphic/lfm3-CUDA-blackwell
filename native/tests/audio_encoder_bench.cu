// audio-encoder-bench: validation/audio-speed-v1 rule 0 (and audio-frontend-v1's baseline). Times the engine's own
// audio-input calls, unchanged, split where lfm3-speech's encode_seconds lumps them together:
//   - log_mel: lfm::audio_features::log_mel on the host (one thread, as the engine calls it);
//   - encode: lfm3_audio_in::Encoder::encode on the GPU, wall time on the host (the call synchronizes and frees its
//     scratch, so this is what the engine thread waits for) and its GPU span by CUDA events.
// Each clip: one untimed run of each, then R timed runs; the JSON holds every run and the median. The mel features'
// FNV-1a hash is recorded too (the front-end round's byte-identity reference; equal across runs or the tool fails).
// --fp64: also the GPU's FMA throughput in FP64 and FP32 (a kernel of independent FMA chains), for the front-end
// round's design (validation/audio-frontend-v1).
// Usage: audio-encoder-bench MODEL_DIR OUT.json [--repeat R] [--fp64] NAME=WAV...
// Timing figures are only meaningful on a quiet Thor (memory: timing checks need a quiet machine).
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include <lfm/audio_encoder.cuh>
#include <lfm/audio_features.hpp>
#include <lfm/wav.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

template <class T>
__global__ void fma_chains(T* out, int iterations, T a, T b) {
  T x[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = T(threadIdx.x + j);
  for (int i = 0; i < iterations; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = fma(x[j], a, b);
  T s = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) s += x[j];
  out[size_t(blockIdx.x) * blockDim.x + threadIdx.x] = s;  // keeps the chains live
}

// FLOP/s of fma_chains<T> (2 FLOP per FMA), the median of 5 timed launches after one untimed
template <class T>
double fma_rate() {
  int sms = 0;
  cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const int blocks = sms * 8, threads = 256, iterations = 1 << 14;
  Buffer<T> out(size_t(blocks) * threads);
  cudaEvent_t e0, e1;
  cuda_check(cudaEventCreate(&e0));
  cuda_check(cudaEventCreate(&e1));
  std::vector<double> rates;
  for (int r = 0; r < 6; ++r) {
    cuda_check(cudaEventRecord(e0));
    fma_chains<T><<<blocks, threads>>>(out.p, iterations, T(0.999999), T(1e-7));
    cuda_check(cudaEventRecord(e1));
    cuda_check(cudaEventSynchronize(e1));
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, e0, e1));
    if (r) rates.push_back(2.0 * 8 * double(iterations) * blocks * threads / (ms * 1e-3));
  }
  cuda_check(cudaEventDestroy(e0));
  cuda_check(cudaEventDestroy(e1));
  std::sort(rates.begin(), rates.end());
  return rates[rates.size() / 2];
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
std::string list(const std::vector<double>& v) {
  std::ostringstream o;
  o.precision(9);
  o << '[';
  for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
  return o.str() + ']';
}
uint64_t fnv1a(const std::vector<float>& f) {
  uint64_t h = 1469598103934665603ull;
  const auto* p = reinterpret_cast<const unsigned char*>(f.data());
  for (size_t i = 0; i < f.size() * sizeof(float); ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
std::string escape(const std::string& s) {
  std::string o;
  for (char c : s) o += (c == '"' || c == '\\') ? std::string("\\") + c : std::string(1, c);
  return o;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 4) {
    std::fprintf(stderr, "usage: audio-encoder-bench MODEL_DIR OUT.json [--repeat R] [--fp64] NAME=WAV...\n");
    return 2;
  }
  const std::string model = argv[1], out_path = argv[2];
  int repeat = 5;
  bool fp64 = false;
  std::vector<std::pair<std::string, std::string>> clips;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--repeat" && i + 1 < argc) {
      const std::string v = argv[++i];
      size_t used = 0;
      repeat = std::stoi(v, &used);
      if (used != v.size() || repeat < 1 || repeat > 1000) throw std::runtime_error("--repeat: an integer 1-1000");
    } else if (a == "--fp64") {
      fp64 = true;
    } else {
      const auto eq = a.find('=');
      if (eq == std::string::npos || eq == 0) throw std::runtime_error("expected NAME=WAV, got " + a);
      clips.emplace_back(a.substr(0, eq), a.substr(eq + 1));
    }
  }
  if (clips.empty() && !fp64) throw std::runtime_error("nothing to do");
  std::ostringstream js;
  js.precision(9);
  js << "{\"repeat\":" << repeat << ",\"clips\":[";
  if (!clips.empty()) {
    auto weights = std::make_shared<lfm::Safetensors>(model + "/model.safetensors");
    lfm3_audio_in::Encoder encoder(weights, model);
    cudaEvent_t e0, e1;
    cuda_check(cudaEventCreate(&e0));
    cuda_check(cudaEventCreate(&e1));
    for (size_t c = 0; c < clips.size(); ++c) {
      const auto audio = lfm::wav::read(clips[c].second);
      if (audio.rate != lfm::audio_features::kRate) throw std::runtime_error(clips[c].second + ": not 16 kHz");
      const size_t n = audio.samples.size(), tokens = lfm::audio_features::tokens(n);
      Buffer<__nv_bfloat16> out(tokens * lfm3_audio_in::Encoder::kOut);
      std::vector<double> mel_s, wall_s, gpu_s;
      std::vector<float> mel = lfm::audio_features::log_mel(audio.samples);  // untimed
      const uint64_t hash = fnv1a(mel);
      encoder.encode(mel, out.p);  // untimed
      for (int r = 0; r < repeat; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        mel = lfm::audio_features::log_mel(audio.samples);
        mel_s.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (fnv1a(mel) != hash) throw std::runtime_error(clips[c].first + ": log_mel differs between runs");
        cuda_check(cudaDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        cuda_check(cudaEventRecord(e0));
        const int got = encoder.encode(mel, out.p);
        cuda_check(cudaEventRecord(e1));
        cuda_check(cudaEventSynchronize(e1));
        wall_s.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, e0, e1));
        gpu_s.push_back(ms * 1e-3);
        if (size_t(got) != tokens) throw std::runtime_error("token count");
      }
      char h[17];
      std::snprintf(h, sizeof h, "%016llx", (unsigned long long)hash);
      js << (c ? "," : "") << "{\"name\":\"" << escape(clips[c].first) << "\",\"wav\":\"" << escape(clips[c].second)
         << "\",\"samples\":" << n << ",\"audio_seconds\":" << double(n) / lfm::audio_features::kRate
         << ",\"frames\":" << lfm::audio_features::frames(n) << ",\"tokens\":" << tokens << ",\"mel_fnv1a\":\"" << h
         << "\",\"log_mel_seconds\":" << median(mel_s) << ",\"encode_wall_seconds\":" << median(wall_s)
         << ",\"encode_gpu_seconds\":" << median(gpu_s) << ",\"runs\":{\"log_mel\":" << list(mel_s)
         << ",\"encode_wall\":" << list(wall_s) << ",\"encode_gpu\":" << list(gpu_s) << "}}";
      std::fprintf(stderr, "%s: %.2f s audio, %zu tokens: log_mel %.4f s, encode %.4f s (GPU span %.4f s)\n",
                   clips[c].first.c_str(), double(n) / lfm::audio_features::kRate, tokens, median(mel_s),
                   median(wall_s), median(gpu_s));
    }
    cuda_check(cudaEventDestroy(e0));
    cuda_check(cudaEventDestroy(e1));
  }
  js << "]";
  if (fp64) {
    const double d = fma_rate<double>(), f = fma_rate<float>();
    js << ",\"fma_flops\":{\"fp64\":" << d << ",\"fp32\":" << f << ",\"fp32_over_fp64\":" << f / d << "}";
    std::fprintf(stderr, "FMA throughput: FP64 %.1f GFLOP/s, FP32 %.1f GFLOP/s (%.1fx)\n", d * 1e-9, f * 1e-9, f / d);
  }
  js << "}\n";
  std::ofstream o(out_path);
  o << js.str();
  return o ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "audio-encoder-bench: %s\n", e.what());
  return 1;
}
