// alloc-extent-probe: validation/alloc-effect-v1 stage 4 (a diagnostic). Stage 3: one 30 MiB host-to-device cudaMemcpy
// from pageable memory makes the mma.sync GEMM at the prefill's slowed shape ~1.9x slower; pinned memory does not; a
// 1 s idle restores; tcgen05 is immune. How far does it reach into the engine?
//   size:    after a 1.5 s idle (restored), 6 launches of the fixed pair (pre), one copy, 6 launches (post); the copy:
//            pageable H2D of 4 KiB, 64 KiB, 1 MiB, 8 MiB, 30 MiB; pageable cudaMemcpyAsync H2D of 64 KiB; pageable
//            D2H of 4 KiB and 30 MiB. 3 rounds, rotated.
//   restore: a 30 MiB pageable H2D, then a host sleep of 0, 1, 10, 100, 300 or 1000 ms, then 6 launches (against the
//            pre of the same trial). 3 rounds, rotated.
//   persist: a 30 MiB pageable H2D, then 2,000 back-to-back launches, medians per 100.
//   shapes:  each engine GEMM shape at decode and prefill row counts, plus a streaming write and a streaming read of
//            30 MiB: 6 launches after a 1.5 s idle (fast), then a 30 MiB pageable H2D, then 6 launches (slow). Each
//            through lfm::gemm::run, so the binary's dispatch decides the kernel (tcgen05 where it is built and
//            eligible). 2 rounds.
// Times are medians of 6 single-launch event timings. One JSON line per trial.
// Usage: alloc-extent-probe
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include <lfm/gemm.cuh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {
using bf16 = __nv_bfloat16;
__global__ void fill(bf16* p, size_t n, float scale) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2bfloat16_rn(scale * float(int(i % 1013) - 506));
}
__global__ void sum(const bf16* p, size_t n, float* out) {
  float s = 0;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    s += __bfloat162float(p[i]);
  if (s == 12345.f) *out = s;
}
void sleep_ms(double ms) { std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms)); }
double median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
}  // namespace

int main() try {
  const int M = 1149, N = 7680, K = 2048;
  const size_t big = size_t(30) << 20;
  const auto blocks = [](size_t n) { return int((n + 255) / 256); };
  // operands large enough for every shape: A up to 4096 x 8192, B up to 7680 x 8192, C up to 4096 x 7680 (FP32)
  bf16 *a, *b, *spare;
  float *c, *out;
  const size_t a_n = size_t(4096) * 8192, b_n = size_t(7680) * 8192;
  cuda_check(cudaMalloc(&a, a_n * 2));
  cuda_check(cudaMalloc(&b, b_n * 2));
  cuda_check(cudaMalloc(&c, size_t(4096) * 7680 * 4));
  cuda_check(cudaMalloc(&spare, big));
  cuda_check(cudaMalloc(&out, 4));
  fill<<<blocks(a_n), 256>>>(a, a_n, 1e-3f);
  fill<<<blocks(b_n), 256>>>(b, b_n, 1e-3f);
  cuda_check(cudaDeviceSynchronize());
  std::vector<char> host(big, 1);
  cudaEvent_t e0, e1;
  cuda_check(cudaEventCreate(&e0));
  cuda_check(cudaEventCreate(&e1));
  auto time6 = [&](const std::function<void()>& f) {
    std::vector<float> t;
    for (int r = 0; r < 6; ++r) {
      cuda_check(cudaEventRecord(e0));
      f();
      cuda_check(cudaEventRecord(e1));
      cuda_check(cudaEventSynchronize(e1));
      float ms = 0;
      cuda_check(cudaEventElapsedTime(&ms, e0, e1));
      t.push_back(ms);
    }
    return median(t);
  };
  auto gemm = [&](int m, int n, int k, bool f32) {
    auto p = lfm::gemm::linear(a, b, nullptr, m, k, n);
    if (f32) p.c_f32 = c; else p.c = reinterpret_cast<bf16*>(c);
    return [p] { lfm::gemm::run(p); };
  };
  const auto fixed = gemm(M, N, K, false);
  auto induce = [&] { cuda_check(cudaMemcpy(spare, host.data(), big, cudaMemcpyHostToDevice)); };
  auto rest = [&] { cuda_check(cudaDeviceSynchronize()); sleep_ms(1500); };
  fixed();
  cuda_check(cudaDeviceSynchronize());
  // size
  const std::vector<std::pair<std::string, std::function<void()>>> copies = {
      {"h2d_4k", [&] { cuda_check(cudaMemcpy(spare, host.data(), 4 << 10, cudaMemcpyHostToDevice)); }},
      {"h2d_64k", [&] { cuda_check(cudaMemcpy(spare, host.data(), 64 << 10, cudaMemcpyHostToDevice)); }},
      {"h2d_1m", [&] { cuda_check(cudaMemcpy(spare, host.data(), 1 << 20, cudaMemcpyHostToDevice)); }},
      {"h2d_8m", [&] { cuda_check(cudaMemcpy(spare, host.data(), 8 << 20, cudaMemcpyHostToDevice)); }},
      {"h2d_30m", induce},
      {"h2d_async_64k", [&] {
         cuda_check(cudaMemcpyAsync(spare, host.data(), 64 << 10, cudaMemcpyHostToDevice));
         cuda_check(cudaStreamSynchronize(nullptr));
       }},
      {"d2h_4k", [&] { cuda_check(cudaMemcpy(host.data(), spare, 4 << 10, cudaMemcpyDeviceToHost)); }},
      {"d2h_30m", [&] { cuda_check(cudaMemcpy(host.data(), spare, big, cudaMemcpyDeviceToHost)); }},
  };
  for (int r = 0; r < 3; ++r)
    for (size_t k = 0; k < copies.size(); ++k) {
      const auto& [name, copy] = copies[(k + 3 * r) % copies.size()];
      rest();
      const double pre = time6(fixed);
      copy();
      const double post = time6(fixed);
      std::printf("{\"part\":\"size\",\"round\":%d,\"copy\":\"%s\",\"pre\":%.4f,\"post\":%.4f}\n", r + 1, name.c_str(), pre, post);
      std::fflush(stdout);
    }
  // restore
  const std::vector<double> gaps = {0, 1, 10, 100, 300, 1000};
  for (int r = 0; r < 3; ++r)
    for (size_t k = 0; k < gaps.size(); ++k) {
      const double g = gaps[(k + 2 * r) % gaps.size()];
      rest();
      const double pre = time6(fixed);
      induce();
      cuda_check(cudaDeviceSynchronize());
      if (g > 0) sleep_ms(g);
      const double post = time6(fixed);
      std::printf("{\"part\":\"restore\",\"round\":%d,\"gap_ms\":%g,\"pre\":%.4f,\"post\":%.4f}\n", r + 1, g, pre, post);
      std::fflush(stdout);
    }
  // persist
  {
    rest();
    const double pre = time6(fixed);
    induce();
    std::string s;
    for (int h = 0; h < 20; ++h) {
      std::vector<float> t;
      for (int i = 0; i < 100; ++i) {
        cuda_check(cudaEventRecord(e0));
        fixed();
        cuda_check(cudaEventRecord(e1));
        cuda_check(cudaEventSynchronize(e1));
        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, e0, e1));
        t.push_back(ms);
      }
      s += (s.empty() ? "" : ",") + std::to_string(median(t)).substr(0, 6);
    }
    std::printf("{\"part\":\"persist\",\"pre\":%.4f,\"by_hundred\":[%s]}\n", pre, s.c_str());
    std::fflush(stdout);
  }
  // shapes: the language model's (BF16 out), the encoders' largest (FP32 out), at decode and prefill row counts
  struct Shape { const char* name; int n, k; bool f32; };
  const Shape shapes[] = {{"lm.in_proj", 6144, 2048, false}, {"lm.q_o", 2048, 2048, false}, {"lm.kv", 512, 2048, false},
                          {"lm.w1w3", 7680, 2048, false},    {"lm.w2", 2048, 7680, false},  {"a.lin_mid", 256, 512, true},
                          {"a.w1w3", 1792, 1536, true},      {"v.w1w3", 2560, 2048, true}};
  const int rows[] = {1, 8, 32, 128, 256, 1149};
  std::vector<std::pair<std::string, std::function<void()>>> work;
  for (const auto& s : shapes)
    for (int m : rows) work.push_back({std::string(s.name) + "@" + std::to_string(m), gemm(m, s.n, s.k, s.f32)});
  work.push_back({"stream_write_30m", [&] { fill<<<blocks(big / 2), 256>>>(spare, big / 2, 1e-3f); }});
  work.push_back({"stream_read_30m", [&] { sum<<<640, 256>>>(spare, big / 2, out); }});
  for (int r = 0; r < 2; ++r)
    for (const auto& [name, f] : work) {
      f();
      rest();
      const double fast = time6(f);
      induce();
      const double slow = time6(f);
      std::printf("{\"part\":\"shapes\",\"round\":%d,\"work\":\"%s\",\"fast\":%.4f,\"slow\":%.4f}\n", r + 1, name.c_str(), fast, slow);
      std::fflush(stdout);
    }
#ifdef LFM_GEMM_TCGEN05
  std::printf("{\"path\":\"tcgen05 build\",\"tcgen05_launches\":%llu}\n", lfm::gemm::tcgen05::launch_count());
#else
  std::printf("{\"path\":\"mma.sync build\"}\n");
#endif
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "alloc-extent-probe: %s\n", e.what());
  return 1;
}
