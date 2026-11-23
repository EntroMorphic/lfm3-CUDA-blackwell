// alloc-state-probe: validation/alloc-effect-v1 stage 2 (a diagnostic). Stage 1 found the slowed prefill GEMM (M 1149,
// N 7680, K 2048, grid 60 x 9) fast (~0.85 ms) right after its operands were written and slow (1.6-1.9 ms) after a
// few hundred launches, in every process. Which preceding action restores the fast state? On one fixed operand pair:
//   sustained: 300 launches, each timed (the slide from fast to slow);
//   then ROUNDS rounds; each condition in a rotated order, each preceded by 30 settling launches (pre: the median of
//   the last 5) and followed by 6 launches, each timed:
//     none           nothing
//     fill_b         a kernel rewrites the weight (the same values)
//     fill_a         a kernel rewrites the activation (the same values)
//     fill_other     a kernel writes an unrelated 30 MiB buffer
//     copy_b         the weight copied to a spare buffer and back (copy engine; the same values)
//     read_b         a kernel reads the whole weight (a sum, no write)
//     idle_5ms, idle_100ms, idle_1000ms   the host sleeps, the GPU idle
//     alloc          a 2 MiB cudaMalloc + cudaFree
// One JSON line for the sustained series and one per (round, condition).
// Built twice: plain (lfm::gemm's mma.sync path) and with -DLFM_GEMM_TCGEN05 + gemm_tcgen05.o (the deployed path).
// Usage: alloc-state-probe [ROUNDS]   (default 5)
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include <lfm/gemm.cuh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <utility>
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
  if (s == 12345.f) *out = s;  // keeps the reads; never true for these values in practice
}
void sleep_s(double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); }
}  // namespace

int main(int argc, char** argv) try {
  const int rounds = argc > 1 ? std::atoi(argv[1]) : 5;
  const int M = 1149, N = 7680, K = 2048;
  const size_t a_n = size_t(M) * K, b_n = size_t(N) * K;
  bf16 *a, *b, *c, *other, *spare;
  float* out;
  cuda_check(cudaMalloc(&a, a_n * 2));
  cuda_check(cudaMalloc(&b, b_n * 2));
  cuda_check(cudaMalloc(&c, size_t(M) * N * 2));
  cuda_check(cudaMalloc(&other, b_n * 2));
  cuda_check(cudaMalloc(&spare, b_n * 2));
  cuda_check(cudaMalloc(&out, 4));
  const auto blocks = [](size_t n) { return int((n + 255) / 256); };
  fill<<<blocks(a_n), 256>>>(a, a_n, 1e-3f);
  fill<<<blocks(b_n), 256>>>(b, b_n, 1e-3f);
  fill<<<blocks(b_n), 256>>>(other, b_n, 1e-3f);
  cuda_check(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cuda_check(cudaEventCreate(&e0));
  cuda_check(cudaEventCreate(&e1));
  const lfm::gemm::Problem p = lfm::gemm::linear(a, b, c, M, K, N);
  auto launches = [&](int n) {
    std::vector<float> t;
    for (int r = 0; r < n; ++r) {
      cuda_check(cudaEventRecord(e0));
      lfm::gemm::run(p);
      cuda_check(cudaEventRecord(e1));
      cuda_check(cudaEventSynchronize(e1));
      float ms = 0;
      cuda_check(cudaEventElapsedTime(&ms, e0, e1));
      t.push_back(ms);
    }
    return t;
  };
  auto join = [](const std::vector<float>& t) {
    std::string s;
    for (float x : t) s += (s.empty() ? "" : ",") + std::to_string(x).substr(0, 6);
    return s;
  };
  const std::vector<std::pair<std::string, std::function<void()>>> conditions = {
      {"none", [] {}},
      {"fill_b", [&] { fill<<<blocks(b_n), 256>>>(b, b_n, 1e-3f); }},
      {"fill_a", [&] { fill<<<blocks(a_n), 256>>>(a, a_n, 1e-3f); }},
      {"fill_other", [&] { fill<<<blocks(b_n), 256>>>(other, b_n, 1e-3f); }},
      {"copy_b", [&] {
         cuda_check(cudaMemcpyAsync(spare, b, b_n * 2, cudaMemcpyDeviceToDevice));
         cuda_check(cudaMemcpyAsync(b, spare, b_n * 2, cudaMemcpyDeviceToDevice));
       }},
      {"read_b", [&] { sum<<<640, 256>>>(b, b_n, out); }},
      {"idle_5ms", [&] { cuda_check(cudaDeviceSynchronize()); sleep_s(0.005); }},
      {"idle_100ms", [&] { cuda_check(cudaDeviceSynchronize()); sleep_s(0.1); }},
      {"idle_1000ms", [&] { cuda_check(cudaDeviceSynchronize()); sleep_s(1.0); }},
      {"alloc", [] {
         void* t = nullptr;
         cuda_check(cudaMalloc(&t, size_t(2) << 20));
         cuda_check(cudaFree(t));
       }},
  };
  launches(1);  // the module load out of the way
  fill<<<blocks(a_n), 256>>>(a, a_n, 1e-3f);
  fill<<<blocks(b_n), 256>>>(b, b_n, 1e-3f);
  cuda_check(cudaDeviceSynchronize());
#ifdef LFM_GEMM_TCGEN05
  const unsigned long long before = lfm::gemm::tcgen05::launch_count();
#endif
  std::printf("{\"sustained\":[%s]}\n", join(launches(300)).c_str());
#ifdef LFM_GEMM_TCGEN05
  std::printf("{\"path\":\"tcgen05\",\"tcgen05_launches\":%llu}\n", lfm::gemm::tcgen05::launch_count() - before);
#else
  std::printf("{\"path\":\"mma.sync\"}\n");
#endif
  std::fflush(stdout);
  const int nc = int(conditions.size());
  for (int r = 0; r < rounds; ++r)
    for (int k = 0; k < nc; ++k) {
      const auto& [name, act] = conditions[size_t((k + 3 * r) % nc)];
      auto settle = launches(30);
      std::vector<float> last(settle.end() - 5, settle.end());
      std::sort(last.begin(), last.end());
      act();
      cuda_check(cudaDeviceSynchronize());
      const auto t = launches(6);
      std::vector<float> s = t;
      std::sort(s.begin(), s.end());
      std::printf("{\"round\":%d,\"condition\":\"%s\",\"pre\":%.4f,\"first\":%.4f,\"median6\":%.4f,\"t\":[%s]}\n", r + 1,
                  name.c_str(), last[2], t[0], (s[2] + s[3]) / 2, join(t).c_str());
      std::fflush(stdout);
    }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "alloc-state-probe: %s\n", e.what());
  return 1;
}
