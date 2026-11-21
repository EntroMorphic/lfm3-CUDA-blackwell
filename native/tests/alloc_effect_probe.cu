// alloc-effect-probe: a diagnostic (validation/audio-ttft-v1): on Thor, the language model's large prefill GEMMs ran
// about 75% slower in a process that had made no cudaMalloc/cudaFree for a while, and one 2 MiB cudaMalloc + cudaFree
// before them restored the speed (build/keep-diag). This times, in one process on a quiet GPU:
//   - lfm::gemm at the slowed shape (M 1149, N 7680, K 2048, BF16 in, BF16 out: the prefill's grid 60 x 9);
//   - a 512 MiB device-to-device copy (bandwidth);
// first with no allocation call for IDLE seconds, then at delays after one 2 MiB cudaMalloc + cudaFree, then after
// other calls (cudaFree(nullptr), cudaMemGetInfo, cudaDeviceSynchronize), each after IDLE seconds without a call.
// The GPU is idle during the waits (the host sleeps). Each reading: the first and the median of 5 runs (events).
// Usage: alloc-effect-probe [IDLE_SECONDS]   (default 5)
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include <lfm/gemm.cuh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

namespace {
using bf16 = __nv_bfloat16;
__global__ void fill(bf16* p, size_t n, float scale) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2bfloat16_rn(scale * float(int(i % 1013) - 506));
}
void sleep_s(double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); }
}  // namespace

int main(int argc, char** argv) try {
  const double idle = argc > 1 ? std::atof(argv[1]) : 5.0;
  const int M = 1149, N = 7680, K = 2048;
  bf16 *a, *b, *c;
  cuda_check(cudaMalloc(&a, size_t(M) * K * 2));
  cuda_check(cudaMalloc(&b, size_t(N) * K * 2));
  cuda_check(cudaMalloc(&c, size_t(M) * N * 2));
  fill<<<int((size_t(M) * K + 255) / 256), 256>>>(a, size_t(M) * K, 1e-3f);
  fill<<<int((size_t(N) * K + 255) / 256), 256>>>(b, size_t(N) * K, 1e-3f);
  const size_t bytes = size_t(512) << 20;
  void *src, *dst;
  cuda_check(cudaMalloc(&src, bytes));
  cuda_check(cudaMalloc(&dst, bytes));
  cuda_check(cudaMemset(src, 1, bytes));
  cuda_check(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cuda_check(cudaEventCreate(&e0));
  cuda_check(cudaEventCreate(&e1));
  auto timed = [&](const std::function<void()>& f) {
    std::vector<float> t;
    for (int r = 0; r < 5; ++r) {
      cuda_check(cudaEventRecord(e0));
      f();
      cuda_check(cudaEventRecord(e1));
      cuda_check(cudaEventSynchronize(e1));
      float ms = 0;
      cuda_check(cudaEventElapsedTime(&ms, e0, e1));
      t.push_back(ms);
    }
    const float first = t[0];
    std::sort(t.begin(), t.end());
    return std::make_pair(first, t[2]);
  };
  lfm::gemm::Problem p = lfm::gemm::linear(a, b, c, M, K, N);
  auto gemm = [&] { lfm::gemm::run(p); };
  auto copy = [&] { cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice)); };
  timed(gemm);  // first launches out of the way
  timed(copy);
  auto reading = [&](const char* label) {
    const auto g = timed(gemm), m = timed(copy);
    std::printf("%-40s gemm first %7.3f median %7.3f ms | copy first %7.3f median %7.3f ms (%6.1f GB/s)\n", label, g.first,
                g.second, m.first, m.second, 2.0 * bytes / (m.second * 1e-3) / 1e9);
    std::fflush(stdout);
  };
  auto touch = [] {
    void* t = nullptr;
    cuda_check(cudaMalloc(&t, size_t(2) << 20));
    cuda_check(cudaFree(t));
  };
  for (int round = 0; round < 2; ++round) {
    std::printf("-- round %d (idle %.1f s before each reading without a call)\n", round + 1, idle);
    sleep_s(idle);
    reading("no call for idle s");
    for (double d : {0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0}) {
      sleep_s(idle);
      touch();
      sleep_s(d);
      char label[64];
      std::snprintf(label, sizeof label, "2 MiB malloc+free, then %.1f s", d);
      reading(label);
    }
    sleep_s(idle);
    cuda_check(cudaFree(nullptr));
    reading("cudaFree(nullptr), then 0 s");
    sleep_s(idle);
    size_t fr = 0, tot = 0;
    cuda_check(cudaMemGetInfo(&fr, &tot));
    reading("cudaMemGetInfo, then 0 s");
    sleep_s(idle);
    cuda_check(cudaDeviceSynchronize());
    reading("cudaDeviceSynchronize, then 0 s");
    sleep_s(idle);
    void* keep = nullptr;
    cuda_check(cudaMalloc(&keep, size_t(2) << 20));
    reading("2 MiB malloc (kept), then 0 s");
    sleep_s(idle);
    cuda_check(cudaFree(keep));
    reading("free of it, then 0 s");
    sleep_s(idle);
    reading("no call for idle s (again)");
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "alloc-effect-probe: %s\n", e.what());
  return 1;
}
