// lfm3-tcgen05-gemm-speed: validation/tcgen05-v2 (development evidence for amendment 2's dispatch condition). Each
// engine GEMM shape at each row count M >= 257 (where tcgen05 may run), timed with lfm::gemm::run, tcgen05 on and off
// (lfm::gemm::tcgen05::set_enabled), the median of 15 runs each after 3 untimed, by CUDA events. One JSON line per
// (shape, M): times, off / on, and whether tcgen05 launched. Normal data (the check tool's kind 1); FP32 out for the
// encoders' shapes (their x2 multiplies store FP32), BF16 for the language model's.
// Built with -DLFM_GEMM_TCGEN05 and linked with gemm_tcgen05.o, as lfm3-tcgen05-gemm-check.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "lfm/gemm.cuh"

#ifndef LFM_GEMM_TCGEN05
#error "build with -DLFM_GEMM_TCGEN05"
#endif

using bf16 = __nv_bfloat16;
namespace tc = lfm::gemm::tcgen05;

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::printf("{\"fatal\":\"%s\",\"error\":\"%s\"}\n", what, cudaGetErrorString(e));
    std::exit(3);
  }
}
__host__ __device__ inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
__global__ void gen(bf16* p, size_t n, uint64_t s) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = __float2bfloat16_rn(float(int64_t(mix64(s ^ mix64(i)) % 2001) - 1000) * 1e-3f);
}

struct Shape { const char* name; int n, k; bool f32; std::vector<int> ms; };

int main() {
  const std::vector<int> common{257, 384, 512, 1024, 1149, 2048, 4096};
  // the audio encoder's downsampler 1x1 convolutions (lin_mid, lin_end): W2 x 32 rows, from ~1,700 (a 2 s clip) to
  // ~72,000 (90 s)
  const std::vector<int> lin_rows{257, 512, 1024, 1700, 4096, 8192, 16384, 36000, 72000};
  const Shape shapes[] = {
      {"lm.in_proj", 6144, 2048, false, common}, {"lm.q_o", 2048, 2048, false, common},
      {"lm.kv", 512, 2048, false, common},       {"lm.w1w3", 7680, 2048, false, common},
      {"lm.w2", 2048, 7680, false, common},
      {"v.patchifier", 1024, 3072, true, common}, {"v.in_proj", 3072, 2048, true, common},
      {"v.q_out", 1024, 2048, true, common},      {"v.kv", 512, 2048, true, common},
      {"v.w1w3", 2560, 2048, true, common},       {"v.w2", 1024, 5120, true, common},
      {"v.proj1", 4096, 8192, true, common},      {"v.proj2", 2048, 8192, true, common},
      {"a.lin_mid", 256, 512, true, lin_rows},    {"a.attn", 768, 1536, true, common},
      {"a.conv_in", 1536, 1536, true, common},    {"a.w1w3", 1792, 1536, true, common},
      {"a.w2", 768, 3584, true, common},          {"a.lin_out", 768, 8192, true, common},
      {"a.proj1", 2048, 1536, true, common},      {"a.proj2", 2048, 4096, true, common}};
  cudaEvent_t e0, e1;
  check(cudaEventCreate(&e0), "event");
  check(cudaEventCreate(&e1), "event");
  for (const auto& s : shapes) {
    const int mmax = *std::max_element(s.ms.begin(), s.ms.end());
    void *a, *b, *c;
    check(cudaMalloc(&a, size_t(mmax) * s.k * 2), "malloc a");
    check(cudaMalloc(&b, size_t(s.n) * s.k * 2), "malloc b");
    check(cudaMalloc(&c, size_t(mmax) * s.n * 4), "malloc c");
    gen<<<640, 256>>>(static_cast<bf16*>(a), size_t(mmax) * s.k, 1);
    gen<<<640, 256>>>(static_cast<bf16*>(b), size_t(s.n) * s.k, 2);
    check(cudaDeviceSynchronize(), "gen");
    for (int m : s.ms) {
      auto p = lfm::gemm::linear(static_cast<bf16*>(a), static_cast<bf16*>(b), nullptr, m, s.k, s.n);
      if (s.f32) p.c_f32 = static_cast<float*>(c); else p.c = static_cast<bf16*>(c);
      double med[2];
      unsigned long long launched = 0;
      for (int on = 1; on >= 0; --on) {
        tc::set_enabled(on == 1);
        const unsigned long long before = tc::launch_count();
        for (int r = 0; r < 3; ++r) lfm::gemm::run(p);
        std::vector<float> t;
        for (int r = 0; r < 15; ++r) {
          check(cudaEventRecord(e0), "record");
          lfm::gemm::run(p);
          check(cudaEventRecord(e1), "record");
          check(cudaEventSynchronize(e1), "sync");
          float ms = 0;
          check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
          t.push_back(ms);
        }
        if (on) launched = tc::launch_count() - before;
        std::sort(t.begin(), t.end());
        med[on] = t[t.size() / 2];
      }
      tc::set_enabled(true);
      std::printf("{\"shape\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"out\":\"%s\",\"on_ms\":%.4f,\"off_ms\":%.4f,"
                  "\"off_over_on\":%.3f,\"tcgen05_launched\":%s}\n",
                  s.name, m, s.n, s.k, s.f32 ? "f32" : "bf16", med[1], med[0], med[0] / med[1], launched ? "true" : "false");
      std::fflush(stdout);
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }
  return 0;
}
