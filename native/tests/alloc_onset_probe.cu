// alloc-onset-probe: validation/alloc-effect-v1 stage 3 (a diagnostic). Stage 1 made the slowed prefill GEMM (M 1149,
// N 7680, K 2048, grid 60 x 9) slow on a fixed operand pair within 8 per-buffer steps (allocate a 30 MiB buffer and
// keep it, upload 30 MiB from pageable host memory into it, a chase kernel, a fill kernel, GEMMs on it); stage 2, with
// none of those steps, never saw it slow. Which step brings it on? One INGREDIENT per process:
//   none         nothing
//   malloc       cudaMalloc 30 MiB, kept, untouched
//   malloc_fill  cudaMalloc 30 MiB, kept, written by a kernel
//   h2d          30 MiB from pageable host memory into a spare allocated at the start
//   h2d_pinned   the same from cudaMallocHost memory
//   malloc_h2d   cudaMalloc 30 MiB, kept, and the pageable upload into it
//   gemm_other   6 GEMMs on one of 8 other weight buffers (allocated and written at the start)
//   stage1       stage 1's whole step (allocate and keep, pageable upload, chase, fill, 6 GEMMs on it)
// The process: 30 launches of the fixed pair (baseline); 8 steps, each followed by 6 timed launches of the fixed pair;
// 30 launches (after); then, each followed by 6 timed launches: idle_1000ms, fill_b, alloc (2 MiB malloc + free),
// free_kept (cudaFree of every buffer the steps kept), idle_1000ms again. One JSON line.
// Usage: alloc-onset-probe INGREDIENT
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include <lfm/gemm.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
using bf16 = __nv_bfloat16;
__global__ void fill(bf16* p, size_t n, float scale) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2bfloat16_rn(scale * float(int(i % 1013) - 506));
}
__global__ void chase(const uint32_t* p, uint32_t start, int hops, uint32_t* out) {
  uint32_t x = start;
  for (int h = 0; h < hops; ++h) x = p[x];
  *out = x;
}
double median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2) throw std::runtime_error("usage: alloc-onset-probe INGREDIENT");
  const std::string what = argv[1];
  const std::vector<std::string> known = {"none", "malloc", "malloc_fill", "h2d", "h2d_pinned", "malloc_h2d", "gemm_other", "stage1"};
  if (std::find(known.begin(), known.end(), what) == known.end()) throw std::runtime_error("unknown ingredient");
  const int M = 1149, N = 7680, K = 2048;
  const size_t a_n = size_t(M) * K, b_n = size_t(N) * K, bytes = b_n * 2;
  const auto blocks = [](size_t n) { return int((n + 255) / 256); };
  bf16 *a, *b, *c, *spare;
  uint32_t* sink;
  cuda_check(cudaMalloc(&a, a_n * 2));
  cuda_check(cudaMalloc(&b, bytes));
  cuda_check(cudaMalloc(&c, size_t(M) * N * 2));
  cuda_check(cudaMalloc(&spare, bytes));
  cuda_check(cudaMalloc(&sink, 4));
  std::vector<bf16*> others;
  if (what == "gemm_other")
    for (int i = 0; i < 8; ++i) {
      bf16* o;
      cuda_check(cudaMalloc(&o, bytes));
      fill<<<blocks(b_n), 256>>>(o, b_n, 1e-3f);
      others.push_back(o);
    }
  std::vector<uint32_t> host(bytes / 4, 0);
  void* pinned = nullptr;
  if (what == "h2d_pinned") {
    cuda_check(cudaMallocHost(&pinned, bytes));
    std::memset(pinned, 0, bytes);
  }
  fill<<<blocks(a_n), 256>>>(a, a_n, 1e-3f);
  fill<<<blocks(b_n), 256>>>(b, b_n, 1e-3f);
  cuda_check(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cuda_check(cudaEventCreate(&e0));
  cuda_check(cudaEventCreate(&e1));
  auto launches = [&](bf16* w, int n) {
    const lfm::gemm::Problem p = lfm::gemm::linear(a, w, c, M, K, N);
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
  // the chase table: one uint32 per 64 KiB region in a simple stride cycle (stage 1 shuffled it; the pattern of the
  // upload's bytes is not the question here)
  const size_t region_words = (size_t(64) << 10) / 4, regions = bytes / (size_t(64) << 10);
  for (size_t r = 0; r < regions; ++r) host[r * region_words] = uint32_t(((r + 7) % regions) * region_words);
  launches(b, 1);  // the module load out of the way
  const double baseline = median(launches(b, 30));
  std::vector<bf16*> kept;
  std::string steps;
  for (int s = 0; s < 8; ++s) {
    if (what == "malloc" || what == "malloc_fill" || what == "malloc_h2d" || what == "stage1") {
      bf16* k;
      cuda_check(cudaMalloc(&k, bytes));
      kept.push_back(k);
    }
    if (what == "malloc_fill") fill<<<blocks(b_n), 256>>>(kept.back(), b_n, 1e-3f);
    if (what == "h2d") cuda_check(cudaMemcpy(spare, host.data(), bytes, cudaMemcpyHostToDevice));
    if (what == "h2d_pinned") cuda_check(cudaMemcpy(spare, pinned, bytes, cudaMemcpyHostToDevice));
    if (what == "malloc_h2d" || what == "stage1") cuda_check(cudaMemcpy(kept.back(), host.data(), bytes, cudaMemcpyHostToDevice));
    if (what == "stage1") {
      chase<<<1, 1>>>(reinterpret_cast<uint32_t*>(kept.back()), 0, int(regions) * 4, sink);
      fill<<<blocks(b_n), 256>>>(kept.back(), b_n, 1e-3f);
      launches(kept.back(), 6);
    }
    if (what == "gemm_other") launches(others[size_t(s)], 6);
    cuda_check(cudaDeviceSynchronize());
    steps += (steps.empty() ? "" : ",") + std::to_string(median(launches(b, 6))).substr(0, 6);
  }
  const double after = median(launches(b, 30));
  std::string restore;
  auto r6 = [&](const char* name) {
    cuda_check(cudaDeviceSynchronize());
    restore += std::string(restore.empty() ? "" : ",") + "\"" + name + "\":" + std::to_string(median(launches(b, 6))).substr(0, 6);
  };
  std::this_thread::sleep_for(std::chrono::seconds(1));
  r6("idle_1000ms");
  fill<<<blocks(b_n), 256>>>(b, b_n, 1e-3f);
  r6("fill_b");
  {
    void* t = nullptr;
    cuda_check(cudaMalloc(&t, size_t(2) << 20));
    cuda_check(cudaFree(t));
  }
  r6("alloc");
  for (auto* k : kept) cuda_check(cudaFree(k));
  r6("free_kept");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  r6("idle_1000ms_again");
  std::printf("{\"ingredient\":\"%s\",\"baseline\":%.4f,\"steps\":[%s],\"after\":%.4f,\"restore\":{%s}}\n", what.c_str(),
              baseline, steps.c_str(), after, restore.c_str());
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "alloc-onset-probe: %s\n", e.what());
  return 1;
}
