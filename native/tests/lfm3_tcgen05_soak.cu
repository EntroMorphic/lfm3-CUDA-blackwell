// validation/tcgen05-gemm-v1 rule 4a, the synthetic tensor-memory soak: 4 host threads, each on its own stream, launch
// a deterministic mix of lfm::gemm problems for DURATION seconds: tcgen05 GEMMs (V3, M >= 257), mma.sync GEMMs (M <=
// 256, same shapes family) and the engine's attention products (batched Q·Kᵀ with FP32 scores, P·V with B stored
// K x N), so tcgen05 kernels holding tensor memory meet each other and the mma.sync kernels on the SMs.
// Pass: every output bitwise equal to the same problem run alone; no launch error or trap; every call's latency (CUDA
// events around it on its stream) at most 10x the problem's solo median. All problems take ~1 ms alone, so
// queueing behind the 3 other streams stays well inside 10x; a stall (a blocked tensor-memory allocation) would not.
// Built with -DLFM_GEMM_TCGEN05 and gemm_tcgen05.o. Needs a quiet Thor.
//   lfm3-tcgen05-soak [DURATION_SECONDS=1200]   progress lines and a final JSON summary on stdout; exit 1 on failure
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "lfm/gemm.cuh"

#ifndef LFM_GEMM_TCGEN05
#error "build with -DLFM_GEMM_TCGEN05"
#endif

using bf16 = __nv_bfloat16;
static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::printf("{\"fatal\":\"%s\",\"error\":\"%s\"}\n", what, cudaGetErrorString(e));
    std::fflush(stdout);
    std::exit(3);
  }
}
__host__ __device__ inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
__global__ void gen(bf16* p, size_t n, uint64_t s, float scale) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = __float2bfloat16_rn((float(mix64(s ^ mix64(i)) >> 40) * 0x1p-24f - 0.5f) * scale);
}
// Counts 32-bit words differing between a and b.
__global__ void differ(const uint32_t* a, const uint32_t* b, size_t words, unsigned long long* count) {
  unsigned long long local = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x)
    local += a[i] != b[i];
  if (local) atomicAdd(count, local);
}

struct Case {
  std::string name;
  bool tcgen05 = false;          // expected to take the tcgen05 path
  lfm::gemm::Problem p;          // a, b set; output pointer set per run
  size_t out_bytes = 0;
  bool f32 = false;
  bf16 *a = nullptr, *b = nullptr;
  void* ref = nullptr;           // solo output
  float solo_ms = 0;
};

static Case linear_case(const char* name, int m, int n, int k, bool f32, uint64_t seed) {
  Case c;
  c.name = name;
  c.f32 = f32;
  check(cudaMalloc(&c.a, size_t(m) * k * 2), "malloc");
  check(cudaMalloc(&c.b, size_t(n) * k * 2), "malloc");
  gen<<<640, 256>>>(c.a, size_t(m) * k, seed, 2.f);
  gen<<<640, 256>>>(c.b, size_t(n) * k, seed + 1, 0.1f);
  c.p = lfm::gemm::linear(c.a, c.b, nullptr, m, k, n);
  c.out_bytes = size_t(m) * n * (f32 ? 4 : 2);
  c.tcgen05 = lfm::gemm::tcgen05::eligible([&] { auto q = c.p; if (f32) q.c_f32 = reinterpret_cast<float*>(256); else q.c = reinterpret_cast<bf16*>(256); return q; }());
  return c;
}
// Attention as the engine issues it (lfm3_main.cu head_scores / head_values): 16 query heads over 4 KV heads, dim 128.
static Case scores_case(int rows, int keys, uint64_t seed) {
  Case c;
  c.name = "attn.q_kT." + std::to_string(rows) + "x" + std::to_string(keys);
  c.f32 = true;
  const int hidden = 2048, dim = 128, heads = 16, kv = 4;
  check(cudaMalloc(&c.a, size_t(rows) * hidden * 2), "malloc");
  check(cudaMalloc(&c.b, size_t(kv) * keys * dim * 2), "malloc");
  gen<<<640, 256>>>(c.a, size_t(rows) * hidden, seed, 2.f);
  gen<<<640, 256>>>(c.b, size_t(kv) * keys * dim, seed + 1, 2.f);
  auto& p = c.p;
  p.a = c.a; p.b = c.b; p.m = rows; p.n = keys; p.k = dim;
  p.lda = hidden; p.ldb = dim; p.ldc = keys;
  p.stride_a = dim; p.stride_b = (long long)keys * dim; p.stride_c = (long long)rows * keys;
  p.batch = heads; p.group = heads / kv;
  c.out_bytes = size_t(heads) * rows * keys * 4;
  return c;
}
static Case values_case(int rows, int keys, uint64_t seed) {
  Case c;
  c.name = "attn.p_v." + std::to_string(rows) + "x" + std::to_string(keys);
  const int hidden = 2048, dim = 128, heads = 16, kv = 4;
  check(cudaMalloc(&c.a, size_t(heads) * rows * keys * 2), "malloc");
  check(cudaMalloc(&c.b, size_t(kv) * keys * dim * 2), "malloc");
  gen<<<640, 256>>>(c.a, size_t(heads) * rows * keys, seed, 0.01f);
  gen<<<640, 256>>>(c.b, size_t(kv) * keys * dim, seed + 1, 2.f);
  auto& p = c.p;
  p.a = c.a; p.b = c.b; p.m = rows; p.n = dim; p.k = keys; p.b_rows = keys; p.b_k_contiguous = false;
  p.lda = keys; p.ldb = dim; p.ldc = hidden;
  p.stride_a = (long long)rows * keys; p.stride_b = (long long)keys * dim; p.stride_c = dim;
  p.batch = heads; p.group = heads / kv;
  c.out_bytes = size_t(rows) * hidden * 2;
  return c;
}
static void set_out(lfm::gemm::Problem& p, const Case& c, void* out) {
  if (c.f32) { p.c_f32 = static_cast<float*>(out); p.c = nullptr; }
  else { p.c = static_cast<bf16*>(out); p.c_f32 = nullptr; }
}

int main(int argc, char** argv) try {
  const double duration = argc > 1 ? std::stod(argv[1]) : 1200.0;
  std::vector<Case> cases;
  // Every problem sized to ~1 ms alone, so queueing behind the 3 other streams stays a few times a call's solo time.
  // tcgen05 (V3): LM and vision projections, both output types
  cases.push_back(linear_case("lm.w1w3.m4096", 4096, 7680, 2048, false, 1));
  cases.push_back(linear_case("lm.w2.m3072", 3072, 2048, 7680, false, 2));
  cases.push_back(linear_case("v.w2.m8192", 8192, 1024, 5120, true, 3));
  cases.push_back(linear_case("v.in_proj.m8192", 8192, 3072, 2048, true, 4));
  // mma.sync (M <= 256: outside the tcgen05 dispatch)
  cases.push_back(linear_case("mma.w1w3.m256", 256, 7680, 7680, false, 6));
  cases.push_back(linear_case("mma.in_proj.m256", 256, 6144, 8192, true, 7));
  // the engine's attention products (never tcgen05)
  cases.push_back(scores_case(1024, 3072, 8));
  cases.push_back(values_case(1024, 8192, 9));
  const size_t workspace = lfm::gemm::workspace_bytes(1024, 128, 4096, 16);
  (void)workspace;  // P·V with M > 8 takes the wide path, no split-K workspace
  check(cudaDeviceSynchronize(), "generate");

  // Solo: each problem alone, its reference output and median latency (1 warm + 9 timed).
  cudaEvent_t e0, e1;
  check(cudaEventCreate(&e0), "event"); check(cudaEventCreate(&e1), "event");
  for (auto& c : cases) {
    check(cudaMalloc(&c.ref, c.out_bytes), "malloc");
    check(cudaMemset(c.ref, 0, c.out_bytes), "memset");
    auto p = c.p;
    set_out(p, c, c.ref);
    const unsigned long long before = lfm::gemm::tcgen05::launch_count();
    lfm::gemm::run(p);
    check(cudaDeviceSynchronize(), "solo warm");
    if ((lfm::gemm::tcgen05::launch_count() > before) != c.tcgen05) {
      std::printf("{\"fatal\":\"dispatch\",\"case\":\"%s\",\"expected_tcgen05\":%s}\n", c.name.c_str(), c.tcgen05 ? "true" : "false");
      return 1;
    }
    std::vector<float> t;
    for (int i = 0; i < 9; ++i) {
      check(cudaEventRecord(e0), "record");
      lfm::gemm::run(p);
      check(cudaEventRecord(e1), "record");
      check(cudaEventSynchronize(e1), "solo");
      float ms; check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
      t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    c.solo_ms = t[4];
    std::printf("{\"solo\":\"%s\",\"tcgen05\":%s,\"median_ms\":%.4f}\n", c.name.c_str(), c.tcgen05 ? "true" : "false", c.solo_ms);
  }
  std::fflush(stdout);

  // Soak: 4 threads x own stream x own output per case; a deterministic per-thread sequence over the cases.
  constexpr int kThreads = 4;
  std::atomic<bool> stop{false};
  std::atomic<unsigned long long> calls{0}, tc_calls{0}, mismatched{0}, slow{0};
  std::mutex mu;
  double worst_ratio = 0;
  std::string worst_case;
  std::vector<std::vector<float>> lat(cases.size());  // per case, soak latencies (information: max / soak median)
  unsigned long long* d_count = nullptr;
  check(cudaMalloc(&d_count, kThreads * sizeof(unsigned long long)), "malloc");
  const auto t_start = std::chrono::steady_clock::now();
  auto worker = [&](int tid) {
    cudaStream_t s;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event"); check(cudaEventCreate(&b), "event");
    std::vector<void*> outs(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) check(cudaMalloc(&outs[i], cases[i].out_bytes), "malloc");
    unsigned long long* count = d_count + tid;
    uint64_t r = mix64(uint64_t(tid) + 77);
    while (!stop.load()) {
      r = mix64(r);
      const size_t i = r % cases.size();
      const Case& c = cases[i];
      auto p = c.p;
      set_out(p, c, outs[i]);
      check(cudaMemsetAsync(outs[i], 0xff, c.out_bytes, s), "memset");
      check(cudaEventRecord(a, s), "record");
      lfm::gemm::run(p, s);
      check(cudaEventRecord(b, s), "record");
      check(cudaMemsetAsync(count, 0, sizeof(unsigned long long), s), "memset");
      differ<<<160, 256, 0, s>>>(static_cast<const uint32_t*>(outs[i]), static_cast<const uint32_t*>(c.ref), c.out_bytes / 4, count);
      unsigned long long h = 0;
      check(cudaMemcpyAsync(&h, count, sizeof h, cudaMemcpyDeviceToHost, s), "copy");
      check(cudaStreamSynchronize(s), "soak call");
      float ms; check(cudaEventElapsedTime(&ms, a, b), "elapsed");
      const double ratio = ms / c.solo_ms;
      ++calls;
      if (c.tcgen05) ++tc_calls;
      if (h) { ++mismatched; std::printf("{\"mismatch\":\"%s\",\"thread\":%d,\"words\":%llu}\n", c.name.c_str(), tid, h); std::fflush(stdout); }
      if (ratio > 10) { ++slow; std::printf("{\"slow\":\"%s\",\"thread\":%d,\"ms\":%.3f,\"solo_ms\":%.3f}\n", c.name.c_str(), tid, ms, c.solo_ms); std::fflush(stdout); }
      std::lock_guard<std::mutex> lock(mu);
      if (ratio > worst_ratio) { worst_ratio = ratio; worst_case = c.name; }
      if (lat[i].size() < 200000) lat[i].push_back(ms);
    }
    for (void* o : outs) cudaFree(o);
    cudaStreamDestroy(s);
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < kThreads; ++t) pool.emplace_back(worker, t);
  double last = 0;
  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    if (el - last >= 60 || el >= duration) {
      last = el;
      std::lock_guard<std::mutex> lock(mu);
      std::printf("{\"progress_s\":%.0f,\"calls\":%llu,\"tcgen05_calls\":%llu,\"mismatched\":%llu,\"slow\":%llu,\"worst_ratio\":%.2f,\"worst_case\":\"%s\"}\n",
                  el, calls.load(), tc_calls.load(), mismatched.load(), slow.load(), worst_ratio, worst_case.c_str());
      std::fflush(stdout);
    }
    if (el >= duration) break;
  }
  stop = true;
  for (auto& t : pool) t.join();
  check(cudaDeviceSynchronize(), "end");
  for (size_t i = 0; i < cases.size(); ++i) {
    auto v = lat[i];
    if (v.empty()) continue;
    std::sort(v.begin(), v.end());
    std::printf("{\"case\":\"%s\",\"calls\":%zu,\"solo_ms\":%.4f,\"soak_median_ms\":%.4f,\"soak_p99_ms\":%.4f,\"soak_max_ms\":%.4f,"
                "\"max_over_solo\":%.2f,\"max_over_soak_median\":%.2f}\n",
                cases[i].name.c_str(), v.size(), cases[i].solo_ms, v[v.size() / 2], v[v.size() * 99 / 100], v.back(),
                v.back() / cases[i].solo_ms, v.back() / v[v.size() / 2]);
  }
  const bool pass = mismatched == 0 && slow == 0 && calls > 0;
  std::printf("{\"summary\":\"tcgen05-soak\",\"seconds\":%.0f,\"threads\":%d,\"calls\":%llu,\"tcgen05_calls\":%llu,\"mismatched\":%llu,"
              "\"slow_over_10x\":%llu,\"worst_ratio\":%.2f,\"worst_case\":\"%s\",\"pass\":%s}\n",
              duration, kThreads, calls.load(), tc_calls.load(), mismatched.load(), slow.load(), worst_ratio, worst_case.c_str(),
              pass ? "true" : "false");
  return pass ? 0 : 1;
} catch (const std::exception& e) {
  std::printf("{\"fatal\":\"exception\",\"error\":\"%s\"}\n", e.what());
  return 2;
}
