// validation/tcgen05-gemm-v1 rule 1: lfm::gemm::run with the tcgen05 path on against off, bitwise, on the engine's
// projection shapes (the probe's 13, the LM head) at the dispatch boundary and beyond, with integer, normal and split
// data, FP32 and BF16 out; and the problems outside the contract shown to take the old kernel (no tcgen05 launch).
// Built with -DLFM_GEMM_TCGEN05 and linked with gemm_tcgen05.o (tools/thor/build_tcgen05_engine.sh).
//   lfm3-tcgen05-gemm-check [--m 256,257,...] [--shapes a,b] [--data int,normal,split]   one JSON line per case
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
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
struct Device {
  void* p = nullptr;
  explicit Device(size_t bytes) { check(cudaMalloc(&p, bytes < 16 ? 16 : bytes), "cudaMalloc"); }
  ~Device() { cudaFree(p); }
  Device(const Device&) = delete;
  template <class T> T* as() const { return static_cast<T*>(p); }
};

__host__ __device__ inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
__device__ inline float unif(uint64_t s, uint64_t i) { return float(mix64(s ^ mix64(i)) >> 40) * 0x1p-24f; }
__device__ inline float gauss(uint64_t s, uint64_t i) {
  return sqrtf(-2.f * logf(unif(s, 2 * i) + 0x1p-25f)) * cospif(2.f * unif(s, 2 * i + 1));
}
__device__ inline float act(uint64_t s, int r, int c, int cols) {
  return gauss(s, uint64_t(r) * cols + c) * exp2f(unif(s + 1, r) * 8.f - 4.f) * exp2f(unif(s + 2, c) * 8.f - 4.f);
}
// kind 0 int [-4, 4]; 1 normal (A with row/column scales, B unit); 2 split: A = [hi | lo], B = [W | W] (K = 2 Kin).
__global__ void gen(bf16* p, int rows, int cols, int kind, bool is_a, uint64_t s) {
  const size_t n = size_t(rows) * cols;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int r = int(i / cols), c = int(i % cols);
    if (kind == 0) { p[i] = __float2bfloat16_rn(float(int(mix64(s ^ mix64(i)) % 9) - 4)); continue; }
    if (kind == 1) { p[i] = __float2bfloat16_rn(is_a ? act(s, r, c, cols) : gauss(s, i)); continue; }
    const int kin = cols / 2, cc = c % kin;
    if (is_a) {
      const float x = act(s, r, cc, kin);
      const bf16 hi = __float2bfloat16_rn(x);
      p[i] = c < kin ? hi : __float2bfloat16_rn(x - __bfloat162float(hi));
    } else {
      p[i] = __float2bfloat16_rn(gauss(s, uint64_t(r) * kin + cc));
    }
  }
}

struct Shape { const char* name; int n, k; bool vision; };
static const Shape kShapes[] = {
    {"v.patchifier", 1024, 3072, true}, {"v.in_proj", 3072, 2048, true}, {"v.q_out", 1024, 2048, true},
    {"v.kv", 512, 2048, true},          {"v.w1w3", 2560, 2048, true},    {"v.w2", 1024, 5120, true},
    {"v.proj1", 4096, 8192, true},      {"v.proj2", 2048, 8192, true},   {"lm.in_proj", 6144, 2048, false},
    {"lm.q_o", 2048, 2048, false},      {"lm.kv", 512, 2048, false},     {"lm.w1w3", 7680, 2048, false},
    {"lm.w2", 2048, 7680, false},       {"lm.head", 128000, 2048, false}};

static std::vector<std::string> split_list(const std::string& s) {
  std::vector<std::string> v;
  size_t i = 0;
  while (i <= s.size()) {
    const size_t j = s.find(',', i);
    const std::string t = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
    if (!t.empty()) v.push_back(t);
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return v;
}

// Runs p twice, path on then off, into separately allocated outputs; returns differing bytes and tcgen05 launches.
struct Outcome { size_t bytes = 0, differing = 0; unsigned long long launches = 0; };
static Outcome on_off(lfm::gemm::Problem p, size_t out_elems, size_t elem_bytes) {
  Outcome o;
  o.bytes = out_elems * elem_bytes;
  Device c_on(o.bytes), c_off(o.bytes);
  check(cudaMemset(c_on.p, 0xff, o.bytes), "memset");
  check(cudaMemset(c_off.p, 0xff, o.bytes), "memset");  // the same fill: strided outputs leave gaps by design
  auto set = [&](void* c) { if (p.c_f32) p.c_f32 = static_cast<float*>(c); else p.c = static_cast<bf16*>(c); };
  const unsigned long long before = tc::launch_count();
  tc::set_enabled(true);
  set(c_on.p);
  lfm::gemm::run(p);
  check(cudaDeviceSynchronize(), "run (on)");
  o.launches = tc::launch_count() - before;
  tc::set_enabled(false);
  set(c_off.p);
  lfm::gemm::run(p);
  check(cudaDeviceSynchronize(), "run (off)");
  tc::set_enabled(true);
  std::vector<uint8_t> a(o.bytes), b(o.bytes);
  check(cudaMemcpy(a.data(), c_on.p, o.bytes, cudaMemcpyDeviceToHost), "download");
  check(cudaMemcpy(b.data(), c_off.p, o.bytes, cudaMemcpyDeviceToHost), "download");
  for (size_t i = 0; i < o.bytes; i += elem_bytes) o.differing += std::memcmp(&a[i], &b[i], elem_bytes) != 0;
  return o;
}

int main(int argc, char** argv) try {
  std::vector<int> ms{256, 257, 258, 4096, 32768};
  std::vector<std::string> shapes, data{"int", "normal", "split"};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
    if (a == "--m") { ms.clear(); for (auto& t : split_list(argv[++i])) ms.push_back(std::stoi(t)); }
    else if (a == "--shapes") shapes = split_list(argv[++i]);
    else if (a == "--data") data = split_list(argv[++i]);
    else throw std::runtime_error("unknown option " + a);
  }
  int failed = 0, cases = 0;
  auto report = [&](const std::string& what, const Outcome& o, unsigned long long want_launches) {
    const bool pass = o.differing == 0 && o.launches == want_launches;
    ++cases; failed += !pass;
    std::printf("%s,\"elements_differing\":%zu,\"tcgen05_launches\":%llu,\"expected_launches\":%llu,\"pass\":%s}\n",
                what.c_str(), o.differing, o.launches, want_launches, pass ? "true" : "false");
    std::fflush(stdout);
  };
  // Engine shapes: on against off.
  for (const auto& kind : data) {
    const int kid = kind == "int" ? 0 : kind == "normal" ? 1 : kind == "split" ? 2 : -1;
    if (kid < 0) throw std::runtime_error("unknown data " + kind);
    for (const auto& s : kShapes) {
      bool sel = shapes.empty();
      for (auto& n : shapes) sel |= n == s.name;
      if (!sel || (kid == 2 && !s.vision)) continue;
      for (int M : ms) {
        // The LM head's output at 32,768 rows is 16.8 GB in FP32 per buffer: not run (recorded in the README).
        if (s.n == 128000 && M > 4096) continue;
        Device a(size_t(M) * s.k * 2), b(size_t(s.n) * s.k * 2);
        const uint64_t seed = mix64(uint64_t(kid) * 1315423911u ^ mix64(std::hash<std::string>{}(s.name)) ^ uint64_t(M));
        gen<<<640, 256>>>(a.as<bf16>(), M, s.k, kid, true, seed);
        gen<<<640, 256>>>(b.as<bf16>(), s.n, s.k, kid, false, seed + 7);
        check(cudaDeviceSynchronize(), "generate");
        for (int out16 = 0; out16 < 2; ++out16) {
          if (s.n == 128000 && out16) continue;  // the engine's LM head stores FP32 only
          auto p = lfm::gemm::linear(a.as<bf16>(), b.as<bf16>(), nullptr, M, s.k, s.n);
          if (out16) p.c = reinterpret_cast<bf16*>(16); else p.c_f32 = reinterpret_cast<float*>(16);  // set in on_off
          const Outcome o = on_off(p, size_t(M) * s.n, out16 ? 2 : 4);
          char head[256];
          std::snprintf(head, sizeof head, "{\"case\":\"engine\",\"data\":\"%s\",\"shape\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"out\":\"%s\"",
                        kind.c_str(), s.name, M, s.n, s.k, out16 ? "bf16" : "f32");
          report(head, o, M >= 257 ? 1 : 0);
        }
      }
    }
  }
  // Outside the contract: must not launch tcgen05 (and so match trivially).
  {
    const int M = 4096, K = 2048;
    Device a(size_t(M) * K * 2), b(size_t(8192) * K * 2);
    gen<<<640, 256>>>(a.as<bf16>(), M, K, 1, true, 11);
    gen<<<640, 256>>>(b.as<bf16>(), 8192, K, 1, false, 12);
    check(cudaDeviceSynchronize(), "generate");
    auto g = lfm::gemm::linear(a.as<bf16>(), b.as<bf16>(), reinterpret_cast<bf16*>(16), M, K, 16);  // gate_proj N = 16
    report("{\"case\":\"refused\",\"what\":\"gate_proj n=16\"", on_off(g, size_t(M) * 16, 2), 0);
    auto l = lfm::gemm::linear(a.as<bf16>(), b.as<bf16>(), reinterpret_cast<bf16*>(16), M, K, 2048);
    l.ldc = 2048 + 8;  // not dense
    report("{\"case\":\"refused\",\"what\":\"ldc != n\"", on_off(l, size_t(M) * (2048 + 8), 2), 0);
    auto m = lfm::gemm::linear(a.as<bf16>(), b.as<bf16>(), reinterpret_cast<bf16*>(16), 256, K, 2048);
    report("{\"case\":\"refused\",\"what\":\"m=256\"", on_off(m, size_t(256) * 2048, 2), 0);
    // Attention Q·Kᵀ as head_scores issues it: 16 query heads over 4 KV heads, K = 128, FP32 scores.
    lfm::gemm::Problem q;
    q.a = a.as<bf16>(); q.b = b.as<bf16>(); q.c_f32 = reinterpret_cast<float*>(16);
    q.m = 512; q.n = 1024; q.k = 128; q.lda = 2048; q.ldb = 128; q.ldc = 1024;
    q.stride_a = 128; q.stride_b = 1024LL * 128; q.stride_c = 512LL * 1024; q.batch = 16; q.group = 4;
    report("{\"case\":\"refused\",\"what\":\"attention q.k batched\"", on_off(q, size_t(16) * 512 * 1024, 4), 0);
    // Attention P·V as head_values issues it: B stored K x N.
    lfm::gemm::Problem v;
    v.a = a.as<bf16>(); v.b = b.as<bf16>(); v.c = reinterpret_cast<bf16*>(16);
    v.m = 512; v.n = 128; v.k = 1024; v.b_rows = 1000; v.b_k_contiguous = false;
    v.lda = 1024; v.ldb = 128; v.ldc = 2048; v.stride_a = 512LL * 1024; v.stride_b = 1024LL * 128; v.stride_c = 128;
    v.batch = 16; v.group = 4;
    report("{\"case\":\"refused\",\"what\":\"attention p.v k x n\"", on_off(v, size_t(512) * 2048, 2), 0);
  }
  std::printf("{\"summary\":\"tcgen05-gemm-check\",\"cases\":%d,\"failed\":%d}\n", cases, failed);
  return failed ? 1 : 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 2;
}
