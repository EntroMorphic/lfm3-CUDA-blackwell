// v5 thin-GEMM candidate probe. Isolated experiment: no production integration,
// no acceptance implied. Compares v5 against the frozen v4 runtime kernels
// bit-for-bit and measures achieved weight-streaming bandwidth.
//
// v5 keeps v4's arithmetic exactly: one warp per 16 output rows, the same PTX
// m16n8k16 BF16 MMA with identical fragment contents, the same K order within
// each split-K partition, the same FP32 partial stores, and v4's unchanged
// wmma_finish reduction. Only operand fetching changes: U consecutive K steps
// are loaded (read-only path) before their MMAs run, so U steps of loads are in
// flight per warp instead of one.
#include V4_HEADER  // frozen v4 runtime header, passed by path at compile time
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace v5 {
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t ld_nc(const bf16* p) {
  uint32_t bits;
  asm("ld.global.nc.u32 %0, [%1];" : "=r"(bits) : "l"(p));
  return bits;
}

__device__ __forceinline__ void mma(float& c0, float& c1, float& c2, float& c3, uint32_t a0, uint32_t a1,
                                    uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int U>
__global__ void mma_thin_partials(const bf16* w, const bf16* x, float* partial, int m, int n, int k, int padded_n,
                                  int padded_m, int parts) {
  int lane = int(threadIdx.x) & 31, warp = int(threadIdx.x) / 32;
  int row = (int(blockIdx.x) * (int(blockDim.x) / 32) + warp) * 16;
  if (row >= m) return;
  int group = lane / 4, t = lane % 4, start = int(blockIdx.z) * (k / parts), end = start + k / parts;
  const bool r0 = row + group < m, r1 = row + group + 8 < m, bx = group < n;
  // Base pointers are only dereferenced under their row/column predicate.
  const bf16* w0 = w + size_t(r0 ? row + group : 0) * k + 2 * t;
  const bf16* w1 = w + size_t(r1 ? row + group + 8 : 0) * k + 2 * t;
  const bf16* xb = x + size_t(bx ? group : 0) * k + 2 * t;
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  int base = start;
  for (; base + 16 * U <= end; base += 16 * U) {
    uint32_t a0[U], a1[U], a2[U], a3[U], b0[U], b1[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      int o = base + 16 * u;
      a0[u] = r0 ? ld_nc(w0 + o) : 0u;
      a1[u] = r1 ? ld_nc(w1 + o) : 0u;
      a2[u] = r0 ? ld_nc(w0 + o + 8) : 0u;
      a3[u] = r1 ? ld_nc(w1 + o + 8) : 0u;
      b0[u] = bx ? ld_nc(xb + o) : 0u;
      b1[u] = bx ? ld_nc(xb + o + 8) : 0u;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) mma(c0, c1, c2, c3, a0[u], a1[u], a2[u], a3[u], b0[u], b1[u]);
  }
  for (; base < end; base += 16) {  // remainder steps, same order
    uint32_t a0 = r0 ? ld_nc(w0 + base) : 0u, a1 = r1 ? ld_nc(w1 + base) : 0u;
    uint32_t a2 = r0 ? ld_nc(w0 + base + 8) : 0u, a3 = r1 ? ld_nc(w1 + base + 8) : 0u;
    uint32_t b0 = bx ? ld_nc(xb + base) : 0u, b1 = bx ? ld_nc(xb + base + 8) : 0u;
    mma(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
  }
  size_t head = size_t(blockIdx.z) * padded_m * padded_n;
  if (r0) { size_t o = head + size_t(row + group) * padded_n + 2 * t; partial[o] = c0; partial[o + 1] = c1; }
  if (r1) { size_t o = head + size_t(row + group + 8) * padded_n + 2 * t; partial[o] = c2; partial[o + 1] = c3; }
}
}  // namespace v5

// v6: same arithmetic as v4/v5; weight operands are staged through shared memory so
// global reads are long contiguous runs. Each warp copies its 16 rows x (16*S) K
// elements with 16-byte vector loads (row-contiguous), into a padded, double-buffered
// per-warp tile, then builds byte-identical m16n8k16 A fragments from it. Ordinary
// loads/stores only (no TMA, no cp.async); synchronization is warp-scoped.
namespace v6 {
using bf16 = __nv_bfloat16;
constexpr int kWarps = 4;

__device__ __forceinline__ uint4 ld_nc_v4(const bf16* p) {
  uint4 v;
  asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

template <int S>
__global__ void __launch_bounds__(32 * kWarps) mma_thin_partials(const bf16* w, const bf16* x, float* partial, int m,
                                                                  int n, int k, int padded_n, int padded_m, int parts) {
  constexpr int kStride = 16 * S + 8;  // bf16 elements per smem row; +16 B pad keeps fragment reads conflict-free
  __shared__ __align__(16) bf16 tile[kWarps][2][16 * kStride];
  int lane = int(threadIdx.x) & 31, warp = int(threadIdx.x) / 32;
  int row = (int(blockIdx.x) * (int(blockDim.x) / 32) + warp) * 16;
  if (row >= m) return;  // whole warp retires; only warp-scoped syncs follow
  int group = lane / 4, t = lane % 4, start = int(blockIdx.z) * (k / parts), end = start + k / parts;
  const bool bx = group < n;
  const bf16* xb = x + size_t(bx ? group : 0) * k + 2 * t;
  int rows_valid = min(16, m - row);
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  // A stage covers cnt <= S K-steps: 16 rows x 2*cnt 16-byte vectors = 32*cnt vectors, cnt per lane.
  uint4 staged[S];
  auto fetch = [&](int base, int cnt) {
#pragma unroll
    for (int i = 0; i < S; ++i) {
      if (i < cnt) {
        int v = lane + 32 * i, r = v / (2 * cnt), c = v % (2 * cnt);
        staged[i] = r < rows_valid ? ld_nc_v4(w + size_t(row + r) * k + base + 8 * c) : make_uint4(0, 0, 0, 0);
      }
    }
  };
  auto deposit = [&](bf16* buf, int cnt) {
#pragma unroll
    for (int i = 0; i < S; ++i) {
      if (i < cnt) {
        int v = lane + 32 * i, r = v / (2 * cnt), c = v % (2 * cnt);
        *reinterpret_cast<uint4*>(buf + r * kStride + 8 * c) = staged[i];
      }
    }
  };
  auto compute = [&](const bf16* buf, int base, int cnt) {
#pragma unroll
    for (int u = 0; u < S; ++u) {
      if (u < cnt) {
        const bf16* r0p = buf + group * kStride + 16 * u + 2 * t;
        const bf16* r1p = buf + (group + 8) * kStride + 16 * u + 2 * t;
        uint32_t a0 = *reinterpret_cast<const uint32_t*>(r0p), a1 = *reinterpret_cast<const uint32_t*>(r1p);
        uint32_t a2 = *reinterpret_cast<const uint32_t*>(r0p + 8), a3 = *reinterpret_cast<const uint32_t*>(r1p + 8);
        int o = base + 16 * u;
        uint32_t b0 = bx ? v5::ld_nc(xb + o) : 0u, b1 = bx ? v5::ld_nc(xb + o + 8) : 0u;
        v5::mma(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
      }
    }
  };
  int steps = (end - start) / 16, stages = (steps + S - 1) / S, buf = 0;
  int cnt0 = min(S, steps);
  fetch(start, cnt0);
  deposit(tile[warp][0], cnt0);
  __syncwarp();
  for (int s = 0; s < stages; ++s) {
    int base = start + s * 16 * S, cnt = min(S, steps - s * S);
    bool more = s + 1 < stages;
    int nbase = base + 16 * S, ncnt = more ? min(S, steps - (s + 1) * S) : 0;
    if (more) fetch(nbase, ncnt);              // next stage's global reads in flight during this stage's MMAs
    compute(tile[warp][buf], base, cnt);       // same K order as v4: steps base, base+16, ...
    if (more) deposit(tile[warp][buf ^ 1], ncnt);  // that buffer's last readers finished before the prior sync
    __syncwarp();
    buf ^= 1;
  }
  size_t head = size_t(blockIdx.z) * padded_m * padded_n;
  if (row + group < m) { size_t o = head + size_t(row + group) * padded_n + 2 * t; partial[o] = c0; partial[o + 1] = c1; }
  if (row + group + 8 < m) { size_t o = head + size_t(row + group + 8) * padded_n + 2 * t; partial[o] = c2; partial[o + 1] = c3; }
}
}  // namespace v6

#ifndef CANDIDATE
#define CANDIDATE v5::mma_thin_partials
#endif

using bf16 = __nv_bfloat16;
static void check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }

struct Case { int m, n, k, parts; bool timed; };

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t next() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return uint32_t(rng >> 16); }
// Finite BF16 with a wide exponent spread (2^-20..2^4) and random sign, to stress rounding.
static uint16_t random_bf16() {
  uint32_t r = next();
  uint32_t exponent = 127 - 20 + (r % 25), sign = (r >> 8) & 1, mantissa = (r >> 9) & 0x7F;
  return uint16_t((sign << 15) | (exponent << 7) | mantissa);
}

template <typename F>
static void launch_thin(F kernel, const bf16* w, const bf16* x, float* p, int m, int n, int k, int pn, int pm,
                        int parts) {
  int warps = std::min(4, pm / 16);
  kernel<<<dim3((pm / 16 + warps - 1) / warps, 1, parts), 32 * warps>>>(w, x, p, m, n, k, pn, pm, parts);
}

template <int U>
static int run_case(const Case& c, std::FILE* report) {
  int pm = (c.m + 15) / 16 * 16, pn = 8;
  size_t wn = size_t(c.m) * c.k, xn = size_t(c.n) * c.k, pcount = size_t(pm) * pn * c.parts, out = size_t(c.m) * c.n;
  std::vector<uint16_t> hw(wn), hx(xn);
  for (auto& v : hw) v = random_bf16();
  for (auto& v : hx) v = random_bf16();
  bf16 *w, *x, *o4, *o5; float *p4, *p5;
  check(cudaMalloc(&w, wn * 2)); check(cudaMalloc(&x, xn * 2));
  check(cudaMalloc(&p4, pcount * 4)); check(cudaMalloc(&p5, pcount * 4));
  check(cudaMalloc(&o4, out * 2)); check(cudaMalloc(&o5, out * 2));
  check(cudaMemcpy(w, hw.data(), wn * 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(x, hx.data(), xn * 2, cudaMemcpyHostToDevice));
  // Harness-only sentinels: a slot one kernel writes and the other does not shows up as a mismatch.
  check(cudaMemset(p4, 0xA5, pcount * 4)); check(cudaMemset(p5, 0x5A, pcount * 4));
  launch_thin(lfm3_wmma_candidate::mma_thin_partials, w, x, p4, c.m, c.n, c.k, pn, pm, c.parts);
  launch_thin(CANDIDATE<U>, w, x, p5, c.m, c.n, c.k, pn, pm, c.parts);
  lfm3_wmma_candidate::wmma_finish<<<(out + 255) / 256, 256>>>(p4, o4, out, c.m, c.n, pm, pn, c.parts);
  lfm3_wmma_candidate::wmma_finish<<<(out + 255) / 256, 256>>>(p5, o5, out, c.m, c.n, pm, pn, c.parts);
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  std::vector<uint32_t> hp4(pcount), hp5(pcount); std::vector<uint16_t> ho4(out), ho5(out);
  check(cudaMemcpy(hp4.data(), p4, pcount * 4, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(hp5.data(), p5, pcount * 4, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(ho4.data(), o4, out * 2, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(ho5.data(), o5, out * 2, cudaMemcpyDeviceToHost));
  size_t partial_diff = 0, partial_checked = 0, out_diff = 0;
  for (int part = 0; part < c.parts; ++part)
    for (int r = 0; r < c.m; ++r)
      for (int col = 0; col < pn; ++col, ++partial_checked) {
        size_t i = size_t(part) * pm * pn + size_t(r) * pn + col;
        partial_diff += hp4[i] != hp5[i];
      }
  for (size_t i = 0; i < out; ++i) out_diff += ho4[i] != ho5[i];
  double t4 = 0, t5 = 0, tb = 0;
  if (c.timed) {
    cudaEvent_t a, b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
    // Cold-cache streaming: rotate through enough weight copies (>= 2 GiB) that no repetition
    // can be served from L2/system cache, matching a real decode's single pass over each layer.
    size_t copies = std::max<size_t>(1, std::min<size_t>(256, (size_t(2) << 30) / (wn * 2) + 1));
    std::vector<bf16*> ws(copies, w);
    for (size_t i = 1; i < copies; ++i) { check(cudaMalloc(&ws[i], wn * 2)); check(cudaMemcpy(ws[i], w, wn * 2, cudaMemcpyDeviceToDevice)); }
    auto time = [&](auto kernel, float* p) {
      for (int i = 0; i < 5; ++i) launch_thin(kernel, ws[i % copies], x, p, c.m, c.n, c.k, pn, pm, c.parts);
      int reps = int(std::max<size_t>(copies, 20));
      check(cudaEventRecord(a));
      for (int i = 0; i < reps; ++i) launch_thin(kernel, ws[(i + 5) % copies], x, p, c.m, c.n, c.k, pn, pm, c.parts);
      check(cudaEventRecord(b)); check(cudaEventSynchronize(b));
      float ms; check(cudaEventElapsedTime(&ms, a, b)); return double(ms) / reps * 1e-3;
    };
    t4 = time(lfm3_wmma_candidate::mma_thin_partials, p4);
    t5 = time(CANDIDATE<U>, p5);
    {  // Production's call, as in v4's non-native fallback: cublasGemmEx, BF16 in/out, FP32 compute.
      cublasHandle_t h; if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate");
      float alpha = 1, beta = 0;
      auto gemm = [&](const bf16* wp) {
        if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, c.m, c.n, c.k, &alpha, wp, CUDA_R_16BF, c.k, x, CUDA_R_16BF, c.k,
                         &beta, o4, CUDA_R_16BF, c.m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("cublasGemmEx");
      };
      for (int i = 0; i < 5; ++i) gemm(ws[i % copies]);
      int reps = int(std::max<size_t>(copies, 20));
      check(cudaEventRecord(a));
      for (int i = 0; i < reps; ++i) gemm(ws[(i + 5) % copies]);
      check(cudaEventRecord(b)); check(cudaEventSynchronize(b));
      float ms; check(cudaEventElapsedTime(&ms, a, b)); tb = double(ms) / reps * 1e-3;
      cublasDestroy(h);
    }
    for (size_t i = 1; i < copies; ++i) check(cudaFree(ws[i]));
    check(cudaEventDestroy(a)); check(cudaEventDestroy(b));
  }
  bool exact = partial_diff == 0 && out_diff == 0;
  std::fprintf(report,
               "{\"U\":%d,\"m\":%d,\"n\":%d,\"k\":%d,\"parts\":%d,\"partials_checked\":%zu,\"partial_diff\":%zu,"
               "\"outputs_checked\":%zu,\"output_diff\":%zu,\"exact\":%s",
               U, c.m, c.n, c.k, c.parts, partial_checked, partial_diff, out, out_diff, exact ? "true" : "false");
  if (c.timed) {
    double bytes = double(wn) * 2;
    std::fprintf(report, ",\"v4_us\":%.2f,\"v5_us\":%.2f,\"cublas_us\":%.2f,\"v4_gbs\":%.1f,\"v5_gbs\":%.1f,\"cublas_gbs\":%.1f,\"speedup\":%.3f",
                 t4 * 1e6, t5 * 1e6, tb * 1e6, bytes / t4 / 1e9, bytes / t5 / 1e9, bytes / tb / 1e9, t4 / t5);
  }
  std::fprintf(report, "}\n"); std::fflush(report);
  check(cudaFree(w)); check(cudaFree(x)); check(cudaFree(p4)); check(cudaFree(p5)); check(cudaFree(o4)); check(cudaFree(o5));
  return exact ? 0 : 1;
}

static std::vector<char> read_file(const std::string& path, size_t bytes) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f || f.tellg() != std::streamoff(bytes)) throw std::runtime_error("fixture size: " + path);
  std::vector<char> b(bytes); f.seekg(0); f.read(b.data(), bytes); return b;
}

// Real checkpoint fixture: layer-0 conv in_proj decode (m=6144, k=2048, n=1, 2 partitions) vs reference.
template <int U>
static int real_fixture(const char* wpath, const char* xpath, const char* epath, std::FILE* report) {
  const int m = 6144, k = 2048, n = 1, parts = 2, pm = m, pn = 8;
  auto hw = read_file(wpath, size_t(m) * k * 2), hx = read_file(xpath, size_t(n) * k * 2),
       he = read_file(epath, size_t(m) * n * 2);
  bf16 *w, *x, *o; float* p;
  check(cudaMalloc(&w, hw.size())); check(cudaMalloc(&x, hx.size())); check(cudaMalloc(&o, he.size()));
  check(cudaMalloc(&p, size_t(pm) * pn * parts * 4));
  check(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice));
  check(cudaMemcpy(x, hx.data(), hx.size(), cudaMemcpyHostToDevice));
  launch_thin(CANDIDATE<U>, w, x, p, m, n, k, pn, pm, parts);
  lfm3_wmma_candidate::wmma_finish<<<(m * n + 255) / 256, 256>>>(p, o, size_t(m) * n, m, n, pm, pn, parts);
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  std::vector<char> ho(he.size()); check(cudaMemcpy(ho.data(), o, ho.size(), cudaMemcpyDeviceToHost));
  size_t diff = 0;
  for (size_t i = 0; i < size_t(m) * n; ++i) diff += std::memcmp(ho.data() + 2 * i, he.data() + 2 * i, 2) != 0;
  std::fprintf(report, "{\"U\":%d,\"real_fixture\":\"layer0 conv.in_proj decode\",\"checked\":%d,\"mismatches\":%zu}\n",
               U, m * n, diff);
  check(cudaFree(w)); check(cudaFree(x)); check(cudaFree(o)); check(cudaFree(p));
  return diff ? 1 : 0;
}

template <int U>
static int suite(int argc, char** argv) {
  std::vector<Case> cases = {
      // Decode shapes observed in the v4 profile (n=1), plus n=2 as used by the grouped worker.
      {128000, 1, 2048, 1, true}, {7680, 1, 2048, 1, true}, {2048, 1, 7680, 5, true}, {6144, 1, 2048, 2, true},
      {2048, 1, 2048, 1, true},   {512, 1, 2048, 2, true},  {16, 1, 2048, 2, true},
      {7680, 2, 2048, 1, false},  {2048, 2, 7680, 5, false}, {6144, 2, 2048, 2, false}, {2048, 2, 2048, 1, false}};
  // Tail geometry: partial row blocks, warp retirement, n up to the thin limit, short K, uneven step counts.
  for (int m : {1, 17, 63, 65})
    for (int n : {1, 2, 7, 8})
      for (int k : {16, 32, 48, 64, 272, 2048})
        for (int parts : {1, 2, 3, 4})
          if (k % parts == 0 && (k / parts) % 16 == 0) cases.push_back({m, n, k, parts, false});
  int failures = 0;
  for (const auto& c : cases) failures += run_case<U>(c, stdout);
  if (argc == 4) failures += real_fixture<U>(argv[1], argv[2], argv[3], stdout);
  std::fprintf(stdout, "{\"U\":%d,\"cases\":%zu,\"failures\":%d,\"full_acceptance\":false}\n", U, cases.size(), failures);
  return failures;
}

int main(int argc, char** argv) try {
  int failures = suite<UNROLL>(argc, argv);
  return failures ? 2 : 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
