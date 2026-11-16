// tcgen05 GEMM probe (validation/tcgen05-probe-v1/README.md): the probe kernels of tcgen05_gemm.cuh against
// lfm::gemm (built apart for sm_110, tcgen05_probe_engine_gemm.cu), an FP64 reference and cuBLAS.
//   tcgen05-gemm-probe unit                       descriptor and TMA unit tests (bit-exact, integer inputs)
//   tcgen05-gemm-probe peak [iters]               tcgen05.mma alone, shared-memory-resident operands
//   tcgen05-gemm-probe correct V [opts]           rules C1-C5, one JSON line per case on stdout
//   tcgen05-gemm-probe sanitize V OUTDIR [opts]   the sanitizer case set; outputs written to OUTDIR for comparison
//   tcgen05-gemm-probe speed V[,V...] [opts]      S1-S3 timing against lfm::gemm and cuBLAS
// opts: --shapes a,b  --m 64,100  --data int,normal,split  --runs 5  --timeout-ms 2000  --reps 20
// Variants: 1 (V1: 1-CTA, one tile per CTA), 2 (V2: 1-CTA persistent, two accumulators), 3 (V3: 2-CTA pairs,
// persistent, two accumulators, 256 x 256 tiles).
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "lfm/gemm.cuh"
#include "tcgen05_gemm.cuh"

using bf16 = __nv_bfloat16;
void engine_gemm(const lfm::gemm::Problem& p, cudaStream_t stream);

// ---- host utilities ----------------------------------------------------------------------------------------------
static int* g_trap_host = nullptr;  // watchdog tag, host-mapped memory (host view)
static int* g_trap = nullptr;       // the same memory, device view (given to kernels)
static void fail_trap(const char* what, cudaError_t e) {
  std::fprintf(stderr, "CUDA error after %s: %s; watchdog tag %d\n", what, cudaGetErrorString(e), g_trap_host ? *g_trap_host : -1);
  std::printf("{\"fatal\":\"%s\",\"error\":\"%s\",\"watchdog_tag\":%d}\n", what, cudaGetErrorString(e), g_trap_host ? *g_trap_host : -1);
  std::fflush(stdout);
  std::exit(3);
}
static void check(cudaError_t e, const char* what = "cuda") { if (e != cudaSuccess) fail_trap(what, e); }
static void check(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublas " + std::to_string(s)); }
static void check(CUresult s) { if (s != CUDA_SUCCESS) throw std::runtime_error("driver " + std::to_string(int(s))); }
static void sync(const char* what) { check(cudaDeviceSynchronize(), what); }

static uint16_t to_bits(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static float from_bits(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static double ulp_bf16(double x) {  // BF16: 8 significant bits
  if (x == 0) return std::ldexp(1.0, -133);
  int e; std::frexp(std::fabs(x), &e);
  return std::ldexp(1.0, e - 8);
}
static int ordered(uint16_t u) { return (u & 0x8000) ? -int(u & 0x7fff) : int(u); }

struct Device {
  void* p = nullptr;
  explicit Device(size_t bytes) { check(cudaMalloc(&p, std::max<size_t>(bytes, 16)), "cudaMalloc"); }
  ~Device() { cudaFree(p); }
  Device(const Device&) = delete;
  template <class T> T* as() const { return static_cast<T*>(p); }
};
template <class T> static std::vector<T> download(const void* p, size_t n) {
  std::vector<T> h(n);
  check(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
  return h;
}
template <class F> static void parallel_for(size_t n, F&& f) {
  const unsigned t = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::vector<std::thread> pool;
  for (unsigned w = 0; w < t; ++w)
    pool.emplace_back([&, w] { for (size_t i = w; i < n; i += t) f(i); });
  for (auto& th : pool) th.join();
}

// ---- data generators (device, deterministic in (seed, index)) ----------------------------------------------------
__host__ __device__ inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
__device__ inline float unif(uint64_t seed, uint64_t i) { return float(mix64(seed ^ mix64(i)) >> 40) * 0x1p-24f; }
__device__ inline float gauss(uint64_t seed, uint64_t i) {
  const float u1 = unif(seed, 2 * i) + 0x1p-25f, u2 = unif(seed, 2 * i + 1);
  return sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);
}
__global__ void gen_int(bf16* p, size_t n, uint64_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = __float2bfloat16_rn(float(int(mix64(seed ^ mix64(i)) % 9) - 4));
}
// Activations: unit normal x 2^(row exponent) x 2^(column exponent), exponents uniform in [-4, 4].
__device__ inline float act(uint64_t seed, int r, int c, int cols) {
  const float re = unif(seed + 1, r) * 8.f - 4.f, ce = unif(seed + 2, c) * 8.f - 4.f;
  return gauss(seed, uint64_t(r) * cols + c) * exp2f(re) * exp2f(ce);
}
__global__ void gen_act(bf16* p, int rows, int cols, uint64_t seed) {
  const size_t n = size_t(rows) * cols;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = __float2bfloat16_rn(act(seed, int(i / cols), int(i % cols), cols));
}
__global__ void gen_weight(bf16* p, size_t n, uint64_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = __float2bfloat16_rn(gauss(seed, i));
}
// The vision encoder's split: A = [hi | lo] per row (x FP32, hi = bf16(x), lo = bf16(x - hi)); B = [W | W].
__global__ void gen_split_a(bf16* p, int rows, int kin, uint64_t seed) {
  const size_t n = size_t(rows) * kin;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int r = int(i / kin), c = int(i % kin);
    const float x = act(seed, r, c, kin);
    const bf16 hi = __float2bfloat16_rn(x), lo = __float2bfloat16_rn(x - __bfloat162float(hi));
    p[size_t(r) * 2 * kin + c] = hi;
    p[size_t(r) * 2 * kin + kin + c] = lo;
  }
}
__global__ void gen_split_b(bf16* p, int rows, int kin, uint64_t seed) {
  const size_t n = size_t(rows) * kin;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int r = int(i / kin), c = int(i % kin);
    const bf16 w = __float2bfloat16_rn(gauss(seed, i));
    p[size_t(r) * 2 * kin + c] = w;
    p[size_t(r) * 2 * kin + kin + c] = w;
  }
}

// ---- FP64 reference: every element, and S = sum |a_k b_k| (BF16 products are exact in FP64) --------------------
__global__ void __launch_bounds__(256) ref_f64(const bf16* A, const bf16* B, int M, int N, int K, double* ref, double* S) {
  __shared__ double as[32][17], bs[32][17];
  const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  const int row = blockIdx.y * 16 + ty, col = blockIdx.x * 16 + tx;
  double acc = 0, s = 0;
  for (int k0 = 0; k0 < K; k0 += 32) {
    for (int e = threadIdx.x; e < 512; e += 256) {
      const int r = e / 32, kk = e % 32, gr = blockIdx.y * 16 + r, gc = blockIdx.x * 16 + r;
      as[kk][r] = gr < M ? double(__bfloat162float(A[size_t(gr) * K + k0 + kk])) : 0.0;
      bs[kk][r] = double(__bfloat162float(B[size_t(gc) * K + k0 + kk]));
    }
    __syncthreads();
#pragma unroll 8
    for (int kk = 0; kk < 32; ++kk) {
      const double p = as[kk][ty] * bs[kk][tx];
      acc += p;
      s += fabs(p);
    }
    __syncthreads();
  }
  if (row < M) { ref[size_t(row) * N + col] = acc; S[size_t(row) * N + col] = s; }
}

// ---- unit tests ---------------------------------------------------------------------------------------------------
// One MMA (ksteps = 1) or one K = 64 stage (ksteps = 4) of 128 x 256, operands written to shared memory by threads
// in the documented swizzled layout (independent of TMA), D read back through tcgen05.ld.
__global__ void __launch_bounds__(128, 1) unit_mma(const bf16* A, const bf16* B, float* D, int ksteps, tc05::Watchdog wd) {
  using namespace tc05;
  extern __shared__ uint8_t dsmem[];
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t slot;
  const uint32_t base = (smem_u32(dsmem) + 1023u) & ~1023u;
  uint8_t* gbase = dsmem + (base - smem_u32(dsmem));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int i = threadIdx.x; i < kBM * 8; i += 128)
    *reinterpret_cast<int4*>(gbase + sw128_offset(i / 8, (i % 8) * 8)) = reinterpret_cast<const int4*>(A)[i];
  for (int i = threadIdx.x; i < kBN * 8; i += 128)
    *reinterpret_cast<int4*>(gbase + kABytes + sw128_offset(i / 8, (i % 8) * 8)) = reinterpret_cast<const int4*>(B)[i];
  fence_proxy_async_smem();
  if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); }
  if (warp == 0) { tmem_alloc(smem_u32(&slot), 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = slot;
  if (threadIdx.x == 0) {
    const uint64_t ad = smem_desc_sw128(base), bd = smem_desc_sw128(base + kABytes);
    for (int kk = 0; kk < ksteps; ++kk) mma_bf16(tmem, ad + 2 * kk, bd + 2 * kk, idesc_bf16_f32(kBM, kBN), kk > 0);
    mma_commit(smem_u32(&bar));
  }
  __syncwarp();
  mbar_wait(smem_u32(&bar), 0, wd, 9);
  tc_fence_after();
  const int row = warp * 32 + lane;
  for (int c = 0; c < 8; ++c) {
    uint32_t r[32];
    tmem_ld32(tmem + (uint32_t(warp * 32) << 16) + c * 32, r);
    tmem_ld_wait();
    for (int j = 0; j < 32; ++j) D[row * 256 + c * 32 + j] = __uint_as_float(r[j]);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(tmem, 256); }
}
// TMA with SWIZZLE_128B: a 64 x 128 box at row y0 copied raw to `out`, for comparison with sw128_offset.
__global__ void unit_tma(const __grid_constant__ CUtensorMap map, uint8_t* out, int y0, tc05::Watchdog wd) {
  using namespace tc05;
  extern __shared__ uint8_t dsmem[];
  __shared__ __align__(8) uint64_t bar;
  const uint32_t base = (smem_u32(dsmem) + 1023u) & ~1023u;
  const uint8_t* gbase = dsmem + (base - smem_u32(dsmem));
  if (threadIdx.x == 0) {
    mbar_init(smem_u32(&bar), 1);
    fence_mbar_init();
    mbar_expect_tx(smem_u32(&bar), kABytes);
    tma_load_2d(base, &map, 0, y0, smem_u32(&bar));
  }
  __syncthreads();
  mbar_wait(smem_u32(&bar), 0, wd, 8);
  for (int i = threadIdx.x; i < int(kABytes / 16); i += blockDim.x)
    reinterpret_cast<int4*>(out)[i] = reinterpret_cast<const int4*>(gbase)[i];
}

// ---- peak: tcgen05.mma alone ---------------------------------------------------------------------------------------
__global__ void __launch_bounds__(128, 1) peak_1cta(int iters, tc05::Watchdog wd) {
  using namespace tc05;
  extern __shared__ uint8_t dsmem[];
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t slot;
  const uint32_t base = (smem_u32(dsmem) + 1023u) & ~1023u;
  uint8_t* gbase = dsmem + (base - smem_u32(dsmem));
  const int warp = threadIdx.x / 32;
  for (int i = threadIdx.x; i < int(kStageBytes / 2); i += 128)
    reinterpret_cast<bf16*>(gbase)[i] = __float2bfloat16_rn(float((i * 7) % 9 - 4) * 0.25f);
  fence_proxy_async_smem();
  if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); }
  if (warp == 0) { tmem_alloc(smem_u32(&slot), 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = slot;
  if (threadIdx.x == 0) {
    const uint64_t ad = smem_desc_sw128(base), bd = smem_desc_sw128(base + kABytes);
    for (int i = 0; i < iters; ++i)
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) mma_bf16(tmem, ad + 2 * kk, bd + 2 * kk, idesc_bf16_f32(kBM, kBN), (i | kk) != 0);
    mma_commit(smem_u32(&bar));
  }
  __syncwarp();
  mbar_wait(smem_u32(&bar), 0, wd, 7);
  tc_fence_before();
  __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(tmem, 256); }
}

// 2-CTA: one MMA (K = 16) or one stage (K = 64) of 256 x 256 issued by the leader over both CTAs' shared memory.
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1)
unit_mma_pair(const bf16* A, const bf16* B, float* D, int ksteps, tc05::Watchdog wd) {
  using namespace tc05;
  extern __shared__ uint8_t dsmem[];
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t slot;
  const uint32_t base = (smem_u32(dsmem) + 1023u) & ~1023u;
  uint8_t* gbase = dsmem + (base - smem_u32(dsmem));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t rank = cluster_rank();
  const int4* a = reinterpret_cast<const int4*>(A) + rank * 128 * 8;
  const int4* b = reinterpret_cast<const int4*>(B) + rank * 128 * 8;
  for (int i = threadIdx.x; i < 128 * 8; i += 128) {
    *reinterpret_cast<int4*>(gbase + sw128_offset(i / 8, (i % 8) * 8)) = a[i];
    *reinterpret_cast<int4*>(gbase + kABytes + sw128_offset(i / 8, (i % 8) * 8)) = b[i];
  }
  fence_proxy_async_smem();
  if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); }
  if (warp == 0) { tmem_alloc_pair(smem_u32(&slot), 256); tmem_relinquish_pair(); }
  tc_fence_before();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = slot;
  if (rank == 0 && threadIdx.x == 0) {
    const uint64_t ad = smem_desc_sw128(base), bd = smem_desc_sw128(base + kABytes);
    for (int kk = 0; kk < ksteps; ++kk) mma_bf16_pair(tmem, ad + 2 * kk, bd + 2 * kk, idesc_bf16_f32(256, 256), kk > 0);
    mma_commit_pair(smem_u32(&bar));
  }
  __syncwarp();
  mbar_wait_cluster(smem_u32(&bar), 0, wd, 19);
  tc_fence_after();
  const int row = int(rank) * 128 + warp * 32 + lane;
  for (int c = 0; c < 8; ++c) {
    uint32_t r[32];
    tmem_ld32(tmem + (uint32_t(warp * 32) << 16) + c * 32, r);
    tmem_ld_wait();
    for (int j = 0; j < 32; ++j) D[row * 256 + c * 32 + j] = __uint_as_float(r[j]);
  }
  tc_fence_before();
  cluster_sync();
  if (warp == 0) { tc_fence_after(); tmem_dealloc_pair(tmem, 256); }
}
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) peak_2cta(int iters, tc05::Watchdog wd) {
  using namespace tc05;
  extern __shared__ uint8_t dsmem[];
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t slot;
  const uint32_t base = (smem_u32(dsmem) + 1023u) & ~1023u;
  uint8_t* gbase = dsmem + (base - smem_u32(dsmem));
  const int warp = threadIdx.x / 32;
  const uint32_t rank = cluster_rank();
  for (int i = threadIdx.x; i < int(kPairStageBytes / 2); i += 128)
    reinterpret_cast<bf16*>(gbase)[i] = __float2bfloat16_rn(float((i * 7) % 9 - 4) * 0.25f);
  fence_proxy_async_smem();
  if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); }
  if (warp == 0) { tmem_alloc_pair(smem_u32(&slot), 256); tmem_relinquish_pair(); }
  tc_fence_before();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = slot;
  if (rank == 0 && threadIdx.x == 0) {
    const uint64_t ad = smem_desc_sw128(base), bd = smem_desc_sw128(base + kABytes);
    for (int i = 0; i < iters; ++i)
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) mma_bf16_pair(tmem, ad + 2 * kk, bd + 2 * kk, idesc_bf16_f32(256, 256), (i | kk) != 0);
    mma_commit_pair(smem_u32(&bar));
  }
  __syncwarp();
  mbar_wait_cluster(smem_u32(&bar), 0, wd, 17);
  tc_fence_before();
  cluster_sync();
  if (warp == 0) { tc_fence_after(); tmem_dealloc_pair(tmem, 256); }
}

// ---- variants ------------------------------------------------------------------------------------------------------
static int g_sms = 0;
static unsigned long long g_timeout_ns = 2000000000ull;

static CUtensorMap make_map(const bf16* p, int rows, int k, int box_rows) {
  CUtensorMap m{};
  cuuint64_t dims[2] = {cuuint64_t(k), cuuint64_t(rows)};
  cuuint64_t strides[1] = {cuuint64_t(k) * 2};
  cuuint32_t box[2] = {64, cuuint32_t(box_rows)}, es[2] = {1, 1};
  check(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<bf16*>(p), dims, strides, box, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}

struct Plan {
  int variant = 1;
  bool bf16_out = false;
  CUtensorMap ma{}, mb{};
  tc05::Args args;
  int grid = 0;
  size_t smem = 0;
};
constexpr int kStages = 4, kPairStages = 6;
template <int ACC, bool BF> static void set_smem() {
  static bool done = false;
  if (!done) {
    check(cudaFuncSetAttribute(tc05::gemm_1cta<kStages, ACC, BF>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               int(kStages * tc05::kStageBytes + 1024)), "smem attribute");
    done = true;
  }
}
template <bool BF> static void set_smem_pair() {
  static bool done = false;
  if (!done) {
    check(cudaFuncSetAttribute(tc05::gemm_2cta<kPairStages, BF>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               int(kPairStages * tc05::kPairStageBytes + 1024)), "smem attribute");
    done = true;
  }
}
static Plan make_plan(int variant, bool bf16_out, const bf16* A, const bf16* B, void* C, int M, int N, int K) {
  if (M < 1 || N % tc05::kBN || K % tc05::kBK || K < tc05::kBK) throw std::runtime_error("probe contract: shape");
  if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) | reinterpret_cast<uintptr_t>(C)) % 16)
    throw std::runtime_error("probe contract: alignment");
  Plan p;
  p.variant = variant;
  p.bf16_out = bf16_out;
  p.ma = make_map(A, M, K, tc05::kBM);
  p.mb = make_map(B, N, K, variant == 3 ? 128 : tc05::kBN);
  auto& a = p.args;
  a.m = M; a.n = N; a.k = K; a.c = C; a.ldc = N;
  const int bm = variant == 3 ? tc05::kPairBM : tc05::kBM;
  a.tiles_m = (M + bm - 1) / bm;
  a.tiles_n = N / tc05::kBN;
  a.tiles = a.tiles_m * a.tiles_n;
  a.band = std::max(1, std::min(a.tiles_n, int((8u << 20) / (uint32_t(tc05::kBN) * K * 2))));
  a.wd.timeout_ns = g_timeout_ns;
  a.wd.trap_tag = g_trap;
  p.smem = kStages * tc05::kStageBytes + 1024;
  if (variant == 1) p.grid = a.tiles;
  else if (variant == 2) p.grid = std::min(a.tiles, g_sms);
  else if (variant == 3) { p.grid = 2 * std::min(a.tiles, g_sms / 2); p.smem = kPairStages * tc05::kPairStageBytes + 1024; }
  else throw std::runtime_error("unknown variant");
  return p;
}
static void run_plan(const Plan& p, cudaStream_t s = 0) {
  using namespace tc05;
  if (p.variant == 1) {
    if (p.bf16_out) { set_smem<1, true>(); gemm_1cta<kStages, 1, true><<<p.grid, 192, p.smem, s>>>(p.ma, p.mb, p.args); }
    else { set_smem<1, false>(); gemm_1cta<kStages, 1, false><<<p.grid, 192, p.smem, s>>>(p.ma, p.mb, p.args); }
  } else if (p.variant == 3) {
    if (p.bf16_out) { set_smem_pair<true>(); gemm_2cta<kPairStages, true><<<p.grid, 192, p.smem, s>>>(p.ma, p.mb, p.args); }
    else { set_smem_pair<false>(); gemm_2cta<kPairStages, false><<<p.grid, 192, p.smem, s>>>(p.ma, p.mb, p.args); }
  } else {
    if (p.bf16_out) { set_smem<2, true>(); gemm_1cta<kStages, 2, true><<<p.grid, 192, p.smem, s>>>(p.ma, p.mb, p.args); }
    else { set_smem<2, false>(); gemm_1cta<kStages, 2, false><<<p.grid, 192, p.smem, s>>>(p.ma, p.mb, p.args); }
  }
  check(cudaGetLastError(), "probe launch");
}
static void run_engine(const bf16* A, const bf16* B, void* C, bool bf16_out, int M, int N, int K, cudaStream_t s = 0) {
  auto p = lfm::gemm::linear(A, B, bf16_out ? static_cast<bf16*>(C) : nullptr, M, K, N);
  if (!bf16_out) p.c_f32 = static_cast<float*>(C);
  engine_gemm(p, s);
  check(cudaGetLastError(), "lfm::gemm launch");
}

// ---- shapes and options --------------------------------------------------------------------------------------------
struct Shape { const char* name; int n, k; bool vision; };
static const Shape kShapes[] = {
    {"v.patchifier", 1024, 3072, true}, {"v.in_proj", 3072, 2048, true}, {"v.q_out", 1024, 2048, true},
    {"v.kv", 512, 2048, true},          {"v.w1w3", 2560, 2048, true},    {"v.w2", 1024, 5120, true},
    {"v.proj1", 4096, 8192, true},      {"v.proj2", 2048, 8192, true},   {"lm.in_proj", 6144, 2048, false},
    {"lm.q_o", 2048, 2048, false},      {"lm.kv", 512, 2048, false},     {"lm.w1w3", 7680, 2048, false},
    {"lm.w2", 2048, 7680, false}};

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
struct Options {
  std::vector<std::string> shapes, data{"int", "normal", "split"};
  std::vector<int> ms{64, 100, 128, 257, 1000, 1980, 4096, 7920, 8192};
  int runs = 5, reps = 20;
};
static Options parse(int argc, char** argv, int first) {
  Options o;
  for (int i = first; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
    if (a == "--shapes") o.shapes = split_list(next());
    else if (a == "--data") o.data = split_list(next());
    else if (a == "--m") { o.ms.clear(); for (auto& t : split_list(next())) o.ms.push_back(std::stoi(t)); }
    else if (a == "--runs") o.runs = std::stoi(next());
    else if (a == "--reps") o.reps = std::stoi(next());
    else if (a == "--timeout-ms") g_timeout_ns = std::stoull(next()) * 1000000ull;
    else throw std::runtime_error("unknown option " + a);
  }
  return o;
}
static bool selected(const Options& o, const Shape& s) {
  if (o.shapes.empty()) return true;
  for (auto& n : o.shapes)
    if (n == s.name || (n == "vision" && s.vision) || (n == "lm" && !s.vision)) return true;
  return false;
}

// Inputs for one case. kind: int (C1), normal (C2), split (C2, vision only: K = 2 x Kin).
struct Inputs {
  Device a, b;
  int m, n, k;
  Inputs(const std::string& kind, int m_, int n_, int k_, uint64_t seed)
      : a(size_t(m_) * k_ * 2), b(size_t(n_) * k_ * 2), m(m_), n(n_), k(k_) {
    const int g = 4 * std::max(g_sms, 1) * 8;
    if (kind == "int") {
      gen_int<<<g, 256>>>(a.as<bf16>(), size_t(m) * k, seed);
      gen_int<<<g, 256>>>(b.as<bf16>(), size_t(n) * k, seed + 7);
    } else if (kind == "normal") {
      gen_act<<<g, 256>>>(a.as<bf16>(), m, k, seed);
      gen_weight<<<g, 256>>>(b.as<bf16>(), size_t(n) * k, seed + 7);
    } else if (kind == "split") {
      gen_split_a<<<g, 256>>>(a.as<bf16>(), m, k / 2, seed);
      gen_split_b<<<g, 256>>>(b.as<bf16>(), n, k / 2, seed + 7);
    } else throw std::runtime_error("unknown data " + kind);
    sync("generate");
  }
};
static uint64_t case_seed(const std::string& kind, const Shape& s, int m) {
  uint64_t h = mix64(std::hash<std::string>{}(kind));
  h = mix64(h ^ std::hash<std::string>{}(s.name));
  return mix64(h ^ uint64_t(m));
}

// ---- unit mode ------------------------------------------------------------------------------------------------------
static int unit_mode() {
  using namespace tc05;
  tc05::Watchdog wd{g_timeout_ns, g_trap};
  int failures = 0;
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<int> d(-4, 4);
  // U1: TMA SWIZZLE_128B layout equals sw128_offset, rows past the tensor zero-filled.
  {
    const int R = 200;
    std::vector<uint16_t> x(size_t(R) * 64);
    for (auto& v : x) v = to_bits(float(d(rng)));
    Device dx(x.size() * 2), dout(kABytes);
    check(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    const CUtensorMap map = make_map(dx.as<bf16>(), R, 64, kBM);
    check(cudaFuncSetAttribute(unit_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kABytes + 1024)));
    for (int y0 : {0, 128}) {
      check(cudaMemset(dout.p, 0x5a, kABytes));
      unit_tma<<<1, 128, kABytes + 1024>>>(map, dout.as<uint8_t>(), y0, wd);
      sync("unit_tma");
      const auto got = download<uint8_t>(dout.p, kABytes);
      size_t bad = 0;
      for (int r = 0; r < kBM; ++r)
        for (int k = 0; k < 64; ++k) {
          const uint16_t want = y0 + r < R ? x[size_t(y0 + r) * 64 + k] : 0;
          uint16_t have;
          std::memcpy(&have, &got[sw128_offset(r, k)], 2);
          bad += have != want;
        }
      std::printf("{\"unit\":\"tma_sw128\",\"y0\":%d,\"rows_in_tensor\":%d,\"mismatches\":%zu,\"pass\":%s}\n", y0,
                  std::max(0, std::min(kBM, R - y0)), bad, bad ? "false" : "true");
      failures += bad != 0;
    }
  }
  // U2, U3: one MMA (K = 16) and one stage (K = 64), 128 x 256, against the exact integer product.
  {
    std::vector<uint16_t> a(kBM * 64), b(kBN * 64);
    for (auto& v : a) v = to_bits(float(d(rng)));
    for (auto& v : b) v = to_bits(float(d(rng)));
    Device da(a.size() * 2), db(b.size() * 2), dd(size_t(kBM) * kBN * 4);
    check(cudaMemcpy(da.p, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.p, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
    check(cudaFuncSetAttribute(unit_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kStageBytes + 1024)));
    for (int ksteps : {1, 4}) {
      check(cudaMemset(dd.p, 0xff, size_t(kBM) * kBN * 4));
      unit_mma<<<1, 128, kStageBytes + 1024>>>(da.as<bf16>(), db.as<bf16>(), dd.as<float>(), ksteps, wd);
      sync("unit_mma");
      const auto got = download<float>(dd.p, size_t(kBM) * kBN);
      size_t bad = 0;
      std::string first;
      for (int i = 0; i < kBM; ++i)
        for (int j = 0; j < kBN; ++j) {
          double want = 0;
          for (int k = 0; k < 16 * ksteps; ++k) want += double(from_bits(a[i * 64 + k])) * from_bits(b[j * 64 + k]);
          if (double(got[i * kBN + j]) != want) {
            if (bad < 4) first += "[" + std::to_string(i) + "," + std::to_string(j) + ":" + std::to_string(got[i * kBN + j]) + " want " + std::to_string(want) + "]";
            ++bad;
          }
        }
      std::printf("{\"unit\":\"mma_128x256\",\"k\":%d,\"idesc\":\"0x%08x\",\"mismatches\":%zu,\"first\":\"%s\",\"pass\":%s}\n",
                  16 * ksteps, idesc_bf16_f32(kBM, kBN), bad, first.c_str(), bad ? "false" : "true");
      failures += bad != 0;
    }
  }
  // U4, U5: 2-CTA, one MMA (K = 16) and one stage (K = 64), 256 x 256, rows 0-127 in the leader, 128-255 in the peer.
  {
    std::vector<uint16_t> a(256 * 64), b(256 * 64);
    for (auto& v : a) v = to_bits(float(d(rng)));
    for (auto& v : b) v = to_bits(float(d(rng)));
    Device da(a.size() * 2), db(b.size() * 2), dd(size_t(256) * 256 * 4);
    check(cudaMemcpy(da.p, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.p, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
    check(cudaFuncSetAttribute(unit_mma_pair, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kPairStageBytes + 1024)));
    for (int ksteps : {1, 4}) {
      check(cudaMemset(dd.p, 0xff, size_t(256) * 256 * 4));
      unit_mma_pair<<<2, 128, kPairStageBytes + 1024>>>(da.as<bf16>(), db.as<bf16>(), dd.as<float>(), ksteps, wd);
      sync("unit_mma_pair");
      const auto got = download<float>(dd.p, size_t(256) * 256);
      size_t bad = 0;
      std::string first;
      for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j) {
          double want = 0;
          for (int k = 0; k < 16 * ksteps; ++k) want += double(from_bits(a[i * 64 + k])) * from_bits(b[j * 64 + k]);
          if (double(got[i * 256 + j]) != want) {
            if (bad < 4) first += "[" + std::to_string(i) + "," + std::to_string(j) + ":" + std::to_string(got[i * 256 + j]) + " want " + std::to_string(want) + "]";
            ++bad;
          }
        }
      std::printf("{\"unit\":\"mma_2cta_256x256\",\"k\":%d,\"idesc\":\"0x%08x\",\"mismatches\":%zu,\"first\":\"%s\",\"pass\":%s}\n",
                  16 * ksteps, idesc_bf16_f32(256, 256), bad, first.c_str(), bad ? "false" : "true");
      failures += bad != 0;
    }
  }
  std::fflush(stdout);
  return failures ? 1 : 0;
}

// ---- peak mode ------------------------------------------------------------------------------------------------------
static int peak_mode(int iters) {
  tc05::Watchdog wd{g_timeout_ns, g_trap};
  const int smem = 200 * 1024;  // one CTA per SM
  check(cudaFuncSetAttribute(peak_1cta, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaEvent_t e0, e1;
  check(cudaEventCreate(&e0)); check(cudaEventCreate(&e1));
  peak_1cta<<<g_sms, 128, smem>>>(iters / 10, wd);
  sync("peak warm-up");
  std::vector<float> t;
  for (int r = 0; r < 5; ++r) {
    check(cudaEventRecord(e0));
    peak_1cta<<<g_sms, 128, smem>>>(iters, wd);
    check(cudaEventRecord(e1));
    sync("peak");
    float ms; check(cudaEventElapsedTime(&ms, e0, e1));
    t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  const double flops = double(g_sms) * iters * 4 * 2.0 * 128 * 256 * 16;
  std::printf("{\"peak\":\"1cta_128x256x16\",\"ctas\":%d,\"iters\":%d,\"median_ms\":%.4f,\"tflops\":%.2f,\"all_ms\":[%.4f,%.4f,%.4f,%.4f,%.4f]}\n",
              g_sms, iters, t[2], flops / (t[2] * 1e-3) / 1e12, t[0], t[1], t[2], t[3], t[4]);
  check(cudaFuncSetAttribute(peak_2cta, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  peak_2cta<<<g_sms, 128, smem>>>(iters / 10, wd);
  sync("peak 2cta warm-up");
  t.clear();
  for (int r = 0; r < 5; ++r) {
    check(cudaEventRecord(e0));
    peak_2cta<<<g_sms, 128, smem>>>(iters, wd);
    check(cudaEventRecord(e1));
    sync("peak 2cta");
    float ms; check(cudaEventElapsedTime(&ms, e0, e1));
    t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  const double flops2 = double(g_sms / 2) * iters * 4 * 2.0 * 256 * 256 * 16;
  std::printf("{\"peak\":\"2cta_256x256x16\",\"pairs\":%d,\"iters\":%d,\"median_ms\":%.4f,\"tflops\":%.2f,\"all_ms\":[%.4f,%.4f,%.4f,%.4f,%.4f]}\n",
              g_sms / 2, iters, t[2], flops2 / (t[2] * 1e-3) / 1e12, t[0], t[1], t[2], t[3], t[4]);
  std::fflush(stdout);
  return 0;
}

// ---- correctness mode -----------------------------------------------------------------------------------------------
constexpr size_t kGuard = 64 * 1024;
struct Guarded {
  Device buf;
  size_t bytes;
  explicit Guarded(size_t b) : buf(b + 2 * kGuard), bytes(b) {}
  void* c() const { return static_cast<uint8_t*>(buf.p) + kGuard; }
  void fill() {
    check(cudaMemset(buf.p, 0xa5, kGuard));
    check(cudaMemset(c(), 0xff, bytes));
    check(cudaMemset(static_cast<uint8_t*>(c()) + bytes, 0xa5, kGuard));
  }
};

static int correct_mode(int variant, const Options& o) {
  int failed_cases = 0, cases = 0;
  for (const auto& kind : o.data)
    for (const auto& s : kShapes) {
      if (!selected(o, s) || (kind == "split" && !s.vision)) continue;
      for (int M : o.ms) {
        const int N = s.n, K = s.k;
        const size_t MN = size_t(M) * N;
        std::fprintf(stderr, "[correct V%d] %s %s M=%d\n", variant, kind.c_str(), s.name, M);
        Inputs in(kind, M, N, K, case_seed(kind, s, M));
        const auto ha = download<uint16_t>(in.a.p, size_t(M) * K);
        const auto hb = download<uint16_t>(in.b.p, size_t(N) * K);
        // reference (C2) and its CPU check, or the exactness premise (C1)
        std::vector<double> ref, S;
        size_t ref_bad = 0;
        std::vector<size_t> sample(4096);
        {
          std::mt19937_64 r(mix64(case_seed(kind, s, M)));
          for (auto& i : sample) i = r() % MN;
        }
        if (kind != "int") {
          Device dref(MN * 8), dS(MN * 8);
          ref_f64<<<dim3(N / 16, (M + 15) / 16), 256>>>(in.a.as<bf16>(), in.b.as<bf16>(), M, N, K, dref.as<double>(), dS.as<double>());
          sync("ref_f64");
          ref = download<double>(dref.p, MN);
          S = download<double>(dS.p, MN);
          std::atomic<size_t> bad{0};
          parallel_for(sample.size(), [&](size_t t) {
            const size_t i = sample[t];
            const size_t row = i / N, col = i % N;
            long double acc = 0, abs_sum = 0;
            for (int k = 0; k < K; ++k) {
              const long double p = (long double)from_bits(ha[row * K + k]) * from_bits(hb[col * K + k]);
              acc += p;
              abs_sum += std::fabs(p);
            }
            if (std::fabs((long double)ref[i] - acc) > 1e-12L * abs_sum || std::fabs((long double)S[i] - abs_sum) > 1e-12L * abs_sum) ++bad;
          });
          ref_bad = bad;
        }
        // lfm::gemm, both output types
        Device ge32(MN * 4), ge16(MN * 2);
        run_engine(in.a.as<bf16>(), in.b.as<bf16>(), ge32.p, false, M, N, K);
        run_engine(in.a.as<bf16>(), in.b.as<bf16>(), ge16.p, true, M, N, K);
        sync("lfm::gemm");
        const auto g32 = download<float>(ge32.p, MN);
        const auto g16 = download<uint16_t>(ge16.p, MN);
        size_t premise_bad = 0;
        if (kind == "int") {
          std::atomic<size_t> bad{0};
          parallel_for(sample.size(), [&](size_t t) {
            const size_t i = sample[t], row = i / N, col = i % N;
            double acc = 0;
            for (int k = 0; k < K; ++k) acc += double(from_bits(ha[row * K + k])) * from_bits(hb[col * K + k]);
            if (double(g32[i]) != acc || g16[i] != to_bits(float(acc))) ++bad;
          });
          premise_bad = bad;
        }
        std::vector<float> p32;  // probe FP32 output of run 0, for the BF16 bound
        for (int out16 = 0; out16 < 2; ++out16) {
          const size_t es = out16 ? 2 : 4;
          Guarded c(MN * es);
          std::vector<uint8_t> first;
          size_t unwritten = 0, guard_bad = 0, nondet_runs = 0;
          for (int run = 0; run < o.runs; ++run) {
            c.fill();
            const Plan plan = make_plan(variant, out16, in.a.as<bf16>(), in.b.as<bf16>(), c.c(), M, N, K);
            run_plan(plan);
            sync("probe");
            const auto all = download<uint8_t>(c.buf.p, MN * es + 2 * kGuard);
            for (size_t i = 0; i < kGuard; ++i) guard_bad += (all[i] != 0xa5) + (all[kGuard + MN * es + i] != 0xa5);
            const uint8_t* body = all.data() + kGuard;
            if (run == 0) {
              first.assign(body, body + MN * es);
              for (size_t i = 0; i < MN; ++i) {
                bool nan_fill = true;
                for (size_t b = 0; b < es; ++b) nan_fill &= body[i * es + b] == 0xff;
                unwritten += nan_fill;
              }
            } else if (std::memcmp(first.data(), body, MN * es) != 0) ++nondet_runs;
          }
          // C1 / C2
          std::string c12;
          bool pass12 = true;
          if (kind == "int") {
            const void* g = out16 ? static_cast<const void*>(g16.data()) : static_cast<const void*>(g32.data());
            size_t mism = 0;
            for (size_t i = 0; i < MN; ++i) mism += std::memcmp(first.data() + i * es, static_cast<const uint8_t*>(g) + i * es, es) != 0;
            pass12 = mism == 0 && premise_bad == 0;
            c12 = "\"c1\":{\"mismatches_vs_gemm\":" + std::to_string(mism) + ",\"premise_sample\":" + std::to_string(sample.size()) +
                  ",\"premise_mismatches\":" + std::to_string(premise_bad) + "}";
          } else {
            const double kb = double(K) * 0x1p-23;
            size_t viol = 0, gviol = 0, beyond1 = 0, not_rne = 0, szero = 0;
            double pmax = 0, psum = 0, gmax = 0, gsum = 0;
            size_t nrel = 0;
            if (!out16) {
              p32.resize(MN);
              std::memcpy(p32.data(), first.data(), MN * 4);
              for (size_t i = 0; i < MN; ++i) {
                const double e = std::fabs(double(p32[i]) - ref[i]), ge = std::fabs(double(g32[i]) - ref[i]);
                if (S[i] == 0) { szero++; viol += p32[i] != 0; gviol += g32[i] != 0; continue; }
                viol += e > kb * S[i];
                gviol += ge > kb * S[i];
                const double r = e / S[i], gr = ge / S[i];
                pmax = std::max(pmax, r); psum += r; gmax = std::max(gmax, gr); gsum += gr; ++nrel;
              }
              const double pmean = nrel ? psum / nrel : 0, gmean = nrel ? gsum / nrel : 0;
              pass12 = viol == 0 && ref_bad == 0 && pmax <= 2 * gmax && pmean <= 1.25 * gmean;
              char b[640];
              std::snprintf(b, sizeof b,
                            "\"c2\":{\"ref_cpu_sample\":%zu,\"ref_cpu_mismatches\":%zu,\"bound_violations\":%zu,\"gemm_bound_violations\":%zu,"
                            "\"s_zero\":%zu,\"max_rel\":%.6g,\"gemm_max_rel\":%.6g,\"mean_rel\":%.6g,\"gemm_mean_rel\":%.6g,"
                            "\"max_ratio\":%.4f,\"mean_ratio\":%.4f,\"identical_to_gemm\":%.6f}",
                            sample.size(), ref_bad, viol, gviol, szero, pmax, gmax, pmean, gmean, gmax ? pmax / gmax : 0.0,
                            gmean ? pmean / gmean : 0.0,
                            [&] { size_t same = 0; for (size_t i = 0; i < MN; ++i) same += std::memcmp(&p32[i], &g32[i], 4) == 0; return double(same) / MN; }());
              c12 = b;
            } else {
              const uint16_t* p16 = reinterpret_cast<const uint16_t*>(first.data());
              double umax = 0, usum = 0, gumax = 0, gusum = 0;
              for (size_t i = 0; i < MN; ++i) {
                const double v = from_bits(p16[i]), gv = from_bits(g16[i]);
                const double e = std::fabs(v - ref[i]);
                viol += e > kb * S[i] + 0.5 * ulp_bf16(p32.empty() ? v : p32[i]);
                if (!p32.empty()) not_rne += p16[i] != to_bits(p32[i]);
                beyond1 += std::abs(ordered(p16[i]) - ordered(g16[i])) > 1;
                const double u = e / ulp_bf16(ref[i]), gu = std::fabs(gv - ref[i]) / ulp_bf16(ref[i]);
                umax = std::max(umax, u); usum += u; gumax = std::max(gumax, gu); gusum += gu;
              }
              pass12 = viol == 0 && ref_bad == 0 && beyond1 == 0 && umax <= std::max(0.55, 1.1 * gumax);
              char b[640];
              std::snprintf(b, sizeof b,
                            "\"c2\":{\"ref_cpu_sample\":%zu,\"ref_cpu_mismatches\":%zu,\"bound_violations\":%zu,\"max_ulp\":%.4f,"
                            "\"gemm_max_ulp\":%.4f,\"mean_ulp\":%.5f,\"gemm_mean_ulp\":%.5f,\"beyond_1ulp_vs_gemm\":%zu,"
                            "\"identical_to_gemm\":%.6f,\"not_rne_of_probe_f32\":%zu}",
                            sample.size(), ref_bad, viol, umax, gumax, usum / MN, gusum / MN, beyond1,
                            [&] { size_t same = 0; for (size_t i = 0; i < MN; ++i) same += p16[i] == g16[i]; return double(same) / MN; }(),
                            not_rne);
              c12 = b;
            }
          }
          if (kind == "int" && !out16) { p32.resize(MN); std::memcpy(p32.data(), first.data(), MN * 4); }
          const bool c3 = unwritten == 0 && guard_bad == 0, c4 = nondet_runs == 0;
          const bool pass = pass12 && c3 && c4;
          ++cases;
          failed_cases += !pass;
          std::printf("{\"variant\":%d,\"data\":\"%s\",\"shape\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"out\":\"%s\",%s,"
                      "\"c3\":{\"unwritten\":%zu,\"guard_bytes_changed\":%zu},\"c4\":{\"runs\":%d,\"differing_runs\":%zu},"
                      "\"c5\":{\"trap\":false},\"pass\":{\"c12\":%s,\"c3\":%s,\"c4\":%s,\"all\":%s}}\n",
                      variant, kind.c_str(), s.name, M, N, K, out16 ? "bf16" : "f32", c12.c_str(), unwritten, guard_bad,
                      o.runs, nondet_runs, pass12 ? "true" : "false", c3 ? "true" : "false", c4 ? "true" : "false",
                      pass ? "true" : "false");
          std::fflush(stdout);
        }
      }
    }
  std::printf("{\"summary\":\"correct\",\"variant\":%d,\"cases\":%d,\"failed\":%d}\n", variant, cases, failed_cases);
  return failed_cases ? 1 : 0;
}

// ---- sanitizer mode: the probe alone on the sanitizer case set, outputs written raw ---------------------------------
static int sanitize_mode(int variant, const std::string& dir, const Options& o) {
  const char* names[] = {"v.w2", "lm.w1w3"};
  for (const char* name : names) {
    const Shape* s = nullptr;
    for (auto& x : kShapes) if (!std::strcmp(x.name, name)) s = &x;
    const bool out16 = !s->vision;
    for (const auto& kind : o.data) {
      if (kind == "split" && !s->vision) continue;
      for (int M : o.ms) {
        Inputs in(kind, M, s->n, s->k, case_seed(kind, *s, M));
        const size_t bytes = size_t(M) * s->n * (out16 ? 2 : 4);
        Device c(bytes);  // deliberately not initialised: initcheck sees any element the probe fails to write
        const Plan plan = make_plan(variant, out16, in.a.as<bf16>(), in.b.as<bf16>(), c.p, M, s->n, s->k);
        run_plan(plan);
        sync("probe");
        const auto h = download<uint8_t>(c.p, bytes);
        const std::string path = dir + "/" + name + "." + kind + ".m" + std::to_string(M) + (out16 ? ".bf16" : ".f32");
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(h.data()), h.size());
        if (!f) throw std::runtime_error("write " + path);
        std::printf("{\"sanitize_case\":\"%s\",\"bytes\":%zu}\n", path.c_str(), bytes);
      }
    }
  }
  std::fflush(stdout);
  return 0;
}

// ---- speed mode -----------------------------------------------------------------------------------------------------
template <class F> static double median_ms(F&& f, int reps) {
  cudaEvent_t a, b;
  check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
  for (int i = 0; i < 3; ++i) f();
  sync("warm-up");
  std::vector<float> t;
  for (int i = 0; i < reps; ++i) {
    check(cudaEventRecord(a));
    f();
    check(cudaEventRecord(b));
    check(cudaEventSynchronize(b), "timed run");
    float ms; check(cudaEventElapsedTime(&ms, a, b));
    t.push_back(ms);
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}
static int speed_mode(const std::vector<int>& variants, const Options& o) {
  cublasHandle_t h;
  check(cublasCreate(&h));
  for (const auto& s : kShapes) {
    if (!selected(o, s)) continue;
    const bool out16 = !s.vision;  // the listed output type
    for (int M : o.ms) {
      const int N = s.n, K = s.k;
      Inputs in("normal", M, N, K, case_seed("normal", s, M));
      Device c(size_t(M) * N * 4);
      const double flop = 2.0 * M * N * K;
      std::string rec = "{\"shape\":\"" + std::string(s.name) + "\",\"m\":" + std::to_string(M) + ",\"n\":" + std::to_string(N) +
                        ",\"k\":" + std::to_string(K) + ",\"out\":\"" + (out16 ? "bf16" : "f32") + "\"";
      auto add = [&](const std::string& name, double ms) {
        char b[160];
        std::snprintf(b, sizeof b, ",\"%s\":{\"ms\":%.5f,\"tflops\":%.2f}", name.c_str(), ms, flop / (ms * 1e-3) / 1e12);
        rec += b;
      };
      for (int v : variants) {
        const Plan plan = make_plan(v, out16, in.a.as<bf16>(), in.b.as<bf16>(), c.p, M, N, K);
        add("V" + std::to_string(v), median_ms([&] { run_plan(plan); }, o.reps));
      }
      add("lfm_gemm", median_ms([&] { run_engine(in.a.as<bf16>(), in.b.as<bf16>(), c.p, out16, M, N, K); }, o.reps));
      add("cublas", median_ms([&] {
            const float alpha = 1, beta = 0;
            check(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, in.b.p, CUDA_R_16BF, K, in.a.p, CUDA_R_16BF, K,
                               &beta, c.p, out16 ? CUDA_R_16BF : CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
          }, o.reps));
      std::printf("%s}\n", rec.c_str());
      std::fflush(stdout);
    }
  }
  cublasDestroy(h);
  return 0;
}

int main(int argc, char** argv) try {
  if (argc < 2) throw std::runtime_error("usage: tcgen05-gemm-probe unit|peak|correct|sanitize|speed ...");
  check(cudaSetDeviceFlags(cudaDeviceMapHost));
  check(cudaHostAlloc(reinterpret_cast<void**>(&g_trap_host), sizeof(int), cudaHostAllocMapped));
  *g_trap_host = 0;
  check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&g_trap), g_trap_host, 0));
  check(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0));
  const std::string mode = argv[1];
  int rc = 0;
  if (mode == "unit") { parse(argc, argv, 2); rc = unit_mode(); }
  else if (mode == "peak") rc = peak_mode(argc > 2 && argv[2][0] != '-' ? std::stoi(argv[2]) : 100000);
  else if (mode == "correct") { if (argc < 3) throw std::runtime_error("correct VARIANT"); rc = correct_mode(std::stoi(argv[2]), parse(argc, argv, 3)); }
  else if (mode == "sanitize") { if (argc < 4) throw std::runtime_error("sanitize VARIANT OUTDIR"); rc = sanitize_mode(std::stoi(argv[2]), argv[3], parse(argc, argv, 4)); }
  else if (mode == "speed") {
    if (argc < 3) throw std::runtime_error("speed VARIANTS");
    std::vector<int> v;
    for (auto& t : split_list(argv[2])) v.push_back(std::stoi(t));
    rc = speed_mode(v, parse(argc, argv, 3));
  } else throw std::runtime_error("unknown mode " + mode);
  return rc;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 2;
}
