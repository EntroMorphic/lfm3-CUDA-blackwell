#pragma once
// tcgen05 BF16 GEMM for Thor (sm_110a): TMA loads, tcgen05.mma with the accumulator in tensor memory (TMEM).
// A probe (validation/tcgen05-probe-v1), not engine code. Compile only with
//   -gencode arch=compute_110a,code=sm_110a      (or compute_110f/sm_110f)
// `-arch=sm_110` rejects every tcgen05 instruction, and the shorthand `-arch=sm_110a` also emits compute_110 PTX,
// which fails the same way.
//
//   C (M x N) = A (M x K, K contiguous) · Bᵀ,  B stored N x K (K contiguous),  FP32 accumulate,
//   C FP32 or BF16 (round to nearest even), row-major with leading dimension ldc.
//   Contract: M >= 1, N % 256 == 0, K % 64 == 0, A, B, C 16-byte aligned, ldc % 8 == 0.
//
// ---------------------------------------------------------------------------------------------------------------
// The encodings below are the reusable part. Each is proven on Thor by the probe's `unit` tests
// (tcgen05_gemm_probe.cu): one MMA, one K = 64 stage, and the TMA swizzle, each bit-exact with integer inputs.
//
// 1. Shared-memory operand layout: K-major, 128-byte swizzle (what TMA writes with CU_TENSOR_MAP_SWIZZLE_128B and a
//    box of 64 BF16 (128 bytes) x R rows). Row r occupies bytes [128 r, 128 r + 128); its 16-byte chunk j (BF16
//    elements 8j..8j+7) sits at chunk position j ^ (r % 8). The tile base must be 1024-byte aligned (the XOR uses
//    address bits 7-9, so the pattern repeats every 8 rows = 1024 bytes).
//
// 2. Shared-memory matrix descriptor (64 bits, PTX ISA "tcgen05 shared memory descriptor"):
//      bits  0-13  start address >> 4
//      bits 16-29  leading-dimension byte offset >> 4  (unused for swizzled K-major; set to 1)
//      bits 32-45  stride-dimension byte offset >> 4   (1024: from one 8-row group to the next)
//      bits 46-48  fixed 0b001 (tcgen05 descriptor version)
//      bits 49-51  base offset (0: tiles 1024-aligned)
//      bit  52     leading-dimension mode (0: relative)
//      bits 61-63  swizzle: 0 none, 1 128B with 32B atom, 2 128B, 4 64B, 6 32B   -> 2
//    Stepping K by 16 elements (one MMA) inside the 128-byte row: start address + 32 bytes (descriptor + 2); the
//    hardware applies the swizzle to the absolute address, so this is correct on a 1024-aligned tile.
//
// 3. Instruction descriptor (32 bits) for tcgen05.mma .kind::f16:
//      bits  0-1 sparsity selector 0, bit 2 sparse 0, bit 3 saturate 0
//      bits  4-5  D format: 1 = F32
//      bits  7-9  A format: 1 = BF16          bits 10-12  B format: 1 = BF16
//      bit  13/14 negate A/B: 0               bit 15/16   A/B major: 0 = K-major
//      bits 17-22 N >> 3                      bits 24-28  M >> 4
//      bits 30-31 max shift (.ws only): 0
//
// 4. TMEM: 128 lanes x 512 columns of 32 bits per SM; an address is (lane << 16) | column. For M = 128 with
//    cta_group::1, row i of D is lane i and column j is column base + j (FP32). A warp may access only lanes
//    32 (warp_id % 4) .. +31, so the four epilogue warps cover the four lane quarters by warp_id % 4.
//    tcgen05.ld .32x32b.xN: thread t of the warp reads lane (32 q + t), N consecutive columns.
//
// 5. Pipeline (mbarriers in shared memory): per stage `full` (1 arrival + TMA bytes) and `empty` (1 arrival, from
//    tcgen05.commit); per accumulator `acc_full` (tcgen05.commit) and `acc_empty` (4 epilogue warps). A fresh barrier
//    is in phase 0; waiting for parity 1 on it returns at once ("the previous phase is complete"), which starts the
//    producer without a pre-arrival.
// ---------------------------------------------------------------------------------------------------------------
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace tc05 {

using bf16 = __nv_bfloat16;

// ---- encodings -------------------------------------------------------------------------------------------------
__host__ __device__ constexpr uint32_t idesc_bf16_f32(int m, int n) {
  return (1u << 4)                     // D: F32
       | (1u << 7)                     // A: BF16
       | (1u << 10)                    // B: BF16
       | (uint32_t(n >> 3) << 17)      // N >> 3
       | (uint32_t(m >> 4) << 24);     // M >> 4
}
__host__ __device__ constexpr uint64_t smem_desc_sw128(uint32_t smem_addr) {
  return uint64_t((smem_addr & 0x3FFFFu) >> 4)   // start address
       | (uint64_t(1) << 16)                     // LBO (unused)
       | (uint64_t(1024 >> 4) << 32)             // SBO: 1024 bytes per 8-row group
       | (uint64_t(1) << 46)                     // version 0b001
       | (uint64_t(2) << 61);                    // SWIZZLE_128B
}
// Byte offset of BF16 element (row, k) in a 128-byte-swizzled K-major tile of 64-element rows.
__host__ __device__ constexpr uint32_t sw128_offset(uint32_t row, uint32_t k) {
  return row * 128 + (((k >> 3) ^ (row & 7)) << 4) + (k & 7) * 2;
}

// ---- PTX wrappers ----------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}

// Bounded waits: past `timeout_ns` of globaltimer the waiter records `tag` in host-mapped memory and traps, so a
// protocol error ends as a launch error rather than a hung GPU.
struct Watchdog {
  unsigned long long timeout_ns = 2000000000ull;
  int* trap_tag = nullptr;  // host-mapped
};
__device__ __noinline__ void watchdog_fire(const Watchdog& w, int tag) {
  if (w.trap_tag) { atomicExch(w.trap_tag, tag); __threadfence_system(); }
  __trap();
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity, const Watchdog& w, int tag) {
  if (mbar_try_wait(bar, parity)) return;
  const uint64_t t0 = globaltimer();
  while (!mbar_try_wait(bar, parity))
    if (globaltimer() - t0 > w.timeout_ns) watchdog_fire(w, tag);
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void prefetch_tensormap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}

// TMEM allocation: one whole warp; the column count a power of two >= 32.
__device__ __forceinline__ void tmem_alloc(uint32_t slot_smem, uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(slot_smem), "r"(cols) : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t tmem, uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(cols) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// D[tmem] (+)= A[adesc] · B[bdesc]ᵀ, one instruction (K = 16 for BF16). Single thread.
__device__ __forceinline__ void mma_bf16(uint32_t tmem_d, uint64_t adesc, uint64_t bdesc, uint32_t idesc, bool accumulate) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
      ::"r"(tmem_d), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(uint32_t(accumulate))
      : "memory");
}
// Arrive once on `bar` when every tcgen05 operation this thread issued before has completed.
__device__ __forceinline__ void mma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

// ---- the GEMM ------------------------------------------------------------------------------------------------------
inline constexpr int kBM = 128, kBN = 256, kBK = 64;
inline constexpr uint32_t kABytes = kBM * kBK * 2, kBBytes = kBN * kBK * 2, kStageBytes = kABytes + kBBytes;

struct Args {
  int m = 0, n = 0, k = 0;
  void* c = nullptr;
  long long ldc = 0;
  int tiles_m = 0, tiles_n = 0, tiles = 0;
  int band = 1;  // N tiles per raster band (L2 reuse of B while walking M)
  Watchdog wd;
};

// Tile t -> (M tile, N tile): bands of `band` N tiles; inside a band, N fastest then M, so the CTAs in flight share
// a few A row blocks and a few B column blocks.
__device__ __forceinline__ void tile_coords(int t, const Args& a, int& mt, int& nt) {
  const int per_band = a.band * a.tiles_m;
  const int band = t / per_band, idx = t % per_band;
  const int width = min(a.band, a.tiles_n - band * a.band);
  nt = band * a.band + idx % width;
  mt = idx / width;
}

template <bool BF16_OUT>
__device__ __forceinline__ void store_row32(const Args& a, int row, int col, const uint32_t (&r)[32]) {
  if constexpr (BF16_OUT) {
    uint32_t p[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const __nv_bfloat162 h = __floats2bfloat162_rn(__uint_as_float(r[2 * j]), __uint_as_float(r[2 * j + 1]));
      p[j] = *reinterpret_cast<const uint32_t*>(&h);
    }
    uint4* out = reinterpret_cast<uint4*>(static_cast<bf16*>(a.c) + row * a.ldc + col);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[j] = make_uint4(p[4 * j], p[4 * j + 1], p[4 * j + 2], p[4 * j + 3]);
  } else {
    uint4* out = reinterpret_cast<uint4*>(static_cast<float*>(a.c) + row * a.ldc + col);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = make_uint4(r[4 * j], r[4 * j + 1], r[4 * j + 2], r[4 * j + 3]);
  }
}

// 1-CTA kernel, 192 threads: warp 0 TMA producer (and TMEM owner), warp 1 MMA issuer, warps 2-5 epilogue.
// ACC = 1: one tile per CTA (V1, grid = tiles). ACC = 2: persistent, two TMEM accumulators so the epilogue of one
// tile overlaps the MMAs of the next (V2, grid = SMs).
// Watchdog tags: 1 producer/empty, 2 MMA/full, 3 MMA/acc_empty, 4 epilogue/acc_full.
template <int STAGES, int ACC, bool BF16_OUT>
__global__ void __launch_bounds__(192, 1)
gemm_1cta(const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b, const Args args) {
  extern __shared__ uint8_t dsmem[];
  __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES], acc_full[ACC], acc_empty[ACC];
  __shared__ uint32_t tmem_slot;
  const uint32_t base = (smem_u32(dsmem) + 1023u) & ~1023u;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  constexpr uint32_t kCols = 256 * ACC;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) { mbar_init(smem_u32(&full[s]), 1); mbar_init(smem_u32(&empty[s]), 1); }
#pragma unroll
    for (int a = 0; a < ACC; ++a) { mbar_init(smem_u32(&acc_full[a]), 1); mbar_init(smem_u32(&acc_empty[a]), 4); }
    fence_mbar_init();
    prefetch_tensormap(&map_a);
    prefetch_tensormap(&map_b);
  }
  if (warp == 0) { tmem_alloc(smem_u32(&tmem_slot), kCols); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = tmem_slot;
  const int ktiles = args.k / kBK;
  const int first = blockIdx.x, step = ACC == 1 ? args.tiles : gridDim.x;

  if (warp == 0) {
    if (lane == 0) {
      int it = 0;
      for (int t = first; t < args.tiles; t += step) {
        int mt, nt;
        tile_coords(t, args, mt, nt);
        for (int kt = 0; kt < ktiles; ++kt, ++it) {
          const int s = it % STAGES;
          const uint32_t ph = (it / STAGES) & 1;
          mbar_wait(smem_u32(&empty[s]), ph ^ 1, args.wd, 1);
          const uint32_t fb = smem_u32(&full[s]), sa = base + s * kStageBytes;
          mbar_expect_tx(fb, kStageBytes);
          tma_load_2d(sa, &map_a, kt * kBK, mt * kBM, fb);
          tma_load_2d(sa + kABytes, &map_b, kt * kBK, nt * kBN, fb);
        }
      }
    }
    __syncwarp();
  } else if (warp == 1) {
    if (lane == 0) {
      constexpr uint32_t idesc = idesc_bf16_f32(kBM, kBN);
      int it = 0, local = 0;
      for (int t = first; t < args.tiles; t += step, ++local) {
        const int a = local % ACC;
        const uint32_t aph = (local / ACC) & 1;
        mbar_wait(smem_u32(&acc_empty[a]), aph ^ 1, args.wd, 3);
        tc_fence_after();
        const uint32_t d = tmem + a * 256;
        for (int kt = 0; kt < ktiles; ++kt, ++it) {
          const int s = it % STAGES;
          const uint32_t ph = (it / STAGES) & 1;
          mbar_wait(smem_u32(&full[s]), ph, args.wd, 2);
          tc_fence_after();
          const uint32_t sa = base + s * kStageBytes;
          const uint64_t ad = smem_desc_sw128(sa), bd = smem_desc_sw128(sa + kABytes);
#pragma unroll
          for (int kk = 0; kk < kBK / 16; ++kk) mma_bf16(d, ad + 2 * kk, bd + 2 * kk, idesc, (kt | kk) != 0);
          mma_commit(smem_u32(&empty[s]));
        }
        mma_commit(smem_u32(&acc_full[a]));
      }
    }
    __syncwarp();
  } else {
    const int q = warp % 4;
    int local = 0;
    for (int t = first; t < args.tiles; t += step, ++local) {
      int mt, nt;
      tile_coords(t, args, mt, nt);
      const int a = local % ACC;
      const uint32_t aph = (local / ACC) & 1;
      mbar_wait(smem_u32(&acc_full[a]), aph, args.wd, 4);
      tc_fence_after();
      const int row = mt * kBM + q * 32 + lane;
      const uint32_t taddr = tmem + a * 256 + (uint32_t(q * 32) << 16);
#pragma unroll 1
      for (int c = 0; c < kBN / 32; ++c) {
        uint32_t r[32];
        tmem_ld32(taddr + c * 32, r);
        tmem_ld_wait();
        if (c == kBN / 32 - 1) {
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(smem_u32(&acc_empty[a]));
        }
        if (row < args.m) store_row32<BF16_OUT>(args, row, nt * kBN + c * 32, r);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) {
    tc_fence_after();
    tmem_dealloc(tmem, kCols);
  }
}

// ---- 2-CTA (cta_group::2) ------------------------------------------------------------------------------------------
// 6. A CTA pair (cluster of 2, ranks 0 = leader and 1 = peer) computes a 256 x 256 tile with one MMA stream issued by
//    the leader: `tcgen05.mma.cta_group::2` with M = 256 reads A rows 0-127 and B rows 0-127 from the leader's shared
//    memory and A rows 128-255 and B rows 128-255 from the peer's, at the same shared-memory offsets (the descriptors
//    hold the leader's offsets). D rows 0-127 land in the leader's TMEM and rows 128-255 in the peer's, each 128 lanes
//    x 256 columns, laid out as in note 4. Each CTA TMA-loads its own halves; with `.cta_group::2` the completion is
//    signalled on the leader's `full` barrier (its address in the peer obtained by `mapa`), so the leader's producer
//    expects the bytes of both CTAs. `tcgen05.commit ... multicast::cluster` with mask 0b11 arrives on the barrier at
//    the same offset in both CTAs (`empty`, `acc_full`). The epilogue warps of both CTAs arrive on the leader's
//    `acc_empty` (count 8). TMEM is allocated and freed with `.cta_group::2` by one warp in each CTA, and only after
//    a cluster barrier: the allocation's SASS (UTCATOMSWS.2CTA, then a remote mbarrier arrive and STAS into the
//    peer) touches the peer CTA's reserved shared memory, which the peer initialises when it starts. Allocating
//    before the barrier is a race (racecheck, build b3: 15 reports, write in the peer's prologue, read at the
//    remote arrive); C1-C5 did not see it.
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t map_to_rank(uint32_t smem_addr, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(smem_addr), "r"(rank));
  return out;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ bool mbar_try_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity, const Watchdog& w, int tag) {
  if (mbar_try_wait_cluster(bar, parity)) return;
  const uint64_t t0 = globaltimer();
  while (!mbar_try_wait_cluster(bar, parity))
    if (globaltimer() - t0 > w.timeout_ns) watchdog_fire(w, tag);
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t cluster_bar) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_bar) : "memory");
}
__device__ __forceinline__ void tma_load_2d_pair(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t leader_bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(leader_bar)
      : "memory");
}
__device__ __forceinline__ void tmem_alloc_pair(uint32_t slot_smem, uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(slot_smem), "r"(cols) : "memory");
}
__device__ __forceinline__ void tmem_relinquish_pair() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc_pair(uint32_t tmem, uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(cols) : "memory");
}
__device__ __forceinline__ void mma_bf16_pair(uint32_t tmem_d, uint64_t adesc, uint64_t bdesc, uint32_t idesc, bool accumulate) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
      ::"r"(tmem_d), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(uint32_t(accumulate))
      : "memory");
}
__device__ __forceinline__ void mma_commit_pair(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
               ::"r"(bar), "h"(uint16_t(3))
               : "memory");
}

inline constexpr int kPairBM = 256;                           // rows per pair tile (128 per CTA)
inline constexpr uint32_t kPairHalfB = 128 * kBK * 2;         // each CTA's half of the B tile
inline constexpr uint32_t kPairStageBytes = kABytes + kPairHalfB;  // per CTA: A 16 KB + B half 16 KB

// V3: persistent CTA pairs, two TMEM accumulators, tiles 256 x 256 x 64. Grid = 2 x pairs, cluster (2, 1, 1).
// Args.tiles_m counts 256-row tiles. Watchdog tags as gemm_1cta, plus 10 + tag in the peer.
template <int STAGES, bool BF16_OUT>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(192, 1)
gemm_2cta(const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b, const Args args) {
  constexpr int ACC = 2;
  extern __shared__ uint8_t dsmem[];
  __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES], acc_full[ACC], acc_empty[ACC];
  __shared__ uint32_t tmem_slot;
  const uint32_t base = (smem_u32(dsmem) + 1023u) & ~1023u;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t rank = cluster_rank();
  const bool leader = rank == 0;
  constexpr uint32_t kCols = 256 * ACC;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) { mbar_init(smem_u32(&full[s]), 1); mbar_init(smem_u32(&empty[s]), 1); }
#pragma unroll
    for (int a = 0; a < ACC; ++a) { mbar_init(smem_u32(&acc_full[a]), 1); mbar_init(smem_u32(&acc_empty[a]), 8); }
    fence_mbar_init();
    prefetch_tensormap(&map_a);
    prefetch_tensormap(&map_b);
  }
  // Both CTAs must have started (their reserved allocator state initialised) before the pair allocation, whose
  // expansion arrives on and stores into the peer's shared memory: racecheck found this (b3), see note 6.
  cluster_sync();
  if (warp == 0) { tmem_alloc_pair(smem_u32(&tmem_slot), kCols); tmem_relinquish_pair(); }
  tc_fence_before();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = tmem_slot;
  const int ktiles = args.k / kBK;
  const int pair = blockIdx.x / 2, pairs = gridDim.x / 2;
  const int tag = leader ? 0 : 10;

  if (warp == 0) {
    if (lane == 0) {
      int it = 0;
      for (int t = pair; t < args.tiles; t += pairs) {
        int mt, nt;
        tile_coords(t, args, mt, nt);
        for (int kt = 0; kt < ktiles; ++kt, ++it) {
          const int s = it % STAGES;
          const uint32_t ph = (it / STAGES) & 1;
          mbar_wait_cluster(smem_u32(&empty[s]), ph ^ 1, args.wd, tag + 1);
          const uint32_t fb = map_to_rank(smem_u32(&full[s]), 0), sa = base + s * kPairStageBytes;
          if (leader) mbar_expect_tx(smem_u32(&full[s]), 2 * kPairStageBytes);
          tma_load_2d_pair(sa, &map_a, kt * kBK, mt * kPairBM + int(rank) * kBM, fb);
          tma_load_2d_pair(sa + kABytes, &map_b, kt * kBK, nt * kBN + int(rank) * 128, fb);
        }
      }
    }
    __syncwarp();
  } else if (warp == 1) {
    if (leader && lane == 0) {
      constexpr uint32_t idesc = idesc_bf16_f32(kPairBM, kBN);
      int it = 0, local = 0;
      for (int t = pair; t < args.tiles; t += pairs, ++local) {
        const int a = local % ACC;
        const uint32_t aph = (local / ACC) & 1;
        mbar_wait_cluster(smem_u32(&acc_empty[a]), aph ^ 1, args.wd, 3);
        tc_fence_after();
        const uint32_t d = tmem + a * 256;
        for (int kt = 0; kt < ktiles; ++kt, ++it) {
          const int s = it % STAGES;
          const uint32_t ph = (it / STAGES) & 1;
          mbar_wait_cluster(smem_u32(&full[s]), ph, args.wd, 2);
          tc_fence_after();
          const uint32_t sa = base + s * kPairStageBytes;
          const uint64_t ad = smem_desc_sw128(sa), bd = smem_desc_sw128(sa + kABytes);
#pragma unroll
          for (int kk = 0; kk < kBK / 16; ++kk) mma_bf16_pair(d, ad + 2 * kk, bd + 2 * kk, idesc, (kt | kk) != 0);
          mma_commit_pair(smem_u32(&empty[s]));
        }
        mma_commit_pair(smem_u32(&acc_full[a]));
      }
    }
    __syncwarp();
  } else {
    const int q = warp % 4;
    int local = 0;
    for (int t = pair; t < args.tiles; t += pairs, ++local) {
      int mt, nt;
      tile_coords(t, args, mt, nt);
      const int a = local % ACC;
      const uint32_t aph = (local / ACC) & 1;
      mbar_wait_cluster(smem_u32(&acc_full[a]), aph, args.wd, tag + 4);
      tc_fence_after();
      const int row = mt * kPairBM + int(rank) * kBM + q * 32 + lane;
      const uint32_t taddr = tmem + a * 256 + (uint32_t(q * 32) << 16);
#pragma unroll 1
      for (int c = 0; c < kBN / 32; ++c) {
        uint32_t r[32];
        tmem_ld32(taddr + c * 32, r);
        tmem_ld_wait();
        if (c == kBN / 32 - 1) {
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive_remote(map_to_rank(smem_u32(&acc_empty[a]), 0));
        }
        if (row < args.m) store_row32<BF16_OUT>(args, row, nt * kBN + c * 32, r);
      }
    }
  }
  tc_fence_before();
  cluster_sync();
  if (warp == 0) {
    tc_fence_after();
    tmem_dealloc_pair(tmem, kCols);
  }
}

}  // namespace tc05
