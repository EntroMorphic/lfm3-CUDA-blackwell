#pragma once
#define LFM3_THIN_POLICY 8
// Isolated candidate only. Production never includes this header.
// Same as lfm3_wmma_runtime_candidate.cuh (v4) except the thin (rows<=8) producer:
// v6 (shared-memory staged, 16-byte row-contiguous loads), or v7 (same loads, quad
// rotation transpose) or v8 (butterfly transpose) -- both without shared memory -- per
// LFM3_THIN_POLICY, for large weights and
// v5 (register-prefetched) otherwise. Both are bitwise-identical to v4's thin
// producer (validation/thin-probe-v1); the choice affects speed, never arithmetic.
#include <mma.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>

namespace lfm3_wmma_candidate {
using bf16=__nv_bfloat16;
__global__ void wmma_partials(const bf16* w,const bf16* x,float* partial,int m,int n,int k,int padded_n,int padded_m,int parts){
  namespace wm=nvcuda::wmma;
  __shared__ __align__(32) bf16 left[256],right[256];
  int lane=int(threadIdx.x),row=int(blockIdx.x)*16,col=int(blockIdx.y)*16;
  int start=int(blockIdx.z)*(k/parts),end=start+k/parts;
  const bool full_left=row+16<=m,full_right=col+16<=n;
  wm::fragment<wm::matrix_a,16,16,16,bf16,wm::row_major> a;
  wm::fragment<wm::matrix_b,16,16,16,bf16,wm::col_major> b;
  wm::fragment<wm::accumulator,16,16,16,float> sum;
  wm::fill_fragment(sum,0.0f);
  for(int base=start;base<end;base+=16){
    // Load complete aligned tiles directly into fragments. Shared memory is
    // needed only for input-tail padding, not as a compulsory staging copy.
    if(!full_left||!full_right){
      for(int i=lane;i<256;i+=32){
        int wr=row+i/16,wk=base+i%16,xc=col+i/16,xk=base+i%16;
        if(!full_left)left[i]=wr<m?w[size_t(wr)*k+wk]:__float2bfloat16_rn(0.0f);
        if(!full_right)right[i]=xc<n?x[size_t(xc)*k+xk]:__float2bfloat16_rn(0.0f);
      }
      __syncwarp();
    }
    if(full_left)wm::load_matrix_sync(a,w+size_t(row)*k+base,k);
    else wm::load_matrix_sync(a,left,16);
    if(full_right)wm::load_matrix_sync(b,x+size_t(col)*k+base,k);
    else wm::load_matrix_sync(b,right,16);
    wm::mma_sync(sum,a,b,sum);__syncwarp();
  }
  wm::store_matrix_sync(partial+size_t(blockIdx.z)*padded_m*padded_n+size_t(row)*padded_n+col,sum,padded_n,wm::mem_row_major);
}
__device__ __forceinline__ uint32_t bf16_pair(const bf16* p){
  // cudaMalloc alignment, K/partition multiples of16, and even BF16 pair
  // offsets guarantee four-byte alignment. Preserve the raw operand bits.
  uint32_t bits;asm volatile("ld.global.u32 %0, [%1];":"=r"(bits):"l"(p));return bits;
}
// PTX m16n8k16 row/column fragment layout. Each warp owns sixteen output rows;
// four independent warps share a block without shared memory or CTA barriers.
__global__ void mma_thin_partials(const bf16* w,const bf16* x,float* partial,int m,int n,int k,int padded_n,int padded_m,int parts){
  int lane=int(threadIdx.x)&31,warp=int(threadIdx.x)/32;
  int row=(int(blockIdx.x)*(int(blockDim.x)/32)+warp)*16;
  if(row>=m)return; // Entire warp retires; no block-wide synchronization follows.
  int group=lane/4,t=lane%4,start=int(blockIdx.z)*(k/parts),end=start+k/parts;
  float c0=0,c1=0,c2=0,c3=0;
  for(int base=start;base<end;base+=16){
    int offset=base+2*t;
    uint32_t a0=row+group<m?bf16_pair(w+size_t(row+group)*k+offset):0;
    uint32_t a1=row+group+8<m?bf16_pair(w+size_t(row+group+8)*k+offset):0;
    uint32_t a2=row+group<m?bf16_pair(w+size_t(row+group)*k+offset+8):0;
    uint32_t a3=row+group+8<m?bf16_pair(w+size_t(row+group+8)*k+offset+8):0;
    uint32_t b0=group<n?bf16_pair(x+size_t(group)*k+offset):0;
    uint32_t b1=group<n?bf16_pair(x+size_t(group)*k+offset+8):0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  size_t head=size_t(blockIdx.z)*padded_m*padded_n;
  if(row+group<m){size_t offset=head+size_t(row+group)*padded_n+2*t;partial[offset]=c0;partial[offset+1]=c1;}
  if(row+group+8<m){size_t offset=head+size_t(row+group+8)*padded_n+2*t;partial[offset]=c2;partial[offset+1]=c3;}
}
__global__ void wmma_finish(const float* p,bf16* out,size_t count,int m,int n,int padded_m,int padded_n,int parts){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  int row=int(i%m),col=int(i/m);float sum=p[size_t(row)*padded_n+col];
  for(int part=1;part<parts;++part)sum+=p[size_t(part)*padded_m*padded_n+size_t(row)*padded_n+col];
  out[i]=__float2bfloat16_rn(sum);
}
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

// v7: v6's coalescing without shared memory. For each row, the four lanes of a quad
// (t=0..3) load consecutive 16-byte vectors (64 contiguous bytes per quad per row). A
// 16-byte vector j holds one 4-byte BF16 pair for each t, so a 4x4 word transpose inside
// the quad (three shuffle rotations plus register selects) yields exactly v4's fragment
// words. Same MMA sequence and K order as v4; registers only (nothing for racecheck to
// track beyond v4/v5); next stage's loads are in flight during the current stage's MMAs.
namespace v7 {
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t pick(uint4 v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int S>
__global__ void __launch_bounds__(128) mma_thin_partials(const bf16* w, const bf16* x, float* partial, int m, int n,
                                                         int k, int padded_n, int padded_m, int parts) {
  static_assert(S % 2 == 0 && S >= 2, "a stage holds 2*S vectors per row, a multiple of the quad width");
  constexpr int Q = S / 2;  // vectors per row per lane per stage
  int lane = int(threadIdx.x) & 31, warp = int(threadIdx.x) / 32;
  int row = (int(blockIdx.x) * (int(blockDim.x) / 32) + warp) * 16;
  if (row >= m) return;  // whole warp retires; the shuffles below always see a full warp
  int group = lane / 4, t = lane % 4, start = int(blockIdx.z) * (k / parts), end = start + k / parts;
  const bool r0 = row + group < m, r1 = row + group + 8 < m, bx = group < n;
  const bf16* w0 = w + size_t(r0 ? row + group : 0) * k;
  const bf16* w1 = w + size_t(r1 ? row + group + 8 : 0) * k;
  const bf16* xb = x + size_t(bx ? group : 0) * k + 2 * t;
  const int quad = lane & ~3;
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  uint4 staged[2][Q];
  auto fetch = [&](int base, int cnt) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int q = 0; q < Q; ++q) {
        int j = 4 * q + t;  // this lane's vector within the stage, for row half h
        bool valid = j < 2 * cnt && (h ? r1 : r0);
        staged[h][q] = valid ? v6::ld_nc_v4((h ? w1 : w0) + base + 8 * j) : make_uint4(0, 0, 0, 0);
      }
  };
  // fr[h][jj] = this lane's 4-byte word of vector jj (elements 8*jj+2t, +1) for row half h.
  uint32_t fr[2][2 * S];
  auto transpose = [&]() {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int q = 0; q < Q; ++q) {
        // Round r: lane s sends word (s-r)&3 of its vector; lane t receives word t of vector 4q+((t+r)&3).
        uint32_t got[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          uint32_t send = pick(staged[h][q], (t - r) & 3);
          got[r] = r == 0 ? send : __shfl_sync(0xffffffffu, send, quad | ((t + r) & 3));
        }
        // Vector 4q+s arrived in round (s-t)&3.
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          int r = (s - t) & 3;
          fr[h][4 * q + s] = r == 0 ? got[0] : r == 1 ? got[1] : r == 2 ? got[2] : got[3];
        }
      }
  };
  auto compute = [&](int base, int cnt) {
#pragma unroll
    for (int u = 0; u < S; ++u) {
      if (u < cnt) {
        int o = base + 16 * u;
        uint32_t b0 = bx ? v5::ld_nc(xb + o) : 0u, b1 = bx ? v5::ld_nc(xb + o + 8) : 0u;
        // v4: a0=W[r0][16u+2t], a1=W[r1][16u+2t], a2=W[r0][16u+8+2t], a3=W[r1][16u+8+2t].
        v5::mma(c0, c1, c2, c3, fr[0][2 * u], fr[1][2 * u], fr[0][2 * u + 1], fr[1][2 * u + 1], b0, b1);
      }
    }
  };
  int steps = (end - start) / 16, stages = (steps + S - 1) / S;
  fetch(start, min(S, steps));
  for (int s = 0; s < stages; ++s) {
    int base = start + s * 16 * S, cnt = min(S, steps - s * S);
    transpose();  // consumes staged[] for this stage
    if (s + 1 < stages) fetch(base + 16 * S, min(S, steps - (s + 1) * S));  // next stage in flight during MMAs
    compute(base, cnt);
  }
  size_t head = size_t(blockIdx.z) * padded_m * padded_n;
  if (r0) { size_t o = head + size_t(row + group) * padded_n + 2 * t; partial[o] = c0; partial[o + 1] = c1; }
  if (r1) { size_t o = head + size_t(row + group + 8) * padded_n + 2 * t; partial[o] = c2; partial[o + 1] = c3; }
}
}  // namespace v7

// v8: v7 with a two-stage XOR-butterfly quad transpose (4 shuffles, fewer selects).
// v7 header for reference -- v6's coalescing without shared memory. For each row, the four lanes of a quad
// (t=0..3) load consecutive 16-byte vectors (64 contiguous bytes per quad per row). A
// 16-byte vector j holds one 4-byte BF16 pair for each t, so a 4x4 word transpose inside
// the quad (three shuffle rotations plus register selects) yields exactly v4's fragment
// words. Same MMA sequence and K order as v4; registers only (nothing for racecheck to
// track beyond v4/v5); next stage's loads are in flight during the current stage's MMAs.
namespace v8 {
using bf16 = __nv_bfloat16;

template <int S>
__global__ void __launch_bounds__(128) mma_thin_partials(const bf16* w, const bf16* x, float* partial, int m, int n,
                                                         int k, int padded_n, int padded_m, int parts) {
  static_assert(S % 2 == 0 && S >= 2, "a stage holds 2*S vectors per row, a multiple of the quad width");
  constexpr int Q = S / 2;  // vectors per row per lane per stage
  int lane = int(threadIdx.x) & 31, warp = int(threadIdx.x) / 32;
  int row = (int(blockIdx.x) * (int(blockDim.x) / 32) + warp) * 16;
  if (row >= m) return;  // whole warp retires; the shuffles below always see a full warp
  int group = lane / 4, t = lane % 4, start = int(blockIdx.z) * (k / parts), end = start + k / parts;
  const bool r0 = row + group < m, r1 = row + group + 8 < m, bx = group < n;
  const bf16* w0 = w + size_t(r0 ? row + group : 0) * k;
  const bf16* w1 = w + size_t(r1 ? row + group + 8 : 0) * k;
  const bf16* xb = x + size_t(bx ? group : 0) * k + 2 * t;
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  uint4 staged[2][Q];
  auto fetch = [&](int base, int cnt) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int q = 0; q < Q; ++q) {
        int j = 4 * q + t;  // this lane's vector within the stage, for row half h
        bool valid = j < 2 * cnt && (h ? r1 : r0);
        staged[h][q] = valid ? v6::ld_nc_v4((h ? w1 : w0) + base + 8 * j) : make_uint4(0, 0, 0, 0);
      }
  };
  // fr[h][jj] = this lane's 4-byte word of vector jj (elements 8*jj+2t, +1) for row half h.
  uint32_t fr[2][2 * S];
  // Two-stage XOR butterfly 4x4 transpose (lane t holds A[t][0..3]; wants A[0..3][t]).
  // Stage 1 (xor 2, b=t>>1): send columns {2-2b,3-2b}, receive partner's {2b,2b+1}.
  // Stage 2 (xor 1, a=t&1): send column 2b+1-a of both held rows, receive column 2b+a.
  const bool b = (t >> 1) & 1, a = t & 1;
  auto transpose = [&]() {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int q = 0; q < Q; ++q) {
        const uint4 v = staged[h][q];
        uint32_t r0 = __shfl_xor_sync(0xffffffffu, b ? v.x : v.z, 2);
        uint32_t r1 = __shfl_xor_sync(0xffffffffu, b ? v.y : v.w, 2);
        uint32_t x0 = b ? v.z : v.x, x1 = b ? v.w : v.y;  // own row t, columns 2b, 2b+1
        // u[rb][cl]: row with bit1 = rb (row t has rb = b), column 2b+cl.
        uint32_t u00 = b ? r0 : x0, u01 = b ? r1 : x1, u10 = b ? x0 : r0, u11 = b ? x1 : r1;
        uint32_t gA = __shfl_xor_sync(0xffffffffu, a ? u00 : u01, 1);  // partner's row 1-a, column 2b+a
        uint32_t gB = __shfl_xor_sync(0xffffffffu, a ? u10 : u11, 1);  // partner's row 3-a, column 2b+a
        uint32_t oA = a ? u01 : u00, oB = a ? u11 : u10;               // own rows a, 2+a, column 2b+a
        fr[h][4 * q + 0] = a ? gA : oA;
        fr[h][4 * q + 1] = a ? oA : gA;
        fr[h][4 * q + 2] = a ? gB : oB;
        fr[h][4 * q + 3] = a ? oB : gB;
      }
  };
  auto compute = [&](int base, int cnt) {
#pragma unroll
    for (int u = 0; u < S; ++u) {
      if (u < cnt) {
        int o = base + 16 * u;
        uint32_t b0 = bx ? v5::ld_nc(xb + o) : 0u, b1 = bx ? v5::ld_nc(xb + o + 8) : 0u;
        // v4: a0=W[r0][16u+2t], a1=W[r1][16u+2t], a2=W[r0][16u+8+2t], a3=W[r1][16u+8+2t].
        v5::mma(c0, c1, c2, c3, fr[0][2 * u], fr[1][2 * u], fr[0][2 * u + 1], fr[1][2 * u + 1], b0, b1);
      }
    }
  };
  int steps = (end - start) / 16, stages = (steps + S - 1) / S;
  fetch(start, min(S, steps));
  for (int s = 0; s < stages; ++s) {
    int base = start + s * 16 * S, cnt = min(S, steps - s * S);
    transpose();  // consumes staged[] for this stage
    if (s + 1 < stages) fetch(base + 16 * S, min(S, steps - (s + 1) * S));  // next stage in flight during MMAs
    compute(base, cnt);
  }
  size_t head = size_t(blockIdx.z) * padded_m * padded_n;
  if (r0) { size_t o = head + size_t(row + group) * padded_n + 2 * t; partial[o] = c0; partial[o + 1] = c1; }
  if (r1) { size_t o = head + size_t(row + group + 8) * padded_n + 2 * t; partial[o] = c2; partial[o + 1] = c3; }
}
}  // namespace v8


class Linear {
  struct Plan {bool native;unsigned parts;};
  std::map<std::tuple<int,int,int>,Plan> plans;
  cublasLtHandle_t handle=nullptr;
  float* scratch=nullptr;size_t capacity=0;
  static void cuda(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
  static void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("candidate BLAS status="+std::to_string(int(s)));}
  // Thin-producer selection, measured cold-cache on Thor (thin-probe-v1): v6 wins once a
  // linear streams >=16 MiB of weights; below that v5 has more useful warp parallelism.
  // LFM3_THIN_POLICY (set by the generator) picks the large-weight producer: 6 = v6 (shared-memory
  // staging), 7 = v7 (quad rotation transpose), 8 = v8 (quad XOR-butterfly transpose). All are
  // bitwise-identical to v4; choices follow interleaved cold-cache runs in validation/thin-probe-v1.
  enum class Thin {v5u4,v5u16,v6s8,v7s4,v7s8,v8s8};
  static Thin thin_kind(int m,int k){
#if LFM3_THIN_POLICY==8
    return m>=2048?Thin::v8s8:Thin::v5u16;
#else
    if(size_t(m)*k*2>=(size_t(16)<<20)){
#if LFM3_THIN_POLICY==7
      return (k>=4096||(m>=7000&&m<8000))?Thin::v7s4:Thin::v7s8;
#else
      return Thin::v6s8;
#endif
    }
    return m>=1024?Thin::v5u4:Thin::v5u16;
#endif
  }
  static const char* thin_name(Thin t){
    switch(t){case Thin::v5u4:return "v5u4";case Thin::v5u16:return "v5u16";case Thin::v6s8:return "v6s8";case Thin::v7s4:return "v7s4";case Thin::v7s8:return "v7s8";default:return "v8s8";}
  }
  Plan plan(int m,int n,int k){
    auto key=std::make_tuple(m,n,k);auto found=plans.find(key);if(found!=plans.end())return found->second;
    if(!handle)blas(cublasLtCreate(&handle));
    cublasLtMatmulDesc_t op;blas(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t trans=CUBLAS_OP_T;blas(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
    cublasLtMatrixLayout_t a,b,c;blas(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,m,k));
    blas(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,n,k));blas(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,m,n,m));
    cublasLtMatmulPreference_t preference;blas(cublasLtMatmulPreferenceCreate(&preference));size_t budget=8519680;
    blas(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&budget,sizeof(budget)));
    cublasLtMatmulHeuristicResult_t result{};int returned=0;
    blas(cublasLtMatmulAlgoGetHeuristic(handle,op,a,b,c,c,preference,1,&result,&returned));
    if(returned!=1||result.state!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("candidate plan unavailable");
    unsigned id=0,parts=0;size_t bytes=0;
    blas(cublasLtMatmulAlgoConfigGetAttribute(&result.algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&bytes));
    blas(cublasLtMatmulAlgoConfigGetAttribute(&result.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&parts,sizeof(parts),&bytes));
    if(!parts)parts=1;
    blas(cublasLtMatmulPreferenceDestroy(preference));blas(cublasLtMatrixLayoutDestroy(a));
    blas(cublasLtMatrixLayoutDestroy(b));blas(cublasLtMatrixLayoutDestroy(c));blas(cublasLtMatmulDescDestroy(op));
    Plan p{id==66,parts};
    if(p.native&&(parts>16||k%parts||(k/parts)%16))throw std::runtime_error("candidate partition unsupported");
    const char* thin=n>8?"none":thin_name(thin_kind(m,k));
    std::fprintf(stderr,"{\"candidate_linear_plan\":true,\"m\":%d,\"n\":%d,\"k\":%d,\"heuristic_id\":%u,\"parts\":%u,\"native_wmma\":%s,\"thin\":\"%s\"}\n",m,n,k,id,parts,p.native?"true":"false",thin);
    plans.emplace(key,p);return p;
  }
 public:
  Linear()=default;Linear(const Linear&)=delete;Linear& operator=(const Linear&)=delete;
  ~Linear(){if(scratch)cudaFree(scratch);if(handle)cublasLtDestroy(handle);}
  void run(cublasHandle_t original,const __nv_bfloat16* w,const __nv_bfloat16* x,__nv_bfloat16* out,int n,int k,int m){
    const auto p=plan(m,n,k);
    if(!p.native){float alpha=1,beta=0;blas(cublasGemmEx(original,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&alpha,w,CUDA_R_16BF,k,x,CUDA_R_16BF,k,&beta,out,CUDA_R_16BF,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));return;}
    int pm=(m+15)/16*16,pn=n<=8?8:(n+15)/16*16;size_t required=size_t(pm)*pn*p.parts;
    if(required>capacity){if(scratch){cuda(cudaFree(scratch));scratch=nullptr;capacity=0;}cuda(cudaMalloc(&scratch,required*sizeof(float)));capacity=required;}
    if(n<=8){
      int warps=std::min(4,pm/16);dim3 grid((pm/16+warps-1)/warps,1,p.parts),block(32*warps);
      // Fail closed: v6 issues 16-byte loads (rows and partition starts must stay 16-byte aligned);
      // v5 keeps v4's 4-byte operand loads.
      if(reinterpret_cast<uintptr_t>(w)%16||k%8)throw std::runtime_error("candidate thin alignment unsupported");
      switch(thin_kind(m,k)){
        case Thin::v6s8:v6::mma_thin_partials<8><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
        case Thin::v5u4:v5::mma_thin_partials<4><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
        case Thin::v5u16:v5::mma_thin_partials<16><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
        case Thin::v7s4:v7::mma_thin_partials<4><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
        case Thin::v7s8:v7::mma_thin_partials<8><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
        case Thin::v8s8:v8::mma_thin_partials<8><<<grid,block>>>(w,x,scratch,m,n,k,pn,pm,p.parts);break;
      }
    }
    else wmma_partials<<<dim3(pm/16,pn/16,p.parts),32>>>(w,x,scratch,m,n,k,pn,pm,p.parts);
    cuda(cudaGetLastError());
    size_t count=size_t(m)*n;wmma_finish<<<(count+255)/256,256>>>(scratch,out,count,m,n,pm,pn,p.parts);cuda(cudaGetLastError());
  }
};
} // namespace lfm3_wmma_candidate
