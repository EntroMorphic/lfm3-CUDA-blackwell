#pragma once
// Isolated candidate only. Production never includes this header.
// Same as lfm3_wmma_runtime_candidate.cuh (v4) except the thin (rows<=8) producer:
// v6 (shared-memory staged, 16-byte row-contiguous loads) for large weights and
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


class Linear {
  struct Plan {bool native;unsigned parts;};
  std::map<std::tuple<int,int,int>,Plan> plans;
  cublasLtHandle_t handle=nullptr;
  float* scratch=nullptr;size_t capacity=0;
  static void cuda(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
  static void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("candidate BLAS status="+std::to_string(int(s)));}
  // Thin-producer selection, measured cold-cache on Thor (thin-probe-v1): v6 wins once a
  // linear streams >=16 MiB of weights; below that v5 has more useful warp parallelism.
  enum class Thin {v5u4,v5u16,v6s8};
  static Thin thin_kind(int m,int k){
    if(size_t(m)*k*2>=(size_t(16)<<20))return Thin::v6s8;
    return m>=1024?Thin::v5u4:Thin::v5u16;
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
    const char* thin=n>8?"none":thin_kind(m,k)==Thin::v6s8?"v6s8":thin_kind(m,k)==Thin::v5u4?"v5u4":"v5u16";
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
      }
    }
    else wmma_partials<<<dim3(pm/16,pn/16,p.parts),32>>>(w,x,scratch,m,n,k,pn,pm,p.parts);
    cuda(cudaGetLastError());
    size_t count=size_t(m)*n;wmma_finish<<<(count+255)/256,256>>>(scratch,out,count,m,n,pm,pn,p.parts);cuda(cudaGetLastError());
  }
};
} // namespace lfm3_wmma_candidate
