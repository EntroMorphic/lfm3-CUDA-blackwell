#pragma once
// Isolated candidate only. Production never includes this header.
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
  uint32_t bits;memcpy(&bits,p,sizeof(bits));return bits;
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

class Linear {
  struct Plan {bool native;unsigned parts;};
  std::map<std::tuple<int,int,int>,Plan> plans;
  cublasLtHandle_t handle=nullptr;
  float* scratch=nullptr;size_t capacity=0;
  static void cuda(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
  static void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("candidate BLAS status="+std::to_string(int(s)));}
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
    std::fprintf(stderr,"{\"candidate_linear_plan\":true,\"m\":%d,\"n\":%d,\"k\":%d,\"heuristic_id\":%u,\"parts\":%u,\"native_wmma\":%s}\n",m,n,k,id,parts,p.native?"true":"false");
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
    if(n<=8){int warps=std::min(4,pm/16);mma_thin_partials<<<dim3((pm/16+warps-1)/warps,1,p.parts),32*warps>>>(w,x,scratch,m,n,k,pn,pm,p.parts);}
    else wmma_partials<<<dim3(pm/16,pn/16,p.parts),32>>>(w,x,scratch,m,n,k,pn,pm,p.parts);
    cuda(cudaGetLastError());
    size_t count=size_t(m)*n;wmma_finish<<<(count+255)/256,256>>>(scratch,out,count,m,n,pm,pn,p.parts);cuda(cudaGetLastError());
  }
};
} // namespace lfm3_wmma_candidate
