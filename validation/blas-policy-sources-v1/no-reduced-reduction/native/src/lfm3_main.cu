// Experimental batch-one text runner. Separate from the guarded LFM2 runtime.
#include <lfm/model_config.hpp>
#include <lfm/safetensors.hpp>
#include <lfm/lfm3_ops.cuh>
#include <lfm/text_serving_limits.hpp>
#include <lfm/text_attention_tiles.hpp>
#include <lfm/text_sampling.hpp>
#include <lfm/handle_setup.hpp>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>

using bf16 = __nv_bfloat16;
namespace {
void cuda_check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void blas_check(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(s)); }
template<class T> struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { cuda_check(cudaMalloc(&p, lfm::checked_product({n, sizeof(T)}))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
__global__ void embed(const bf16* weights, const int* ids, bf16* out, int rows, int width) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(rows) * width) out[i] = weights[size_t(ids[i / width]) * width + i % width];
}
// Vector-4 schedule from ATen Reduce.cuh; retained PyTorch license applies.
// Match the reduction's row-count-dependent block width, not just its formula.
__global__ void norm(const bf16* x, const bf16* w, bf16* y, int width, float eps,
                     const bf16* residual = nullptr) {
  const size_t base = size_t(blockIdx.x) * width;
  float accum[4] = {0,0,0,0};
  for (int d = threadIdx.x*4; d < width; d += blockDim.x*4) {
    for(int j=0;j<4;++j) {
      float v = __bfloat162float(x[base+d+j]);
      accum[j] = __fadd_rn(accum[j],__fmul_rn(v,v));
    }
  }
  float sum=((accum[0]+accum[1])+accum[2])+accum[3];
  __shared__ float sums[512];
  sums[threadIdx.x] = sum;
  for (int s = blockDim.x/2; s>=32; s/=2) {
    __syncthreads();
    if (threadIdx.x < s) { sum+=sums[threadIdx.x+s]; sums[threadIdx.x]=sum; }
  }
  __syncthreads();
  if(threadIdx.x<32) for(int s=16;s;s/=2) sum+=__shfl_down_sync(0xffffffff,sum,s);
  __shared__ float scale;
  if(threadIdx.x==0) scale=rsqrtf(__fadd_rn(__fmul_rn(sum,1.0f/width),eps));
  __syncthreads();
  for (int d = threadIdx.x; d < width; d += blockDim.x) {
    auto value=__hmul(__float2bfloat16_rn(__fmul_rn(__bfloat162float(x[base+d]),scale)),w[d]);
    y[base+d]=residual?__hadd(value,residual[base+d]):value;
  }
}
int norm_threads(int rows,int width) {
  auto power=[](int n) {int x=1; while(x<=n/2 && x<512) x*=2; return x;};
  int dim0=power(width/4),dim1=power(rows);
  int bx=std::min(dim0,32),by=std::min(dim1,512/bx);
  return std::min(dim0,512/by);
}
__global__ void swiglu(const bf16* a, const bf16* b, bf16* out, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = __bfloat162float(a[i]);
    // ATen SiLU multiplies by the reciprocal sigmoid, then rounds to BF16.
    const float sigmoid = 1.0f / (1.0f + expf(-x));
    out[i] = __hmul(__float2bfloat16_rn(__fmul_rn(x, sigmoid)), b[i]);
  }
}
__global__ void rotary_from_frequencies(bf16* x,const int* positions,const float* inverse,
                                        int rows,int heads,int dim,int pairs) {
  const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(rows)*heads*(dim/2)) return;
  const int d=i%(dim/2);
  if(d>=pairs) return;
  const size_t h=i/(dim/2),base=h*dim+d;
  const float angle=__fmul_rn(float(positions[h/heads]),inverse[d]);
  const float cosine=__bfloat162float(__float2bfloat16_rn(cosf(angle)));
  const float sine=__bfloat162float(__float2bfloat16_rn(sinf(angle)));
  const float a=__bfloat162float(x[base]),b=__bfloat162float(x[base+dim/2]);
  x[base]=__float2bfloat16_rn(__fadd_rn(__fmul_rn(a,cosine),-__fmul_rn(b,sine)));
  x[base+dim/2]=__float2bfloat16_rn(__fadd_rn(__fmul_rn(b,cosine),__fmul_rn(a,sine)));
}
__global__ void scatter(const bf16* k, const bf16* v, bf16* kc, bf16* vc,
                        int rows, int heads, int dim, int capacity, int start) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(rows) * heads * dim) {
    int d = i % dim, h = (i / dim) % heads, t = i / (heads * dim);
    size_t dst = (size_t(h) * capacity + start + t) * dim + d;
    kc[dst] = k[i]; vc[dst] = v[i];
  }
}
__global__ void probabilities(const bf16* scores,bf16* probs,int rows,int end,int start,int window,int dim,int key_start) {
  const int row=blockIdx.x%rows,tid=threadIdx.x;
  const size_t base=size_t(blockIdx.x)*end;
  const int position=start+row;
  const int begin=window?max(0,position-window+1):0;
  // ATen persistent-softmax order: lane-strided accumulators, XOR butterfly.
  float elements[128];
  int power=1; while(power<end) power*=2;
  int iterations=(power+31)/32;
  float maximum=-INFINITY;
  for(int i=0;i<iterations;++i) {
    int t=tid+i*32;
    float x=-INFINITY;
    if(t<end && t+key_start>=begin && t+key_start<=position)
      x=__bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(scores[base+t]),rsqrtf(float(dim)))));
    elements[i]=x; maximum=fmaxf(maximum,x);
  }
  for(int offset=16;offset;offset/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset));
  float sum=0;
  for(int i=0;i<iterations;++i) {
    elements[i]=expf(elements[i]-maximum); sum+=elements[i];
  }
  for(int offset=16;offset;offset/=2) sum+=__shfl_xor_sync(0xffffffff,sum,offset);
  for(int i=0;i<iterations;++i) {
    int t=tid+i*32;
    if(t<end) probs[base+t]=__float2bfloat16_rn(elements[i]/sum);
  }
}
template<bool Maximum> __device__ float block_softmax_reduce(float value,float* shared) {
  const int tid=threadIdx.x,lane=tid%32;
  for(int offset=16;offset;offset/=2) {
    const float other=__shfl_down_sync(0xffffffff,value,offset);
    value=Maximum?fmaxf(value,other):__fadd_rn(value,other);
  }
  __syncthreads();
  if(lane==0) shared[tid/32]=value;
  __syncthreads();
  if(tid<32) {
    value=shared[tid];
    for(int offset=16;offset;offset/=2) {
      const float other=__shfl_down_sync(0xffffffff,value,offset);
      value=Maximum?fmaxf(value,other):__fadd_rn(value,other);
    }
    if(tid==0) shared[0]=value;
  }
  __syncthreads();
  return shared[0];
}
// ATen FP32 forward switches from persistent warp to register/block at 2049.
// See native/third_party/PYTORCH_LICENSE and audit/CONTEXT.md for attribution.
__global__ void probabilities_large(const bf16* scores,bf16* probs,int rows,int end,int start,int window,int dim,int key_start) {
  const int tid=threadIdx.x,row=blockIdx.x%rows,position=start+row;
  const int begin=window?max(0,position-window+1):0;
  const size_t base=size_t(blockIdx.x)*end;
  float elements[32];
  const int iterations=(end+1023)/1024;
  float maximum=-3.402823466e38f;
  for(int i=0;i<iterations;++i) {
    const int t=tid+i*1024;
    float value=-INFINITY;
    if(t<end && t+key_start>=begin && t+key_start<=position)
      value=__bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(scores[base+t]),rsqrtf(float(dim)))));
    elements[i]=value;
    if(t<end) maximum=fmaxf(maximum,value);
  }
  __shared__ float shared[32];
  maximum=block_softmax_reduce<true>(maximum,shared);
  float sum=0;
  for(int i=0;i<iterations;++i) if(tid+i*1024<end) sum=__fadd_rn(sum,expf(elements[i]-maximum));
  sum=block_softmax_reduce<false>(sum,shared);
  for(int i=0;i<iterations;++i) if(tid+i*1024<end)
    probs[base+tid+i*1024]=__float2bfloat16_rn(expf(elements[i]-maximum)/sum);
}
unsigned grid(size_t n) { return (n + 255) / 256; }
lfm::Json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  if (f.tellg() > 4 * 1024 * 1024) throw std::runtime_error("request too large");
  f.seekg(0);
  return lfm::Json::parse(std::string(std::istreambuf_iterator<char>(f), {}));
}
std::vector<int> ids(const lfm::Json& req, const std::string& key, int vocab) {
  std::vector<int> out;
  if (!req.has(key)) return out;
  for (const auto& x : req.at(key).array()) {
    int id = x.integer();
    if (id >= vocab) throw std::runtime_error("token outside vocabulary: " + key);
    out.push_back(id);
  }
  return out;
}

class Runner {
 public:
  lfm::ModelConfig config;
  lfm::RuntimeShape s;
  std::shared_ptr<lfm::Safetensors> weights;
  cublasHandle_t blas = nullptr;
  Buffer<bf16> hidden, normalized, projected, operated, updated, q, k, v, gates, attended, gated, a, b, act, logits;
  Buffer<int> positions, tokens;
  Buffer<unsigned char> workspace;
  Buffer<bf16> score_buffer, probability_buffer;
  Buffer<const bf16*> pointer_a, pointer_b;
  Buffer<bf16*> pointer_c;
  std::vector<std::unique_ptr<Buffer<bf16>>> keys, values, states;
  std::vector<std::unique_ptr<Buffer<float>>> frequencies;
  std::string trace;
  bool trace_logits_only = false;
  int step = 0, cursor = 0;

  Runner(const std::string& directory, lfm::ModelConfig cfg,
         std::shared_ptr<lfm::Safetensors> shared_weights = {})
      : config(cfg), s(cfg.shape), weights(shared_weights?std::move(shared_weights):
          std::make_shared<lfm::Safetensors>(directory + "/model.safetensors")),
        hidden(size_t(s.context)*s.hidden), normalized(size_t(s.context)*s.hidden),
        projected(size_t(s.context)*3*s.hidden), operated(size_t(s.context)*s.hidden), updated(size_t(s.context)*s.hidden),
        q(size_t(s.context)*s.hidden), k(size_t(s.context)*s.kv_width()), v(size_t(s.context)*s.kv_width()),
        gates(size_t(s.context)*s.query_heads), attended(size_t(s.context)*s.hidden), gated(size_t(s.context)*s.hidden),
        a(size_t(s.context)*s.mlp), b(size_t(s.context)*s.mlp), act(size_t(s.context)*s.mlp), logits(s.vocabulary),
        positions(s.context), tokens(s.context), workspace(4096*1024*2+16*1024*8),
        score_buffer(lfm::text::attention_scratch_elements(s.context,s.query_heads)),
        probability_buffer(lfm::text::attention_scratch_elements(s.context,s.query_heads)),
        pointer_a(s.query_heads),pointer_b(s.query_heads),pointer_c(s.query_heads) {
    // Validate the complete selected text schema before launching any forward kernels.
    weight("embed_tokens.weight", {s.vocabulary, s.hidden}); weight("embedding_norm.weight", {s.hidden});
    for (int i = 0; i < s.layers; ++i) {
      std::string p = "layers." + std::to_string(i) + ".";
      for (auto name : {"operator_norm.weight", "ffn_norm.weight", "post_operator_norm.weight", "post_ffn_norm.weight"})
        weight(p + name, {s.hidden});
      weight(p+"feed_forward.w1.weight", {s.mlp,s.hidden}); weight(p+"feed_forward.w3.weight", {s.mlp,s.hidden});
      weight(p+"feed_forward.w2.weight", {s.hidden,s.mlp});
      if (config.attention[i]) {
        for (auto name : {"q_proj.weight", "o_proj.weight"}) weight(p+"self_attn."+name, {s.hidden,s.hidden});
        for (auto name : {"k_proj.weight", "v_proj.weight"}) weight(p+"self_attn."+name, {s.kv_width(),s.hidden});
        for (auto name : {"q_norm.weight", "k_norm.weight"}) weight(p+"self_attn."+name, {s.head_dim});
        weight(p+"self_attn.gate_proj.weight", {s.query_heads,s.hidden});
        keys.emplace_back(new Buffer<bf16>(s.kv_layer_stride())); values.emplace_back(new Buffer<bf16>(s.kv_layer_stride()));
        states.emplace_back(nullptr);
        frequencies.emplace_back(new Buffer<float>(s.head_dim/2));
        std::vector<float> inverse(s.head_dim/2,0.0f);
        const auto spec=config.layer_specs[i];
        // The checkpoint builds arange/pow/reciprocal on CPU, then transfers.
        // GPU powf differs by an ULP at some frequencies (observable at pos850).
        for(int d=0;d<spec.rotary_pairs;++d)
          inverse[d]=1.0f/::powf(spec.rope_theta,float(2*d)/s.head_dim);
        cuda_check(cudaMemcpy(frequencies.back()->p,inverse.data(),inverse.size()*sizeof(float),cudaMemcpyHostToDevice));
      } else {
        weight(p+"conv.conv.weight", {s.hidden,1,7}); weight(p+"conv.in_proj.weight", {3*s.hidden,s.hidden});
        weight(p+"conv.out_proj.weight", {s.hidden,s.hidden});
        keys.emplace_back(nullptr); values.emplace_back(nullptr); states.emplace_back(new Buffer<bf16>(size_t(s.hidden)*7));
        frequencies.emplace_back(nullptr);
      }
    }
    blas_check(cublasCreate(&blas));
    lfm::finish_handle_setup(blas,[](cublasHandle_t handle) noexcept {cublasDestroy(handle);},[&] {
      // Match PyTorch 2.10's non-Hopper cuBLAS workspace budget.
      blas_check(cublasSetWorkspace(blas, workspace.p, 8519680));
      // Match default PyTorch BF16 cuBLAS policy; outputs materialize in BF16.
      blas_check(cublasSetMathMode(blas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
      reset();
    });
  }
  ~Runner() { if (blas) cublasDestroy(blas); }
  const bf16* weight(const std::string& name, std::initializer_list<int64_t> shape) {
    const auto& t = weights->at(config.tensor_prefix + name);
    if (t.dtype != "BF16" || t.shape != std::vector<int64_t>(shape)) throw std::runtime_error("weight contract: " + name);
    return t.device_data;
  }
  const bf16* w(const std::string& name) { return weights->at(config.tensor_prefix + name).device_data; }
  void reset() {
    cuda_check(cudaDeviceSynchronize());
    cursor = 0; step = 0;
    for (int i=0;i<s.layers;++i) {
      if (states[i]) cuda_check(cudaMemset(states[i]->p,0,size_t(s.hidden)*7*sizeof(bf16)));
      if (keys[i]) {
        cuda_check(cudaMemset(keys[i]->p,0,s.kv_layer_stride()*sizeof(bf16)));
        cuda_check(cudaMemset(values[i]->p,0,s.kv_layer_stride()*sizeof(bf16)));
      }
    }
  }
  void dump(const std::string& name, const bf16* ptr, size_t n) {
    if (trace.empty() || (trace_logits_only && name!="logits")) return;
    std::vector<bf16> data(n);
    cuda_check(cudaMemcpy(data.data(),ptr,n*sizeof(bf16),cudaMemcpyDeviceToHost));
    std::ofstream f(trace+"/"+std::to_string(step)+"."+name+".bf16",std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),n*sizeof(bf16));
    if (!f) throw std::runtime_error("trace write failed");
  }
  void linear(const std::string& name, const bf16* x, bf16* out, int rows, int input, int output) {
    float alpha=1, beta=0;
    // Row-major X W^T = column-major W X^T, with W stored row-major.
    blas_check(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,output,rows,input,&alpha,
      w(name),CUDA_R_16BF,input,x,CUDA_R_16BF,input,&beta,out,CUDA_R_16BF,output,
      CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  void normalize(const bf16* x, const std::string& weight_name, bf16* out, int rows, int width) {
    norm<<<rows,norm_threads(rows,width)>>>(x,w(weight_name),out,width,s.epsilon);
  }
  void attention_tile(int layer,int rows,int window,int first,int total_rows) {
    // Dynamic sliding cache returns only its last window for one-token decode.
    // Masking a larger product is mathematically equivalent, but not BF16-equivalent.
    const auto range=lfm::text::attention_key_range(s.context,cursor,total_rows,first,rows,window);
    const int key_start=range.begin;
    const int end=range.span();
    std::vector<const bf16*> ap(s.query_heads),bp(s.query_heads);
    std::vector<bf16*> cp(s.query_heads);
    for(int h=0;h<s.query_heads;++h) {
      ap[h]=keys[layer]->p+(size_t(h/(s.query_heads/s.kv_heads))*s.context+key_start)*s.head_dim;
      bp[h]=q.p+size_t(first)*s.hidden+h*s.head_dim;
      cp[h]=score_buffer.p+size_t(h)*rows*end;
    }
    auto upload=[&] {
      cuda_check(cudaMemcpy(pointer_a.p,ap.data(),s.query_heads*sizeof(bf16*),cudaMemcpyHostToDevice));
      cuda_check(cudaMemcpy(pointer_b.p,bp.data(),s.query_heads*sizeof(bf16*),cudaMemcpyHostToDevice));
      cuda_check(cudaMemcpy(pointer_c.p,cp.data(),s.query_heads*sizeof(bf16*),cudaMemcpyHostToDevice));
    };
    upload();
    float alpha=1,beta=0;
    blas_check(cublasGemmBatchedEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,end,rows,s.head_dim,&alpha,
      reinterpret_cast<const void* const*>(pointer_a.p),CUDA_R_16BF,s.head_dim,
      reinterpret_cast<const void* const*>(pointer_b.p),CUDA_R_16BF,s.hidden,&beta,
      reinterpret_cast<void* const*>(pointer_c.p),CUDA_R_16BF,end,s.query_heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if(end<=2048) probabilities<<<rows*s.query_heads,32>>>(score_buffer.p,probability_buffer.p,rows,end,cursor+first,window,s.head_dim,key_start);
    else probabilities_large<<<rows*s.query_heads,1024>>>(score_buffer.p,probability_buffer.p,rows,end,cursor+first,window,s.head_dim,key_start);
    std::string label="layers."+std::to_string(layer)+".self_attn.probabilities";
    if(total_rows>lfm::text::scratch_rows(s.context)) label+=".tile"+std::to_string(first);
    dump(label,probability_buffer.p,size_t(s.query_heads)*rows*end);
    for(int h=0;h<s.query_heads;++h) {
      ap[h]=values[layer]->p+(size_t(h/(s.query_heads/s.kv_heads))*s.context+key_start)*s.head_dim;
      bp[h]=probability_buffer.p+size_t(h)*rows*end;
      cp[h]=attended.p+size_t(first)*s.hidden+h*s.head_dim;
    }
    upload();
    blas_check(cublasGemmBatchedEx(blas,CUBLAS_OP_N,CUBLAS_OP_N,s.head_dim,rows,end,&alpha,
      reinterpret_cast<const void* const*>(pointer_a.p),CUDA_R_16BF,s.head_dim,
      reinterpret_cast<const void* const*>(pointer_b.p),CUDA_R_16BF,end,&beta,
      reinterpret_cast<void* const*>(pointer_c.p),CUDA_R_16BF,s.hidden,s.query_heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  void attention(int layer,int rows,int window) {
    const int tile=lfm::text::scratch_rows(s.context);
    for(int first=0;first<rows;first+=tile)
      attention_tile(layer,std::min(tile,rows-first),window,first,rows);
  }
  std::vector<float> forward(const std::vector<int>& input) {
    const int rows = input.size();
    if (rows < 1 || rows > s.context-cursor || (cursor && rows != 1))
      throw std::runtime_error("only empty-cache prefill or single-token decode within capacity is supported");
    for (int id : input) if (id < 0 || id >= s.vocabulary) throw std::runtime_error("invalid token");
    std::vector<int> pos(rows);
    for(int t=0;t<rows;++t) pos[t]=cursor+t;
    cuda_check(cudaMemcpy(tokens.p,input.data(),rows*sizeof(int),cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(positions.p,pos.data(),rows*sizeof(int),cudaMemcpyHostToDevice));
    embed<<<grid(size_t(rows)*s.hidden),256>>>(w("embed_tokens.weight"),tokens.p,hidden.p,rows,s.hidden);
    dump("embed",hidden.p,size_t(rows)*s.hidden);
    for(int layer=0;layer<s.layers;++layer) {
      const std::string p="layers."+std::to_string(layer)+".";
      const auto spec=config.layer_specs[layer];
      normalize(hidden.p,p+"operator_norm.weight",normalized.p,rows,s.hidden);
      dump(p+"operator_norm",normalized.p,size_t(rows)*s.hidden);
      if(!config.attention[layer]) {
        linear(p+"conv.in_proj.weight",normalized.p,projected.p,rows,s.hidden,3*s.hidden);
        dump(p+"conv.in_proj",projected.p,size_t(rows)*3*s.hidden);
        if(cursor==0) lfm::lfm3::convolution7_prefill<<<grid(size_t(rows)*s.hidden),256>>>(projected.p,w(p+"conv.conv.weight"),states[layer]->p,operated.p,1,rows,s.hidden);
        else lfm::lfm3::convolution7<<<grid(s.hidden),256>>>(projected.p,w(p+"conv.conv.weight"),states[layer]->p,operated.p,s.hidden,1);
        dump(p+"conv.product",operated.p,size_t(rows)*s.hidden);
        linear(p+"conv.out_proj.weight",operated.p,updated.p,rows,s.hidden,s.hidden);
      } else {
        const std::string at=p+"self_attn.";
        linear(at+"q_proj.weight",normalized.p,q.p,rows,s.hidden,s.hidden);
        linear(at+"k_proj.weight",normalized.p,k.p,rows,s.hidden,s.kv_width());
        linear(at+"v_proj.weight",normalized.p,v.p,rows,s.hidden,s.kv_width());
        normalize(q.p,at+"q_norm.weight",q.p,rows*s.query_heads,s.head_dim);
        normalize(k.p,at+"k_norm.weight",k.p,rows*s.kv_heads,s.head_dim);
        rotary_from_frequencies<<<grid(size_t(rows)*s.hidden/2),256>>>(q.p,positions.p,frequencies[layer]->p,rows,s.query_heads,s.head_dim,spec.rotary_pairs);
        rotary_from_frequencies<<<grid(size_t(rows)*s.kv_width()/2),256>>>(k.p,positions.p,frequencies[layer]->p,rows,s.kv_heads,s.head_dim,spec.rotary_pairs);
        dump(at+"q_rotary",q.p,size_t(rows)*s.hidden); dump(at+"k_rotary",k.p,size_t(rows)*s.kv_width());
        scatter<<<grid(size_t(rows)*s.kv_width()),256>>>(k.p,v.p,keys[layer]->p,values[layer]->p,rows,s.kv_heads,s.head_dim,s.context,cursor);
        attention(layer,rows,spec.window);
        dump(at+"attended",attended.p,size_t(rows)*s.hidden);
        linear(at+"gate_proj.weight",normalized.p,gates.p,rows,s.hidden,s.query_heads);
        lfm::lfm3::head_gate<<<grid(size_t(rows)*s.hidden),256>>>(attended.p,gates.p,gated.p,size_t(rows)*s.hidden,s.head_dim);
        dump(at+"gated",gated.p,size_t(rows)*s.hidden);
        linear(at+"o_proj.weight",gated.p,updated.p,rows,s.hidden,s.hidden);
      }
      dump(p+"operator",updated.p,size_t(rows)*s.hidden);
      norm<<<rows,norm_threads(rows,s.hidden)>>>(updated.p,w(p+"post_operator_norm.weight"),hidden.p,s.hidden,s.epsilon,hidden.p);
      dump(p+"residual",hidden.p,size_t(rows)*s.hidden);
      normalize(hidden.p,p+"ffn_norm.weight",normalized.p,rows,s.hidden);
      dump(p+"ffn_norm",normalized.p,size_t(rows)*s.hidden);
      linear(p+"feed_forward.w1.weight",normalized.p,a.p,rows,s.hidden,s.mlp);
      linear(p+"feed_forward.w3.weight",normalized.p,b.p,rows,s.hidden,s.mlp);
      dump(p+"feed_forward.w1",a.p,size_t(rows)*s.mlp);
      dump(p+"feed_forward.w3",b.p,size_t(rows)*s.mlp);
      swiglu<<<grid(size_t(rows)*s.mlp),256>>>(a.p,b.p,act.p,size_t(rows)*s.mlp);
      dump(p+"feed_forward.product",act.p,size_t(rows)*s.mlp);
      linear(p+"feed_forward.w2.weight",act.p,updated.p,rows,s.mlp,s.hidden);
      dump(p+"feed_forward",updated.p,size_t(rows)*s.hidden);
      norm<<<rows,norm_threads(rows,s.hidden)>>>(updated.p,w(p+"post_ffn_norm.weight"),hidden.p,s.hidden,s.epsilon,hidden.p);
      dump(p+"output",hidden.p,size_t(rows)*s.hidden);
    }
    normalize(hidden.p,"embedding_norm.weight",normalized.p,rows,s.hidden);
    dump("final_norm",normalized.p,size_t(rows)*s.hidden);
    linear("embed_tokens.weight",normalized.p+size_t(rows-1)*s.hidden,logits.p,1,s.hidden,s.vocabulary);
    dump("logits",logits.p,s.vocabulary);
    cuda_check(cudaGetLastError());
    std::vector<bf16> raw(s.vocabulary);
    cuda_check(cudaMemcpy(raw.data(),logits.p,s.vocabulary*sizeof(bf16),cudaMemcpyDeviceToHost));
    std::vector<float> result(s.vocabulary);
    for(int i=0;i<s.vocabulary;++i) {
      result[i]=__bfloat162float(raw[i]);
      if(!std::isfinite(result[i])) throw std::runtime_error("nonfinite logits");
    }
    cursor+=rows; ++step;
    return result;
  }
};
void print_ids(std::ostream& f,const std::vector<int>& x) {
  f<<"["; for(size_t i=0;i<x.size();++i) { if(i) f<<","; f<<x[i]; } f<<"]";
}
} // namespace

#ifndef LFM3_RUNTIME_ONLY
int main(int argc,char** argv) try {
  if(argc!=5 || std::string(argv[1])!="--experimental")
    throw std::runtime_error("usage: liquid-lfm3 --experimental MODEL_DIRECTORY REQUEST.json OUTPUT.json");
  auto request=read_json(argv[3]);
  int capacity=request.has("capacity")?request.at("capacity").integer():4096;
  lfm::text::scratch_rows(capacity);
  const bool long_context=request.has("experimental_long_context") && request.at("experimental_long_context").boolean();
  if(capacity>4096 && !long_context)
    throw std::runtime_error("capacity must be 1..4096 unless experimental_long_context is enabled (unaccepted 32K bring-up)");
  auto cfg=lfm::ModelConfig::load(std::string(argv[2])+"/config.json",capacity);
  if(!cfg.is_lfm3() || !cfg.sandwich_norm || !cfg.tied || cfg.shape.conv_width!=7 || cfg.shape.head_dim!=128 || cfg.shape.hidden!=2048)
    throw std::runtime_error("runner requires tied, sandwich-normalized width-7 LFM3 with hidden=2048, head_dim=128");
  auto input=ids(request,"input_ids",cfg.shape.vocabulary);
  auto forced=ids(request,"teacher_ids",cfg.shape.vocabulary);
  auto suppress=ids(request,"suppress_tokens",cfg.shape.vocabulary);
  auto stop=ids(request,"stop_tokens",cfg.shape.vocabulary);
  stop.insert(stop.end(),cfg.eos.begin(),cfg.eos.end());
  if(cfg.architecture==lfm::ModelConfig::Architecture::Lfm3Omni) {
    auto omni=read_json(std::string(argv[2])+"/config.json");
    auto special=[&](const std::string& name) {
      int token=omni.at(name).integer();
      if(token>=cfg.shape.vocabulary) throw std::runtime_error("invalid Omni modality token: "+name);
      return token;
    };
    for(auto name:{"audio_token_id","audio_out_token_id","image_token_id","video_token_id","mixed_start_token_id","audio_start_token_id"})
      suppress.push_back(special(name));
    stop.push_back(special("text_end_token_id"));
  }
  int count=request.has("max_new_tokens")?request.at("max_new_tokens").integer():0;
  int warmups=request.has("warmup_runs")?request.at("warmup_runs").integer():0;
  if(warmups>10) throw std::runtime_error("at most 10 warmup runs");
  const bool check_reset=request.has("check_reset") && request.at("check_reset").boolean();
  const bool trace_logits_only=request.has("trace_logits_only") && request.at("trace_logits_only").boolean();
  if(input.empty() || input.size()>size_t(capacity) || count>capacity || forced.size()>size_t(capacity) ||
     input.size()+std::max(size_t(count?count-1:0),forced.size())>size_t(capacity) || (count && !forced.empty()))
    throw std::runtime_error("invalid or overflowing prefill/decode request");
  // New serving contract counts the final emitted prediction too. Legacy
  // <=4096 fixture accounting remains unchanged unless explicitly opted in.
  if(long_context) lfm::text::require_budget(input.size(),size_t(count),capacity);
  if(std::set<int>(suppress.begin(),suppress.end()).size()==size_t(cfg.shape.vocabulary))
    throw std::runtime_error("cannot suppress whole vocabulary");
  lfm::TextSamplingOptions sampling;
  if(request.has("do_sample")) sampling.do_sample=request.at("do_sample").boolean();
  if(request.has("temperature")) sampling.temperature=request.at("temperature").number();
  if(request.has("top_k")) sampling.top_k=request.at("top_k").integer();
  if(request.has("top_p")) sampling.top_p=request.at("top_p").number();
  if(request.has("min_p")) sampling.min_p=request.at("min_p").number();
  if(request.has("repetition_penalty")) sampling.repetition_penalty=request.at("repetition_penalty").number();
  if(request.has("seed")) sampling.seed=request.at("seed").integer();
  if(sampling.top_k>cfg.shape.vocabulary) throw std::runtime_error("top_k exceeds vocabulary size");
  lfm::TextSampler sampler(sampling); // Validates every scalar before GPU upload.
  auto greedy=[&](const std::vector<float>& logits) {
    int best=-1;
    for(int token=0;token<cfg.shape.vocabulary;++token)
      if(std::find(suppress.begin(),suppress.end(),token)==suppress.end() &&
         (best<0 || logits[token]>logits[best])) best=token;
    if(best<0) throw std::runtime_error("cannot suppress whole vocabulary");
    return best;
  };
  auto sample=[&](lfm::TextSampler& picker,const std::vector<float>& logits,const std::vector<int>& history) {
    if(!sampling.do_sample && sampling.repetition_penalty==1.0) return greedy(logits);
    return picker.sample(logits.data(),logits.size(),history,suppress);
  };
  Runner runner(argv[2],cfg);
  runner.trace_logits_only=trace_logits_only;
  if(request.has("trace_directory")) {
    runner.trace=request.at("trace_directory").string();
    if(!runner.trace.empty()) std::filesystem::create_directories(runner.trace);
  }
  auto now=[] { return std::chrono::steady_clock::now(); };
  auto elapsed=[](auto a,auto b) { return std::chrono::duration<double>(b-a).count(); };
  const std::string trace_directory=runner.trace;
  runner.trace.clear();
  for(int run=0;run<warmups;++run) {
    lfm::TextSampler warm_sampler(sampling);
    std::vector<int> warm_history;
    auto warm_scores=runner.forward(input);
    for(int token:forced) warm_scores=runner.forward({token});
    // Warm precisely the same greedy workload as the timed run, not one decode.
    if(forced.empty()) for(int t=0;t<count;++t) {
      int token=sample(warm_sampler,warm_scores,warm_history);
      warm_history.push_back(token);
      if(std::find(stop.begin(),stop.end(),token)!=stop.end() || t==count-1) break;
      warm_scores=runner.forward({token});
    }
    runner.reset();
  }
  runner.trace=trace_directory;
  auto begin=now(); auto scores=runner.forward(input); auto prefill_end=now();
  std::vector<std::vector<float>> checkpoints;
  if(check_reset) checkpoints.push_back(scores);
  std::vector<int> generated;
  std::vector<int> argmax;
  auto pick=[&] {
    return greedy(scores); // Teacher diagnostic never consumes stochastic RNG.
  };
  argmax.push_back(pick());
  for(int token:forced) {
    scores=runner.forward({token});
    if(check_reset) checkpoints.push_back(scores);
    argmax.push_back(pick());
  }
  for(int t=0;t<count;++t) {
    int token=sample(sampler,scores,generated); generated.push_back(token);
    if(std::find(stop.begin(),stop.end(),token)!=stop.end() || t==count-1) break;
    scores=runner.forward({token});
    if(check_reset) checkpoints.push_back(scores);
  }
  auto end=now();
  // Optional same-process reset regression: exactly repeat prefill and teacher forcing.
  bool reset_ok=true;
  if(check_reset) {
    runner.trace.clear(); runner.reset(); scores=runner.forward(input);
    reset_ok=scores==checkpoints[0];
    std::vector<int> replay=forced;
    if(!generated.empty()) replay.assign(generated.begin(),generated.end()-1);
    for(size_t i=0;i<replay.size();++i) {
      scores=runner.forward({replay[i]}); reset_ok &= scores==checkpoints[i+1];
    }
    if(!reset_ok) throw std::runtime_error("reset exact-logit regression");
  }
  std::ofstream f(argv[4]);
  f<<"{\"backend\":\"native-cuda-experimental\",\"attention\":\"eager-bf16\",\"input_ids\":"; print_ids(f,input);
  f<<",\"generated_ids\":"; print_ids(f,generated); f<<",\"teacher_argmax\":"; print_ids(f,argmax);
  f<<",\"prefill_seconds\":"<<elapsed(begin,prefill_end)<<",\"decode_seconds\":"<<elapsed(prefill_end,end)
   <<",\"decode_forward_calls\":"<<(forced.empty()?std::max(0,int(generated.size())-1):int(forced.size()))
   <<",\"reset_checked\":"<<(check_reset?"true":"false")
   <<",\"reset_exact_logits_ok\":"<<(check_reset?(reset_ok?"true":"false"):"null")
   <<",\"warmup_runs\":"<<warmups<<",\"tracing\":"<<(trace_directory.empty()?"false":"true")
   <<",\"do_sample\":"<<(sampling.do_sample?"true":"false")<<",\"seed\":"<<sampling.seed<<"}\n";
  if(!f) throw std::runtime_error("output write failed");
  return 0;
} catch(const std::exception& e) { std::cerr<<"lfm3: "<<e.what()<<"\n"; return 1; }
#endif
