#pragma once
// The split multiply on the tensor cores: y = x·Wᵀ (+ bias) with the FP32 activations x as BF16 parts against the
// stored BF16 weights W, FP32 accumulation and output (validation/images-speed-v1, where the vision encoder introduced
// it; shared since validation/audio-ttft-v1 with the audio encoder). Parts:
// - 1 (x1): x rounded to BF16, one GEMM;
// - 2 (x2): hi = bf16(x), lo = bf16(x - hi), so hi + lo holds x to about 2^-16 of it; one GEMM over a doubled K,
//   [hi | lo] against [W | W], which sums hi·W + lo·W in one accumulation. [W | W] is not stored: the GEMM reads W's
//   k index modulo `in` (lfm::gemm::Problem::b_k_wrap; validation/audio-ttft-v1 amendment 5), the same values in the
//   same order as the copy the vision encoder still builds.
// The kernels below are the vision encoder's, moved here unchanged.
#include <lfm/fp32_ops.cuh>
#include <lfm/gemm.cuh>

#include <cstddef>

namespace lfm::split {

using bf16 = __nv_bfloat16;

namespace kernels {
// A float as BF16 parts: hi = bf16(v); lo = bf16(v - hi) (x2), so hi + lo holds v to about 2^-16 of it.
__device__ __forceinline__ void split_bf16(float v, bf16& hi, bf16& lo) {
  hi = __float2bfloat16_rn(v);
  lo = __float2bfloat16_rn(__fsub_rn(v, __bfloat162float(hi)));
}
// x [rows][in] FP32 -> lfm::gemm's A operand [rows][parts * in]: hi in columns 0..in, lo (x2) in in..2 in, so one
// GEMM against [W | W] (double_rows) sums hi·W + lo·W.
__global__ void split_rows(const float* x, bf16* out, int rows, int in, int parts) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * in) return;
  const size_t r = i / in, c = i % in;
  bf16 hi, lo;
  split_bf16(x[i], hi, lo);
  out[r * parts * in + c] = hi;
  if (parts == 2) out[r * parts * in + in + c] = lo;
}
// [W | W]: each row of W [rows][in] twice, the B operand of the split multiply
__global__ void double_rows(const bf16* w, bf16* out, int rows, int in) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * in) return;
  const size_t r = i / in, c = i % in;
  out[r * 2 * in + c] = out[r * 2 * in + in + c] = w[i];
}
// y[i] = c[i] + bias[i % cols]
__global__ void add_bias(const float* c, const bf16* bias, size_t n, int cols, float* y) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __fadd_rn(c[i], __bfloat162float(bias[i % cols]));
}
}  // namespace kernels

// The multiply with its state: the scratch for the BF16 operand and the pre-bias output (kept until release()).
class Linear {
 public:
  // y [rows][out] = x [rows][in] · Wᵀ (+ bias), x as `parts` (1 or 2) BF16 parts, on the tensor cores.
  void mul(const float* x, const bf16* W, const bf16* bias, float* y, int rows, int out, int in, int parts) {
    namespace f = lfm::fp32;
    const size_t a = size_t(rows) * parts * in, c = size_t(rows) * out;
    if (operand_.n < a) operand_ = f::DeviceArray<bf16>(a);
    kernels::split_rows<<<f::blocks(size_t(rows) * in), 256>>>(x, operand_.p, rows, in, parts);
    float* target = y;
    if (bias) {
      if (unbiased_.n < c) unbiased_ = f::DeviceArray<float>(c);
      target = unbiased_.p;
    }
    lfm::gemm::Problem p = lfm::gemm::linear(operand_.p, W, nullptr, rows, parts * in, out);
    p.ldb = in;  // W [out][in]; over the doubled K, its k index wraps (x2)
    if (parts == 2) p.b_k_wrap = in;
    p.c = nullptr, p.c_f32 = target;
    lfm::gemm::run(p);
    if (bias) kernels::add_bias<<<f::blocks(c), 256>>>(target, bias, c, out, y);
  }
  // the scratch freed; the caller has synchronized
  void release() {
    operand_ = lfm::fp32::DeviceArray<bf16>();
    unbiased_ = lfm::fp32::DeviceArray<float>();
  }
  size_t scratch_bytes() const { return operand_.bytes() + unbiased_.bytes(); }

 private:
  lfm::fp32::DeviceArray<bf16> operand_;
  lfm::fp32::DeviceArray<float> unbiased_;
};

}  // namespace lfm::split
