#pragma once
// Native LFM3-Omni depth decoder (HF `Lfm3OmniDepthDecoder`): from the language model's hidden state and a frame's
// revealed codes, the 2017-way logits of each of the frame's 12 codebooks. #2 step 3, quality bars A1/B1
// (journal/lfm3-tts-quality-bar.md).
//
// Per frame: depth_linear projects the hidden state (2048) to 12 slots of 1024, once (HF recomputes it every pass;
// the input is the same, so is the result). Per pass: each slot adds its codebook's code embedding, or the mask
// embedding while hidden; a 3-layer bidirectional LFM3 trunk (full attention, 32 query / 8 KV heads of 32, FFN
// 2816, no sandwich norm) and its final norm; then each requested codebook's head: RMSNorm with the codebook's own
// weight, and logits against its tied embedding [2017, 1024].
//
// Precision as the codec: FP32 activations over the stored BF16 weights (lfm/fp32_ops.cuh), rotary in FP64.
#include <lfm/fp32_ops.cuh>
#include <lfm/safetensors.hpp>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm::depth {

constexpr int kModelHidden = 2048, kCodebooks = 12, kCodes = 2017, kHidden = 1024, kFfn = 2816, kLayers = 3;
constexpr int kHeads = 32, kKvHeads = 8, kHeadDim = 32;
constexpr double kRopeTheta = 1e6;
constexpr float kNormEps = 1e-5f;

namespace kernels {
using bf16 = __nv_bfloat16;

__global__ void widen(const bf16* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __bfloat162float(x[i]);
}

struct Tables { const bf16* embedding[kCodebooks]; };
// slots[c] = projected[c] + (masked[c] ? mask_embedding : embedding_c[code[c]])
__global__ void slot_inputs(const float* projected, Tables tables, const bf16* mask_embedding, const int32_t* state,
                            float* slots) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kCodebooks * kHidden) return;
  const int c = i / kHidden, d = i % kHidden, code = state[c], masked = state[kCodebooks + c];
  const bf16 e = masked ? mask_embedding[d] : tables.embedding[c][size_t(code) * kHidden + d];
  slots[i] = __fadd_rn(projected[i], __bfloat162float(e));
}

__global__ void add(float* a, const float* b, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = __fadd_rn(a[i], b[i]);
}

// One warp per (slot, query head), a lane per head dimension: bidirectional softmax(q.k * 32^-0.5) over the 12
// slots, times V, times sigmoid(gate). Query head h reads KV head h / 4.
__global__ void attention(const float* q, const float* k, const float* v, const float* gate, float* out) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (warp >= kCodebooks * kHeads) return;
  const int t = warp / kHeads, h = warp % kHeads, kv = h / (kHeads / kKvHeads);
  const float qv = q[size_t(warp) * kHeadDim + lane], scale = 0.17677669529663687f;  // float(32 ** -0.5), as HF
  float scores[kCodebooks], top = -INFINITY;
#pragma unroll
  for (int s = 0; s < kCodebooks; ++s) {
    float dot = __fmul_rn(qv, k[(size_t(s) * kKvHeads + kv) * kHeadDim + lane]);
    for (int o = 16; o; o /= 2) dot = __fadd_rn(dot, __shfl_down_sync(0xffffffffu, dot, o));
    scores[s] = __fmul_rn(__shfl_sync(0xffffffffu, dot, 0), scale);
    top = fmaxf(top, scores[s]);
  }
  float total = 0.0f;
#pragma unroll
  for (int s = 0; s < kCodebooks; ++s) total = __fadd_rn(total, scores[s] = expf(__fsub_rn(scores[s], top)));
  float o = 0.0f;
#pragma unroll
  for (int s = 0; s < kCodebooks; ++s)
    o = __fmaf_rn(__fdiv_rn(scores[s], total), v[(size_t(s) * kKvHeads + kv) * kHeadDim + lane], o);
  const float g = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gate[size_t(t) * kHeads + h])));
  out[size_t(t) * kHeads * kHeadDim + h * kHeadDim + lane] = __fmul_rn(o, g);
}
}  // namespace kernels

class Decoder {
 public:
  using bf16 = __nv_bfloat16;

  // `weights`: the main checkpoint (model.safetensors), shared with the text engine when both run.
  explicit Decoder(std::shared_ptr<Safetensors> weights) : weights_(std::move(weights)) {
    const std::string p = "model.depth_decoder.";
    depth_linear_ = tensor(p + "depth_linear.weight", {kCodebooks * kHidden, kModelHidden});
    depth_linear_bias_ = tensor(p + "depth_linear.bias", {kCodebooks * kHidden});
    mask_embedding_ = tensor(p + "mask_embedding", {kHidden});
    final_norm_ = tensor(p + "model.embedding_norm.weight", {kHidden});
    std::vector<bf16> head_norms(size_t(kCodebooks) * kHidden);
    for (int c = 0; c < kCodebooks; ++c) {
      const std::string e = p + "depth_embeddings." + std::to_string(c) + ".";
      tables_.embedding[c] = tensor(e + "embedding.weight", {kCodes, kHidden});
      fp32::check(cudaMemcpy(head_norms.data() + size_t(c) * kHidden, tensor(e + "embedding_norm.weight", {kHidden}),
                             kHidden * sizeof(bf16), cudaMemcpyDeviceToHost), "depth head norms");
    }
    head_norms_ = fp32::DeviceArray<bf16>(head_norms.size());  // contiguous, one row per codebook (24 KiB)
    fp32::check(cudaMemcpy(head_norms_.p, head_norms.data(), head_norms.size() * sizeof(bf16), cudaMemcpyHostToDevice),
                "depth head norms");
    for (int i = 0; i < kLayers; ++i) {
      const std::string l = p + "model.layers." + std::to_string(i) + ".";
      Layer& y = layers_[i];
      y.operator_norm = tensor(l + "operator_norm.weight", {kHidden});
      y.ffn_norm = tensor(l + "ffn_norm.weight", {kHidden});
      y.q = tensor(l + "self_attn.q_proj.weight", {kHeads * kHeadDim, kHidden});
      y.k = tensor(l + "self_attn.k_proj.weight", {kKvHeads * kHeadDim, kHidden});
      y.v = tensor(l + "self_attn.v_proj.weight", {kKvHeads * kHeadDim, kHidden});
      y.o = tensor(l + "self_attn.o_proj.weight", {kHidden, kHeads * kHeadDim});
      y.gate = tensor(l + "self_attn.gate_proj.weight", {kHeads, kHidden});
      y.q_norm = tensor(l + "self_attn.q_norm.weight", {kHeadDim});
      y.k_norm = tensor(l + "self_attn.k_norm.weight", {kHeadDim});
      y.w1 = tensor(l + "feed_forward.w1.weight", {kFfn, kHidden});
      y.w3 = tensor(l + "feed_forward.w3.weight", {kFfn, kHidden});
      y.w2 = tensor(l + "feed_forward.w2.weight", {kHidden, kFfn});
    }
    if (weights_->has(p + "model.layers.3.operator_norm.weight")) throw std::runtime_error("depth: expected 3 layers");
  }

  // Start a frame from the language model's hidden state (BF16 [2048], on the device).
  void begin_frame(const bf16* hidden) {
    kernels::widen<<<fp32::blocks(kModelHidden), 256>>>(hidden, hidden_.p, kModelHidden);
    fp32::linear(hidden_.p, depth_linear_, depth_linear_bias_, projected_.p, 1, kCodebooks * kHidden, kModelHidden);
  }

  // One pass: `codes` and `masked` (host, 12 each). Writes the FP32 logits of the codebooks c with score[c] set to
  // logits + c * 2017 (device). The others are not computed.
  void pass(const int32_t* codes, const bool* masked, const bool* score, float* logits) {
    int32_t state[2 * kCodebooks];
    for (int c = 0; c < kCodebooks; ++c) {
      if (codes[c] < 0 || codes[c] >= kCodes) throw std::runtime_error("depth: code out of range [0, 2017)");
      state[c] = codes[c];
      state[kCodebooks + c] = masked[c];
    }
    fp32::check(cudaMemcpy(state_.p, state, sizeof state, cudaMemcpyHostToDevice), "depth state");
    constexpr int R = kCodebooks;
    kernels::slot_inputs<<<fp32::blocks(R * kHidden), 256>>>(projected_.p, tables_, mask_embedding_, state_.p, x_.p);
    for (const Layer& y : layers_) {
      fp32::rms_norm(x_.p, y.operator_norm, normed_.p, nullptr, R, kHidden, kNormEps);
      float *q = q_.p, *k = k_.p, *v = v_.p, *gate = gate_.p;
      fp32::linear(normed_.p, y.q, nullptr, q, R, kHeads * kHeadDim, kHidden);
      fp32::linear(normed_.p, y.k, nullptr, k, R, kKvHeads * kHeadDim, kHidden);
      fp32::linear(normed_.p, y.v, nullptr, v, R, kKvHeads * kHeadDim, kHidden);
      fp32::linear(normed_.p, y.gate, nullptr, gate, R, kHeads, kHidden);
      fp32::rms_norm(q, y.q_norm, q, nullptr, R * kHeads, kHeadDim, kNormEps);
      fp32::rms_norm(k, y.k_norm, k, nullptr, R * kKvHeads, kHeadDim, kNormEps);
      fp32::rotary(q, R, kHeads, kHeadDim, 0, kRopeTheta);
      fp32::rotary(k, R, kKvHeads, kHeadDim, 0, kRopeTheta);
      kernels::attention<<<fp32::blocks(size_t(R) * kHeads * 32), 256>>>(q, k, v, gate, attended_.p);
      fp32::linear(attended_.p, y.o, nullptr, out_.p, R, kHidden, kHeads * kHeadDim);
      kernels::add<<<fp32::blocks(R * kHidden), 256>>>(x_.p, out_.p, R * kHidden);
      fp32::rms_norm(x_.p, y.ffn_norm, normed_.p, nullptr, R, kHidden, kNormEps);
      fp32::linear(normed_.p, y.w1, nullptr, ffn_a_.p, R, kFfn, kHidden);
      fp32::linear(normed_.p, y.w3, nullptr, ffn_b_.p, R, kFfn, kHidden);
      fp32::swiglu(ffn_a_.p, ffn_b_.p, size_t(R) * kFfn);
      fp32::linear(ffn_a_.p, y.w2, nullptr, out_.p, R, kHidden, kFfn);
      kernels::add<<<fp32::blocks(R * kHidden), 256>>>(x_.p, out_.p, R * kHidden);
    }
    fp32::rms_norm(x_.p, final_norm_, normed_.p, nullptr, R, kHidden, kNormEps);
    // the heads' norm: each codebook's own weight (HF Lfm3OmniRMSNorm; in FP32 the same arithmetic as above)
    fp32::rms_norm(normed_.p, head_norms_.p, head_in_.p, nullptr, R, kHidden, kNormEps, kHidden);
    for (int c = 0; c < kCodebooks; ++c)
      if (score[c])
        fp32::linear(head_in_.p + size_t(c) * kHidden, tables_.embedding[c], nullptr, logits + size_t(c) * kCodes, 1,
                     kCodes, kHidden);
    fp32::check(cudaGetLastError(), "depth pass");
  }

  size_t buffer_bytes() const {
    return hidden_.bytes() + projected_.bytes() + state_.bytes() + x_.bytes() + normed_.bytes() + q_.bytes() +
           k_.bytes() + v_.bytes() + gate_.bytes() + attended_.bytes() + out_.bytes() + ffn_a_.bytes() +
           ffn_b_.bytes() + head_in_.bytes() + head_norms_.bytes();
  }

 private:
  struct Layer {
    const bf16 *operator_norm, *ffn_norm, *q, *k, *v, *o, *gate, *q_norm, *k_norm, *w1, *w2, *w3;
  };
  const bf16* tensor(const std::string& name, std::vector<int64_t> shape) const {
    const auto& t = weights_->at(name);
    if (t.dtype != "BF16" || t.shape != shape) throw std::runtime_error("depth: unexpected tensor " + name);
    return t.device_data;
  }

  std::shared_ptr<Safetensors> weights_;
  const bf16 *depth_linear_, *depth_linear_bias_, *mask_embedding_, *final_norm_;
  kernels::Tables tables_{};
  Layer layers_[kLayers];
  fp32::DeviceArray<bf16> head_norms_;
  fp32::DeviceArray<int32_t> state_{2 * kCodebooks};
  fp32::DeviceArray<float> hidden_{kModelHidden}, projected_{kCodebooks * kHidden}, x_{kCodebooks * kHidden},
      normed_{kCodebooks * kHidden}, q_{kCodebooks * kHeads * kHeadDim}, k_{kCodebooks * kKvHeads * kHeadDim},
      v_{kCodebooks * kKvHeads * kHeadDim}, gate_{kCodebooks * kHeads}, attended_{kCodebooks * kHeads * kHeadDim},
      out_{kCodebooks * kHidden}, ffn_a_{size_t(kCodebooks) * kFfn}, ffn_b_{size_t(kCodebooks) * kFfn},
      head_in_{kCodebooks * kHidden};
};

}  // namespace lfm::depth
