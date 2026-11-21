// audio-attention-check: a development check (validation/audio-ttft-v1 amendment 1): window_attention_tiled against window_attention on random
// inputs, every float compared bit for bit, at several lengths.
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include <lfm/audio_encoder.cuh>
#include <cstdio>
#include <cstring>
#include <random>
int main() {
  using namespace lfm3_audio_in::kernels;
  const int heads = 8, kv = 4, dim = 96;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  int bad = 0;
  for (int rows : {1, 2, 31, 32, 33, 63, 64, 65, 100, 1125, 1149}) {
    std::vector<float> q(size_t(rows) * heads * dim), k(size_t(rows) * kv * dim), v(k.size());
    for (auto* a : {&q, &k, &v}) for (auto& x : *a) x = nd(rng);
    float *dq, *dk, *dv, *o1, *o2;
    const size_t no = size_t(rows) * heads * dim;
    cudaMalloc(&dq, q.size() * 4); cudaMalloc(&dk, k.size() * 4); cudaMalloc(&dv, v.size() * 4);
    cudaMalloc(&o1, no * 4); cudaMalloc(&o2, no * 4);
    cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
    const float scale = float(1.0 / std::sqrt(double(dim)));
    window_attention<<<int((size_t(rows) * heads * 32 + 255) / 256), 256>>>(dq, dk, dv, o1, rows, heads, kv, dim, scale);
    window_attention_tiled<<<dim3((rows + 31) / 32, heads), 256>>>(dq, dk, dv, o2, rows, heads, kv, scale);
    std::vector<float> a(no), b(no);
    cudaMemcpy(a.data(), o1, no * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), o2, no * 4, cudaMemcpyDeviceToHost);
    const bool same = std::memcmp(a.data(), b.data(), no * 4) == 0;
    std::printf("rows %5d: %s\n", rows, same ? "identical" : "DIFFERENT");
    bad += !same;
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(o1); cudaFree(o2);
  }
  std::printf("%s\n", bad ? "ATTENTION CHECK FAIL" : "ATTENTION CHECK PASS");
  return bad != 0;
}
