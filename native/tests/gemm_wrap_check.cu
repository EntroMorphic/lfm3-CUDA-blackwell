// gemm-wrap-check: a development check (validation/audio-ttft-v1 amendment 5): lfm::gemm with b_k_wrap against the
// same GEMM over an explicit [W | W] copy (the split multiply's x2 before the amendment), on random BF16 inputs, every
// output compared bit for bit, FP32 and BF16 outputs, on the audio encoder's shapes and on row counts that take the
// thin path (M <= 8) and each wide tile (M <= 32, <= 64, larger). Also: the invalid uses are refused.
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include <lfm/gemm.cuh>

#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using bf16 = __nv_bfloat16;

template <class T>
T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  cuda_check(cudaMalloc(&d, h.size() * sizeof(T)));
  cuda_check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <class T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  cuda_check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
}  // namespace

int main() {
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  int bad = 0, cases = 0;
  // (in, out): the encoder's projections (q/k/v/out 768x768, conv in 768->1536, conv out, w1/w3 768->1792,
  // w2 1792->768, the downsampler's 4096->768, the projector's 768->2048 and 2048->2048)
  const int shapes[][2] = {{768, 768}, {768, 1536}, {768, 1792}, {1792, 768}, {4096, 768}, {768, 2048}, {2048, 2048}};
  for (int rows : {1, 3, 8, 9, 27, 32, 33, 64, 65, 131, 1125}) {
    for (const auto& sh : shapes) {
      const int in = sh[0], out = sh[1];
      std::vector<bf16> a(size_t(rows) * 2 * in), w(size_t(out) * in), ww(size_t(out) * 2 * in);
      for (auto& x : a) x = __float2bfloat16_rn(nd(rng));
      for (auto& x : w) x = __float2bfloat16_rn(nd(rng) * 0.05f);
      for (int r = 0; r < out; ++r)
        for (int c = 0; c < in; ++c) ww[size_t(r) * 2 * in + c] = ww[size_t(r) * 2 * in + in + c] = w[size_t(r) * in + c];
      bf16 *da = upload(a), *dw = upload(w), *dww = upload(ww);
      float *f1, *f2;
      bf16 *h1, *h2;
      const size_t nc = size_t(rows) * out;
      cuda_check(cudaMalloc(&f1, nc * 4));
      cuda_check(cudaMalloc(&f2, nc * 4));
      cuda_check(cudaMalloc(&h1, nc * 2));
      cuda_check(cudaMalloc(&h2, nc * 2));
      for (bool f32 : {true, false}) {
        lfm::gemm::Problem p1 = lfm::gemm::linear(da, dww, f32 ? nullptr : h1, rows, 2 * in, out);  // [W | W], ldb 2 in
        lfm::gemm::Problem p2 = lfm::gemm::linear(da, dw, f32 ? nullptr : h2, rows, 2 * in, out);
        p2.ldb = in;
        p2.b_k_wrap = in;
        if (f32) p1.c_f32 = f1, p2.c_f32 = f2;
        lfm::gemm::run(p1);
        lfm::gemm::run(p2);
        cuda_check(cudaDeviceSynchronize());
        const bool same = f32 ? download(f1, nc) == download(f2, nc)
                              : std::memcmp(download(h1, nc).data(), download(h2, nc).data(), nc * 2) == 0;
        ++cases;
        if (!same) {
          ++bad;
          std::printf("rows %4d, in %4d, out %4d, %s: DIFFERENT\n", rows, in, out, f32 ? "fp32" : "bf16");
        }
      }
      for (void* q : {(void*)da, (void*)dw, (void*)dww, (void*)f1, (void*)f2, (void*)h1, (void*)h2}) cudaFree(q);
    }
    std::printf("rows %4d: done\n", rows);
  }
  // refusals: a wrap that is not a multiple of 8; ldb below it; a K x N B; the K = 128 decode path
  int refused = 0;
  {
    std::vector<bf16> z(64 * 256);
    bf16* d = upload(z);
    float* c;
    cuda_check(cudaMalloc(&c, 64 * 256 * 4));
    auto bad_problem = [&](int variant) {
      lfm::gemm::Problem p = lfm::gemm::linear(d, d, nullptr, variant == 3 ? 4 : 16, variant == 3 ? 128 : 128, 16);
      p.c_f32 = c;
      p.ldb = 64;
      p.b_k_wrap = 64;
      if (variant == 0) p.b_k_wrap = 60;
      if (variant == 1) p.ldb = 56;
      if (variant == 2) p.b_k_contiguous = false, p.b_rows = 128;
      try {
        lfm::gemm::run(p);
        cuda_check(cudaDeviceSynchronize());
        std::printf("refusal %d: NOT REFUSED\n", variant);
      } catch (const std::runtime_error&) {
        ++refused;
      }
    };
    for (int v = 0; v < 4; ++v) bad_problem(v);
    cudaFree(d);
    cudaFree(c);
  }
  std::printf("%d of %d cases identical; %d of 4 invalid uses refused\n", cases - bad, cases, refused);
  std::printf("%s\n", bad || refused != 4 ? "GEMM WRAP CHECK FAIL" : "GEMM WRAP CHECK PASS");
  return bad || refused != 4;
}
