// gemm-wrap-check (validation/encoder-wrap-v1 rule 1; first written for audio-ttft-v1 amendment 5, c968b5d):
// lfm::gemm with b_k_wrap against the same GEMM over an explicit [W | W] copy, on random BF16 inputs, every output
// compared bit for bit, FP32 and BF16 outputs.
//   mma.sync (the tcgen05 path off): c968b5d's 154 cases, the old audio encoder's shapes on row counts that take the
//     thin path (M <= 8) and each wide tile (M <= 32, <= 64, larger);
//   tcgen05: both encoders' x2 shapes with out >= 512 and out % 256 == 0 at M 257, 300, 1125 and 4500; the tcgen05
//     launch counter must rise by two on each (the wrapped and the explicit GEMM: the positive control); then the same
//     cases with the path off (mma.sync);
//   a tcgen05-sized problem whose wrap is not a multiple of 64 takes mma.sync (no launch) and matches;
//   the invalid uses are refused (c968b5d's five).
// Built with -DLFM_GEMM_TCGEN05 and linked with gemm_tcgen05.o. Prints one line per failure and a summary.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "lfm/gemm.cuh"

#ifndef LFM_GEMM_TCGEN05
#error "build with -DLFM_GEMM_TCGEN05"
#endif

namespace {
using bf16 = __nv_bfloat16;
namespace tc = lfm::gemm::tcgen05;

void cuda_check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    std::exit(3);
  }
}
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

std::mt19937 rng(11);
int cases = 0, bad = 0, control_bad = 0;

// One (rows, in, out): the wrapped GEMM against the explicit [W | W], FP32 and BF16 out. `launches`: the tcgen05
// launches each pair must add (2 on the tcgen05 path, 0 off it); -1 to skip that test.
void compare(int rows, int in, int out, int launches, const char* label) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> a(size_t(rows) * 2 * in), w(size_t(out) * in), ww(size_t(out) * 2 * in);
  for (auto& x : a) x = __float2bfloat16_rn(nd(rng));
  for (auto& x : w) x = __float2bfloat16_rn(nd(rng) * 0.05f);
  for (int r = 0; r < out; ++r)
    for (int c = 0; c < in; ++c) ww[size_t(r) * 2 * in + c] = ww[size_t(r) * 2 * in + in + c] = w[size_t(r) * in + c];
  bf16 *da = upload(a), *dw = upload(w), *dww = upload(ww);
  const size_t nc = size_t(rows) * out;
  float *f1, *f2;
  bf16 *h1, *h2;
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
    cuda_check(cudaMemset(f32 ? (void*)f1 : (void*)h1, 0xff, nc * (f32 ? 4 : 2)));
    cuda_check(cudaMemset(f32 ? (void*)f2 : (void*)h2, 0x7f, nc * (f32 ? 4 : 2)));
    const unsigned long long before = tc::launch_count();
    lfm::gemm::run(p1);
    lfm::gemm::run(p2);
    cuda_check(cudaDeviceSynchronize());
    const unsigned long long added = tc::launch_count() - before;
    const bool same = f32 ? std::memcmp(download(f1, nc).data(), download(f2, nc).data(), nc * 4) == 0
                          : std::memcmp(download(h1, nc).data(), download(h2, nc).data(), nc * 2) == 0;
    ++cases;
    if (!same) {
      ++bad;
      std::printf("%s rows %4d, in %4d, out %4d, %s: DIFFERENT\n", label, rows, in, out, f32 ? "fp32" : "bf16");
    }
    if (launches >= 0 && added != (unsigned long long)launches) {
      ++control_bad;
      std::printf("%s rows %4d, in %4d, out %4d, %s: %llu tcgen05 launches, expected %d\n", label, rows, in, out,
                  f32 ? "fp32" : "bf16", added, launches);
    }
  }
  for (void* q : {(void*)da, (void*)dw, (void*)dww, (void*)f1, (void*)f2, (void*)h1, (void*)h2}) cudaFree(q);
}
}  // namespace

int main() {
  // 1. mma.sync: c968b5d's cases. (in, out): the old audio encoder's projections (q/k/v/out 768x768, conv in
  // 768->1536, w1/w3 768->1792, w2 1792->768, the downsampler's 4096->768, the projector's 768->2048 and 2048->2048).
  tc::set_enabled(false);
  const int mma_shapes[][2] = {{768, 768}, {768, 1536}, {768, 1792}, {1792, 768}, {4096, 768}, {768, 2048}, {2048, 2048}};
  for (int rows : {1, 3, 8, 9, 27, 32, 33, 64, 65, 131, 1125})
    for (const auto& sh : mma_shapes) compare(rows, sh[0], sh[1], 0, "mma.sync");
  const int mma_cases = cases, mma_bad = bad;
  // 2. tcgen05: both encoders' x2 shapes with out >= 512 and out % 256 == 0. Audio: q/k/v/out 768x768, conv in
  // 768->2304, w1/w3 768->1792, w2 1792->768, lin_out 4096->768, projector 768->2048, 2048->2048. Vision: patch
  // 1536->1024, attention 1024->1024, w1 1024->2560, w2 2560->1024, projector 4096->4096, 4096->2048.
  const int tc_shapes[][2] = {{768, 768},   {768, 2304},   {768, 1792},   {1792, 768},  {4096, 768},
                              {768, 2048},  {2048, 2048},  {1536, 1024},  {1024, 1024}, {1024, 2560},
                              {2560, 1024}, {4096, 4096},  {4096, 2048}};
  tc::set_enabled(true);
  for (int rows : {257, 300, 1125, 4500})
    for (const auto& sh : tc_shapes) compare(rows, sh[0], sh[1], 2, "tcgen05");
  const int tc_cases = cases - mma_cases, tc_bad = bad - mma_bad;
  tc::set_enabled(false);
  for (int rows : {257, 300, 1125, 4500})
    for (const auto& sh : tc_shapes) compare(rows, sh[0], sh[1], 0, "tcgen05-off");
  tc::set_enabled(true);
  // 3. a wrap that is not a multiple of 64 on a tcgen05-sized problem: in 544 (k 1088 = 17 x 64), out 1024. The
  // explicit [W | W] takes tcgen05 (ldb == k); the wrapped one must not: one launch per pair.
  compare(1125, 544, 1024, 1, "wrap-not-64");
  const int off_cases = cases - mma_cases - tc_cases, off_bad = bad - mma_bad - tc_bad;
  // 4. refusals: a wrap that is not a multiple of 8; ldb below it; a K x N B; the K = 128 decode path; k over twice it
  int refused = 0;
  {
    std::vector<bf16> z(64 * 256);
    bf16* d = upload(z);
    float* c;
    cuda_check(cudaMalloc(&c, 64 * 256 * 4));
    for (int variant = 0; variant < 5; ++variant) {
      lfm::gemm::Problem p = lfm::gemm::linear(d, d, nullptr, variant == 3 ? 4 : 16, 128, 16);
      p.c_f32 = c;
      p.ldb = 64;
      p.b_k_wrap = 64;
      if (variant == 0) p.b_k_wrap = 60;
      if (variant == 1) p.ldb = 56;
      if (variant == 2) p.b_k_contiguous = false, p.b_rows = 128;
      if (variant == 4) p.b_k_wrap = 56, p.ldb = 56;  // k 128 > 2 x 56
      try {
        lfm::gemm::run(p);
        cuda_check(cudaDeviceSynchronize());
        std::printf("refusal %d: NOT REFUSED\n", variant);
      } catch (const std::runtime_error&) {
        ++refused;
      }
    }
    cudaFree(d);
    cudaFree(c);
  }
  std::printf("mma.sync: %d of %d identical; tcgen05: %d of %d identical; tcgen05 off and the wrap not a multiple of 64: "
              "%d of %d identical; launch-count mismatches %d; %d of 5 invalid uses refused\n",
              mma_cases - mma_bad, mma_cases, tc_cases - tc_bad, tc_cases, off_cases - off_bad, off_cases, control_bad,
              refused);
  const bool pass = !bad && !control_bad && refused == 5;
  std::printf("%s\n", pass ? "GEMM WRAP CHECK PASS" : "GEMM WRAP CHECK FAIL");
  return pass ? 0 : 1;
}
