// alloc-placement-probe: validation/alloc-effect-v1 stage 1 (a diagnostic). Does the slowed prefill GEMM's speed follow
// the buffer (where the driver placed its memory) or the time (a GPU state that changes)? In one process, back to back,
// with no idle waits:
//   B: COUNT_B fresh weight buffers (N 7680 x K 2048 BF16, 30 MiB, the shape slowed 1.78x), each allocated and kept;
//      lfm::gemm at M 1149 (grid 60 x 9) timed on each in allocation order (pass 1), in reverse (pass 2), and in
//      allocation order again (pass 3), with one fixed activation buffer. Pass 1 also times a one-thread pointer chase
//      over the buffer, one hop per 64 KiB region in a shuffled order (ns per hop: GPU address translation cost).
//   A: COUNT_A fresh activation buffers (M 1149 x K 2048 BF16, 4.5 MiB, re-read by all 60 column tiles), each kept, with
//      one fixed weight buffer, timed in allocation order and in reverse.
//   control: the fixed pair, timed every 8 buffers (a time series of the GPU's state).
// Each GEMM time is the median of 5 launches after 1 untimed (CUDA events). /proc/buddyinfo before and after.
// One JSON line per buffer, per control reading, and for buddyinfo.
// Usage: alloc-placement-probe [COUNT_B] [COUNT_A]   (default 96 64)
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include <lfm/gemm.cuh>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {
using bf16 = __nv_bfloat16;
__global__ void fill(bf16* p, size_t n, float scale, unsigned salt) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2bfloat16_rn(scale * float(int((i + salt) % 1013) - 506));
}
__global__ void chase(const uint32_t* p, uint32_t start, int hops, uint32_t* out) {
  uint32_t x = start;
  for (int h = 0; h < hops; ++h) x = p[x];
  *out = x;
}
uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
std::string buddyinfo() {
  std::ifstream f("/proc/buddyinfo");
  std::string line, out;
  while (std::getline(f, line))
    if (line.find("Normal") != std::string::npos) out = line.substr(line.find("Normal") + 6);
  return out;
}
}  // namespace

int main(int argc, char** argv) try {
  const int count_b = argc > 1 ? std::atoi(argv[1]) : 96, count_a = argc > 2 ? std::atoi(argv[2]) : 64;
  const int M = 1149, N = 7680, K = 2048;
  const size_t a_n = size_t(M) * K, b_n = size_t(N) * K;
  std::printf("{\"buddyinfo_normal_before\":\"%s\"}\n", buddyinfo().c_str());
  cudaEvent_t e0, e1;
  cuda_check(cudaEventCreate(&e0));
  cuda_check(cudaEventCreate(&e1));
  bf16 *a0, *b0, *c;
  uint32_t* sink;
  cuda_check(cudaMalloc(&a0, a_n * 2));
  cuda_check(cudaMalloc(&b0, b_n * 2));
  cuda_check(cudaMalloc(&c, size_t(M) * N * 2));
  cuda_check(cudaMalloc(&sink, 4));
  fill<<<int((a_n + 255) / 256), 256>>>(a0, a_n, 1e-3f, 0);
  fill<<<int((b_n + 255) / 256), 256>>>(b0, b_n, 1e-3f, 0);
  cuda_check(cudaDeviceSynchronize());
  auto elapsed = [&](auto&& f) {
    cuda_check(cudaEventRecord(e0));
    f();
    cuda_check(cudaEventRecord(e1));
    cuda_check(cudaEventSynchronize(e1));
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
  };
  auto gemm_ms = [&](bf16* a, bf16* b) {
    const lfm::gemm::Problem p = lfm::gemm::linear(a, b, c, M, K, N);
    lfm::gemm::run(p);
    std::vector<float> t;
    for (int r = 0; r < 5; ++r) t.push_back(elapsed([&] { lfm::gemm::run(p); }));
    std::sort(t.begin(), t.end());
    return t[2];
  };
  int control_n = 0;
  auto control = [&](const char* where) {
    std::printf("{\"control\":%d,\"where\":\"%s\",\"ms\":%.4f}\n", control_n++, where, gemm_ms(a0, b0));
    std::fflush(stdout);
  };
  // the chase: one uint32 per 64 KiB region, the regions in a shuffled cycle; the time of a second launch
  const size_t region_words = (size_t(64) << 10) / 4, regions = b_n * 2 / (size_t(64) << 10);
  std::vector<uint32_t> order(regions), host(b_n / 2, 0);
  auto chase_ns = [&](bf16* b, unsigned seed) {
    for (size_t r = 0; r < regions; ++r) order[r] = uint32_t(r);
    for (size_t r = regions - 1; r > 0; --r) std::swap(order[r], order[mix64(seed * 1000003ull + r) % (r + 1)]);
    std::fill(host.begin(), host.end(), 0u);
    for (size_t r = 0; r < regions; ++r)
      host[order[r] * region_words] = uint32_t(order[(r + 1) % regions] * region_words);
    cuda_check(cudaMemcpy(b, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    const int hops = int(regions) * 4;
    chase<<<1, 1>>>(reinterpret_cast<uint32_t*>(b), uint32_t(order[0] * region_words), hops, sink);
    const float ms = elapsed([&] {
      chase<<<1, 1>>>(reinterpret_cast<uint32_t*>(b), uint32_t(order[0] * region_words), hops, sink);
    });
    return double(ms) * 1e6 / hops;
  };
  gemm_ms(a0, b0);  // first launches out of the way
  control("start");
  // B: weight buffers
  std::vector<bf16*> bs(count_b);
  std::vector<double> chase_b(count_b), b1(count_b), b2(count_b), b3(count_b);
  for (int i = 0; i < count_b; ++i) {
    cuda_check(cudaMalloc(&bs[i], b_n * 2));
    chase_b[i] = chase_ns(bs[i], unsigned(i + 1));
    fill<<<int((b_n + 255) / 256), 256>>>(bs[i], b_n, 1e-3f, unsigned(i));
    b1[i] = gemm_ms(a0, bs[i]);
    if (i % 8 == 7) control("B pass 1");
  }
  for (int i = count_b - 1; i >= 0; --i) {
    b2[i] = gemm_ms(a0, bs[i]);
    if (i % 8 == 0) control("B pass 2");
  }
  for (int i = 0; i < count_b; ++i) {
    b3[i] = gemm_ms(a0, bs[i]);
    if (i % 8 == 7) control("B pass 3");
  }
  for (int i = 0; i < count_b; ++i)
    std::printf("{\"exp\":\"B\",\"i\":%d,\"pass1_ms\":%.4f,\"pass2_ms\":%.4f,\"pass3_ms\":%.4f,\"chase_ns\":%.1f}\n", i, b1[i],
                b2[i], b3[i], chase_b[i]);
  std::fflush(stdout);
  // A: activation buffers
  std::vector<bf16*> as(count_a);
  std::vector<double> a1(count_a), a2(count_a);
  for (int j = 0; j < count_a; ++j) {
    cuda_check(cudaMalloc(&as[j], a_n * 2));
    fill<<<int((a_n + 255) / 256), 256>>>(as[j], a_n, 1e-3f, unsigned(j));
    a1[j] = gemm_ms(as[j], b0);
    if (j % 8 == 7) control("A pass 1");
  }
  for (int j = count_a - 1; j >= 0; --j) {
    a2[j] = gemm_ms(as[j], b0);
    if (j % 8 == 0) control("A pass 2");
  }
  for (int j = 0; j < count_a; ++j)
    std::printf("{\"exp\":\"A\",\"i\":%d,\"pass1_ms\":%.4f,\"pass2_ms\":%.4f}\n", j, a1[j], a2[j]);
  control("end");
  std::printf("{\"buddyinfo_normal_after\":\"%s\"}\n", buddyinfo().c_str());
  for (auto* p : bs) cudaFree(p);
  for (auto* p : as) cudaFree(p);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "alloc-placement-probe: %s\n", e.what());
  return 1;
}
