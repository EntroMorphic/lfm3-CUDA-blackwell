// pinned-staging-check: validation/pinned-uploads-v1. lfm::PinnedStaging::upload against the bytes: for sizes around the
// 4 MiB chunk (0, 1, 3, 4 MiB - 1, 4 MiB, 4 MiB + 1, 8 MiB + 3, 13 MiB + 5) and unaligned source and destination
// offsets, the device holds exactly the source's bytes and nothing outside the target range changes (guard bytes).
// One JSON line per case and a summary.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lfm/pinned_staging.hpp"

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::printf("{\"fatal\":\"%s\",\"error\":\"%s\"}\n", what, cudaGetErrorString(e));
    std::exit(3);
  }
}

int main() {
  const size_t MiB = size_t(1) << 20;
  const size_t sizes[] = {0, 1, 3, 4 * MiB - 1, 4 * MiB, 4 * MiB + 1, 8 * MiB + 3, 13 * MiB + 5};
  const size_t offsets[] = {0, 1, 7};
  const size_t guard = 4096, cap = 13 * MiB + 5 + 2 * guard + 16;
  std::vector<unsigned char> src(cap), back(cap);
  for (size_t i = 0; i < cap; ++i) src[i] = static_cast<unsigned char>((i * 2654435761u) >> 13);
  unsigned char* dev = nullptr;
  check(cudaMalloc(&dev, cap), "malloc");
  lfm::PinnedStaging staging;
  int cases = 0, failed = 0;
  for (size_t n : sizes)
    for (size_t so : offsets)
      for (size_t dof : offsets) {
        check(cudaMemset(dev, 0xA5, cap), "memset");
        check(staging.upload(dev + guard + dof, src.data() + so, n), "upload");
        check(cudaMemcpy(back.data(), dev, cap, cudaMemcpyDeviceToHost), "back");
        bool ok = std::memcmp(back.data() + guard + dof, src.data() + so, n) == 0;
        for (size_t i = 0; i < guard + dof && ok; ++i) ok = back[i] == 0xA5;
        for (size_t i = guard + dof + n; i < cap && ok; ++i) ok = back[i] == 0xA5;
        ++cases;
        failed += !ok;
        std::printf("{\"bytes\":%zu,\"src_offset\":%zu,\"dst_offset\":%zu,\"pass\":%s}\n", n, so, dof, ok ? "true" : "false");
      }
  std::printf("{\"summary\":\"pinned-staging-check\",\"cases\":%d,\"failed\":%d}\n", cases, failed);
  cudaFree(dev);
  return failed ? 1 : 0;
}
