#pragma once
// Pinned staging for large host-to-device uploads (validation/pinned-uploads-v1). On Thor, a host-to-device cudaMemcpy
// of 1 MiB or more from pageable memory leaves the GPU's memory reads at about half speed until the GPU next idles for
// 1 ms; the same copy from pinned memory does not (validation/alloc-effect-v1, stages 3 and 4). upload() copies through
// a 4 MiB pinned buffer, allocated on first use and kept for the owner's life, in chunks: std::memcpy into the buffer,
// then cudaMemcpy from it. A host-to-device cudaMemcpy from pinned memory returns when the copy is done, so the buffer
// is free for the next chunk. The bytes reaching the device are the source's.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace lfm {

class PinnedStaging {
 public:
  static constexpr size_t kBytes = size_t(4) << 20;
  PinnedStaging() = default;
  PinnedStaging(const PinnedStaging&) = delete;
  PinnedStaging& operator=(const PinnedStaging&) = delete;
  PinnedStaging(PinnedStaging&& o) noexcept : buffer_(o.buffer_) { o.buffer_ = nullptr; }
  PinnedStaging& operator=(PinnedStaging&& o) noexcept {
    if (this != &o) {
      if (buffer_) cudaFreeHost(buffer_);
      buffer_ = o.buffer_;
      o.buffer_ = nullptr;
    }
    return *this;
  }
  ~PinnedStaging() {
    if (buffer_) cudaFreeHost(buffer_);
  }
  // Copies `bytes` from host `src` to device `dst`; synchronous, as cudaMemcpy. Returns the first error.
  cudaError_t upload(void* dst, const void* src, size_t bytes) {
    if (!bytes) return cudaSuccess;
    if (!buffer_) {
      const cudaError_t e = cudaMallocHost(&buffer_, kBytes);
      if (e != cudaSuccess) {
        buffer_ = nullptr;
        return e;
      }
    }
    for (size_t done = 0; done < bytes;) {
      const size_t n = std::min(kBytes, bytes - done);
      std::memcpy(buffer_, static_cast<const std::byte*>(src) + done, n);
      const cudaError_t e = cudaMemcpy(static_cast<std::byte*>(dst) + done, buffer_, n, cudaMemcpyHostToDevice);
      if (e != cudaSuccess) return e;
      done += n;
    }
    return cudaSuccess;
  }

 private:
  void* buffer_ = nullptr;
};

}  // namespace lfm
