#pragma once

// validation/ttft-rest-v2 stage 0: timestamps from the first byte to the first text event. Only a build with
// -DLFM3_TTFT_TRACE has them: there LFM3_TTFT_MARK(name) appends "name CLOCK_MONOTONIC-nanoseconds\n" to the file named
// by the environment variable LFM3_TTFT_TRACE (nothing when it is unset). Elsewhere it compiles to nothing.
#ifdef LFM3_TTFT_TRACE
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>

namespace lfm::ttft_trace {
inline void mark(const char* name) {
  static std::mutex lock;
  static std::FILE* file = [] {
    const char* path = std::getenv("LFM3_TTFT_TRACE");
    return path ? std::fopen(path, "a") : nullptr;
  }();
  if (!file) return;
  timespec t{};
  ::clock_gettime(CLOCK_MONOTONIC, &t);
  std::lock_guard g(lock);
  std::fprintf(file, "%s %lld\n", name, static_cast<long long>(t.tv_sec) * 1000000000LL + t.tv_nsec);
  std::fflush(file);
}
}  // namespace lfm::ttft_trace

#define LFM3_TTFT_MARK(name) ::lfm::ttft_trace::mark(name)
#else
#define LFM3_TTFT_MARK(name) ((void)0)
#endif
