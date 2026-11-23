// lfm::gemm's tcgen05 path (validation/tcgen05-gemm-v1): V3 of validation/tcgen05-probe-v1 (2-CTA, tensor-memory
// accumulators, TMA loads; native/include/lfm/tcgen05_gemm.cuh) behind lfm::gemm::run. Bitwise identical to the
// mma.sync path on every case the probe ran (612 per variant), so taking it never changes a result.
// Compiled only for sm_110a (-gencode arch=compute_110a,code=sm_110a); every other unit stays sm_110. Built and
// declared only with LIQUID_LFM3_TCGEN05 (LFM_GEMM_TCGEN05).
//
// Dispatch: try_run takes a problem only inside the probe's contract (B stored N x K, batch 1, N % 256 == 0,
// K % 64 == 0, dense lda/ldb/ldc, 16-byte aligned) with M >= 257 (the probe's S3: faster than mma.sync on every
// shape from there). Otherwise it returns false and the caller runs its usual kernel.
// Fallback: per device, at first use: compute capability 11.0, the kernels' attributes readable, the shared-memory
// opt-in set, the tensor-map encoder found. Any failure turns the path off for that device (one line on stderr).
// LFM3_TCGEN05=0 turns it off for the process.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "lfm/gemm.cuh"
#include "lfm/tcgen05_gemm.cuh"

namespace lfm::gemm::tcgen05 {
namespace {

constexpr int kStages = 6;                                    // V3 as measured (probe b4)
constexpr int kMinRows = 257;                                 // probe S3
constexpr unsigned long long kTimeoutNs = 10000000000ull;     // bounded mbarrier waits: a trap, never a hang
constexpr size_t kSmem = kStages * tc05::kPairStageBytes + 1024;
constexpr int kMaxDevices = 64;

struct DeviceState {
  int state = 0;  // 0 not yet checked, 1 on, -1 off
  int sms = 0;
};
std::mutex g_mutex;
DeviceState g_devices[kMaxDevices];
PFN_cuTensorMapEncodeTiled_v12000 g_encode = nullptr;
std::atomic<bool> g_enabled{[] {
  const char* v = std::getenv("LFM3_TCGEN05");
  return !(v && std::strcmp(v, "0") == 0);
}()};
std::atomic<unsigned long long> g_launches{0};

bool off(int device, const char* why) {
  std::fprintf(stderr, "lfm::gemm: tcgen05 path off on device %d: %s\n", device, why);
  return false;
}
template <bool BF> bool configure() {
  cudaFuncAttributes a;
  if (cudaFuncGetAttributes(&a, tc05::gemm_2cta<kStages, BF>) != cudaSuccess) return false;
  return cudaFuncSetAttribute(tc05::gemm_2cta<kStages, BF>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem)) == cudaSuccess;
}
// Under g_mutex.
bool check_device(int device, DeviceState& d) {
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
    return off(device, "compute capability unreadable");
  if (major != 11 || minor != 0) return off(device, "not compute capability 11.0 (sm_110a code)");
  if (cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || d.sms < 2)
    return off(device, "SM count");
  if (!configure<true>() || !configure<false>()) { cudaGetLastError(); return off(device, "kernel image or shared-memory opt-in"); }
  if (!g_encode) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess || !fn) {
      cudaGetLastError();
      return off(device, "cuTensorMapEncodeTiled not found");
    }
    g_encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }
  return true;
}
const DeviceState* device_state() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices) return nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  DeviceState& d = g_devices[device];
  if (d.state == 0) d.state = check_device(device, d) ? 1 : -1;
  return d.state == 1 ? &d : nullptr;
}
bool encode(CUtensorMap& m, const void* p, int rows, int k, int box_rows) {
  cuuint64_t dims[2] = {cuuint64_t(k), cuuint64_t(rows)};
  cuuint64_t strides[1] = {cuuint64_t(k) * 2};
  cuuint32_t box[2] = {64, cuuint32_t(box_rows)}, es[2] = {1, 1};
  return g_encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), dims, strides, box, es,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
bool aligned(const void* q) { return reinterpret_cast<std::uintptr_t>(q) % 16 == 0; }

}  // namespace

bool eligible(const Problem& p) {
  // N >= 2 x kBN (validation/tcgen05-v2 amendment 2): with one 256-wide output tile per row block tcgen05 leaves most
  // SMs idle and lost to mma.sync (the audio downsampler's N = 256 multiplies, 0.73-0.82x); every N >= 512 won
  return p.b_k_contiguous && p.batch == 1 && p.group == 1 && p.m >= kMinRows && p.n >= 2 * tc05::kBN &&
         p.n % tc05::kBN == 0 &&
         p.k % tc05::kBK == 0 && p.k >= tc05::kBK && p.lda == p.k && p.ldb == p.k && p.ldc == p.n && aligned(p.a) &&
         aligned(p.b) && aligned(p.c_f32 ? static_cast<const void*>(p.c_f32) : static_cast<const void*>(p.c));
}

bool try_run(const Problem& p, cudaStream_t stream) {
  if (!g_enabled.load(std::memory_order_relaxed) || !eligible(p)) return false;
  const DeviceState* d = device_state();
  if (!d) return false;
  CUtensorMap ma, mb;
  if (!encode(ma, p.a, p.m, p.k, tc05::kBM) || !encode(mb, p.b, p.n, p.k, 128)) {
    static std::once_flag once;
    std::call_once(once, [] { std::fprintf(stderr, "lfm::gemm: tcgen05 tensor-map encoding failed; this call ran mma.sync\n"); });
    return false;
  }
  tc05::Args a;
  a.m = p.m; a.n = p.n; a.k = p.k; a.ldc = p.ldc;
  a.c = p.c_f32 ? static_cast<void*>(p.c_f32) : static_cast<void*>(p.c);
  a.tiles_m = (p.m + tc05::kPairBM - 1) / tc05::kPairBM;
  a.tiles_n = p.n / tc05::kBN;
  a.tiles = a.tiles_m * a.tiles_n;
  a.band = std::max(1, std::min(a.tiles_n, int((8u << 20) / (uint32_t(tc05::kBN) * p.k * 2))));
  a.wd.timeout_ns = kTimeoutNs;
  a.wd.trap_tag = nullptr;
  const int grid = 2 * std::min(a.tiles, d->sms / 2);
  if (p.c_f32) tc05::gemm_2cta<kStages, false><<<grid, 192, kSmem, stream>>>(ma, mb, a);
  else tc05::gemm_2cta<kStages, true><<<grid, 192, kSmem, stream>>>(ma, mb, a);
  g_launches.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void set_enabled(bool on) { g_enabled.store(on); }
unsigned long long launch_count() { return g_launches.load(); }

}  // namespace lfm::gemm::tcgen05
