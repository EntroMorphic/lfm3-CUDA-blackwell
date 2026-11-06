// Teacher-forced perplexity scorer for the native LFM3 runner. Not a serving path.
//
// Reuses Runner from src/lfm3_main.cu unmodified (LFM3_RUNTIME_ONLY drops its main): each window
// is prefilled with Runner::forward, which leaves the final-normed hidden state of every row in
// Runner::normalized. The tied LM head is then applied to all rows in tiles with Runner::linear,
// and the per-position negative log-likelihood is computed on the device in FP32, exactly as
// `-log_softmax(logits.float())[target]` defines it over all 128000 logits.
//
// Usage: lfm3-score MODEL_DIR WINDOWS.i32 LENGTH OUT.f32
//   WINDOWS.i32  int32 little-endian token ids, W windows of LENGTH ids each (W = size / LENGTH / 4)
//   OUT.f32      float32 NLL of positions 1..LENGTH-1 of each window, W*(LENGTH-1) values
// Prints one JSON summary line (mean NLL, perplexity, timing) to stdout.
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"

#include <cmath>
#include <cstdio>

namespace {
constexpr int kThreads = 1024;

// One block per row. Deterministic: fixed per-thread strided order, then a fixed tree.
__global__ void row_nll(const bf16* logits, const int* targets, float* out, int rows, int vocab) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const bf16* x = logits + size_t(row) * vocab;
  __shared__ float red[kThreads];
  float m = -INFINITY;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) m = fmaxf(m, __bfloat162float(x[i]));
  red[threadIdx.x] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float mx = red[0];
  __syncthreads();
  float sum = 0.0f;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) sum += expf(__bfloat162float(x[i]) - mx);
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = mx + logf(red[0]) - __bfloat162float(x[targets[row]]);
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 5) throw std::runtime_error("usage: lfm3-score MODEL_DIR WINDOWS.i32 LENGTH OUT.f32");
  const std::string model = argv[1];
  const int length = std::stoi(argv[3]);
  if (length < 2 || length > 32768) throw std::runtime_error("LENGTH must be 2..32768");
  std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open windows file");
  const auto bytes = size_t(in.tellg());
  if (bytes == 0 || bytes % (size_t(length) * 4)) throw std::runtime_error("windows file size is not a multiple of LENGTH ids");
  const size_t windows = bytes / (size_t(length) * 4);
  std::vector<int> ids(bytes / 4);
  in.seekg(0);
  in.read(reinterpret_cast<char*>(ids.data()), bytes);
  if (!in) throw std::runtime_error("windows read failed");

  lfm::text::scratch_rows(length);
  auto cfg = lfm::ModelConfig::load(model + "/config.json", length);
  if (!cfg.is_lfm3() || !cfg.tied) throw std::runtime_error("scorer requires the tied LFM3 checkpoint");
  for (int id : ids)
    if (id < 0 || id >= cfg.shape.vocabulary) throw std::runtime_error("token outside vocabulary");
  Runner runner(model, cfg);

  const int vocab = cfg.shape.vocabulary, hidden = cfg.shape.hidden, tile = 256;
  Buffer<bf16> tile_logits(size_t(tile) * vocab);
  Buffer<int> targets(length);
  Buffer<float> nll(length);
  std::vector<float> all;
  all.reserve(windows * (length - 1));
  std::vector<float> host(length);
  double total = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (size_t w = 0; w < windows; ++w) {
    std::vector<int> window(ids.begin() + w * length, ids.begin() + (w + 1) * length);
    runner.reset();
    runner.forward(window);  // fills runner.normalized with the final-normed hidden state of every row
    // Row r predicts token r+1; the last row has no target.
    cuda_check(cudaMemcpy(targets.p, window.data() + 1, size_t(length - 1) * sizeof(int), cudaMemcpyHostToDevice));
    for (int r0 = 0; r0 < length - 1; r0 += tile) {
      const int rows = std::min(tile, length - 1 - r0);
      runner.linear("embed_tokens.weight", runner.normalized.p + size_t(r0) * hidden, tile_logits.p, rows, hidden, vocab);
      row_nll<<<rows, kThreads>>>(tile_logits.p, targets.p + r0, nll.p + r0, rows, vocab);
      cuda_check(cudaGetLastError());
    }
    cuda_check(cudaMemcpy(host.data(), nll.p, size_t(length - 1) * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < length - 1; ++i) {
      if (!std::isfinite(host[i])) throw std::runtime_error("nonfinite NLL");
      total += host[i];
      all.push_back(host[i]);
    }
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::ofstream out(argv[4], std::ios::binary);
  out.write(reinterpret_cast<const char*>(all.data()), all.size() * sizeof(float));
  if (!out) throw std::runtime_error("output write failed");
  const double mean = total / all.size();
  std::printf("{\"engine\":\"native\",\"length\":%d,\"windows\":%zu,\"scored_tokens\":%zu,\"mean_nll\":%.9f,\"perplexity\":%.6f,\"seconds\":%.3f}\n",
              length, windows, all.size(), mean, std::exp(mean), seconds);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
