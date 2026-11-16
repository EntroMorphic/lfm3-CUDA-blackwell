// lfm::gemm for the tcgen05 probe, compiled apart with the engine's own flag (-arch=sm_110) so the comparison runs
// exactly the engine's kernels, not an sm_110a recompilation of them.
#include "lfm/gemm.cuh"

void engine_gemm(const lfm::gemm::Problem& p, cudaStream_t stream) { lfm::gemm::run(p, stream); }
