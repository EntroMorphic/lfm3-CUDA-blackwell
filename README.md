# LFM3 native CUDA development

## Full text-serving work in progress

The tokenizer-only CLI now supports explicit textual histories and seeded native
sampling. An opt-in tiled native path passes fixed 32K combined-context numerical
checks against Transformers SDPA; short-context exact checks remain separate.
The resident worker passes finite persistent-KV isolation and 32K budget gates.
The authenticated loopback HTTP service also passes finite native and admission
gates. Cooperative token-boundary cancellation and a finite host-resource soak
also pass. This is **not yet full-serving acceptance**: worker/service GPU batching,
long-soak/GPU-resource/performance and the complete sanitizer gate remain open. See
[`audit/TEXT_SERVING_ROADMAP.md`](audit/TEXT_SERVING_ROADMAP.md) for current evidence
and deliberately unclaimed features. Historical milestone details follow.

Independent source snapshot of the existing Liquid Blackwell inference engine.
The original repository is not modified. This copy includes native source, native
tests/tools, the MIT license, and the original CPU configuration tests; it does
not include weights, build artifacts, the Python CLI, or experimental evidence.

`SOURCE_SNAPSHOT.json` records source paths, hashes, revision, and the dirty
working-tree status at copy time. The snapshot includes the source's current
uncommitted changes; it is not a clean upstream release. Upstream correctness
limitations still apply.

## Current milestone

**Experimental text forward/generation is now implemented separately in
`liquid-lfm3`.** Four real-checkpoint cases pass 6,890 bitwise activation/logit
comparisons against Transformers eager, plus greedy generation and cache-reset
checks. A checked text launcher is available in `tools/lfm3_chat.py`. The expanded
context suite also passes bitwise logits through the 4096-token limit, including
sliding-window transitions, final-slot decode and exact reset. Native tokenizer
parity and full integrated initcheck acceptance remain outstanding; the latter
has a reproduced cuBLAS/tool tracking blocker, not a credited pass.
The current combined host suite has 327 passing tests. See
[audit/ACCEPTANCE.md](audit/ACCEPTANCE.md) for findings, fixes and evidence limits.
See [audit/FORWARD.md](audit/FORWARD.md) and `validation/forward-final/`.
The preserved `liquid-native` runtime still rejects LFM3; its guard is unchanged.

The earlier isolated-operator milestone established:

- CPU configuration parsing for standalone LFM3 and the LFM3-Omni text backbone.
- Per-layer operator kind, RoPE theta/partial rotation, sliding window, attention
  gates, sandwich norms, checkpoint namespace, and actual SwiGLU width.
- Host-only safetensors text inventory validation.
- Isolated native CUDA operators for partial/per-type RoPE, head gating, sandwich
  RMSNorm/residual, width-7 convolution, and full/sliding decode attention.
- Explicit rejection of native LFM3 execution before weight allocation. Metadata
  support is **not** a working LFM3 forward pass.

Run from this directory:

```bash
PYTHONPATH=tests python3 -m pytest tests audit/test_host_contracts.py -q
python3 tools/inspect_checkpoint.py ../models/LFM3-Omni-2B
```

The inspector strictly validates JSON, configuration, all serialized tensor types
and byte spans (including other modalities), plus expected text names/shapes/BF16
dtype. It does not hash payloads or establish numerical parity. No model copies are needed.

## Next milestones

1. Expand operator coverage to real checkpoint activations and longer contexts;
   retain separate tests for the reference/eager and optimized numerical policies.
2. Wire LFM3 tensor mappings and buffer allocation into the forward pass; preserve
   LFM2 behavior. Do not remove the execution guard until these paths are complete.
3. Compare teacher-forced states/logits against the pinned Transformers model.
   Include sequence lengths around 256, cache reset, prefill/decode, batch isolation,
   and the Omni wrapper's text-mode token suppression and stopping behavior.
4. Validate free-running generation, then benchmark matched batch-1 workloads.
5. Add multimodal components separately; no full native Omni support is claimed.

## Thor operator validation

The copied native executable and optional operator bridge both compile successfully
on Thor with CUDA 13.0.88. After red-team remediation, three seeds (42, 43, 2026)
pass 145 isolated GPU checks each. Separate suites cover 336 exact convolution
checks and 48 input/stream boundary checks. The host suite has 110 passing tests,
including the five originally failing review cases. Raw reports are in `validation/`.

These are bounded fixtures, not a promise of exact model parity. Attention remains
tolerance-tested against eager BF16 semantics; this does not establish parity with
SDPA or a complete model. Original pre-remediation results are retained and are
historical, not evidence for the current binary. See `audit/REMEDIATION.md` for scope
and sanitizer results.

Coverage includes global/partial RoPE positions through 32767; head gates; norms;
convolution decode/cache/reset and each path's own checkpoint reference;
per-row GQA positions around the 256-token boundary; and exact exclusion of future
tokens and out-of-window values. The attention prototype supports contiguous full
causal caches up to 4096 slots, not ring buffers or optimized prefill. It is an
operator bring-up kernel, not a production attention implementation.

### Numerical and interface contracts

- `conv7_prefill` accepts contiguous `[B,T,3,D]` projections and an empty logical
  cache; it overwrites the entire `[B,D,7]` state and uses the checkpoint's sequential
  prefill reduction. Padding, packed documents and incremental prefill are unsupported.
- `conv7` accepts `[B,3,D]` and uses the pinned Triton decode's 8-wide tree reduction.
  Under cancellation, checkpoint prefill and decode themselves can differ; we test
  each against its own reference instead of demanding a false cross-path equality.
- Python callers must use `tools/lfm3_bridge.py`, not private raw ctypes symbols.
  It enforces dtype, shape, contiguous layout, devices, writable non-aliasing,
  integer/scalar/launch bounds, and positions. The synchronous correctness harness
  establishes input-producer readiness and synchronizes the caller's stream.
  Concurrent mutations are forbidden and CUDA graph capture is explicitly rejected.
- The C ABI independently checks pointers, launch geometry, and position ranges.
  Raw pointers cannot encode tensor shapes/strides/allocation extents; the checked
  wrapper is mandatory. Neither interface is a production integration boundary.

Reproduce on Thor from the project copy:

```bash
bash build-ops-thor.sh
docker exec \
  -e TRITON_PTXAS_PATH=/workspace/tools/ptxas \
  -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas \
  lfm3-omni-dev python /workspace/liquid-CUDA/tools/check_lfm3_ops.py \
  --library /workspace/liquid-CUDA/build/liblfm3_ops.so \
  --checkpoint /workspace/models/LFM3-Omni-2B \
  --seed 42 --output /workspace/liquid-CUDA/build/operator-parity-seed-42.json
```

Native builds use `native/CMakeLists.txt` (CUDA 13, sm_110, C++20 and PCRE2).
`-DLIQUID_LFM3_OPS_TESTS=ON` adds the test bridge. The execution guard remains:
the model forward pass has not yet been integrated or validated.
