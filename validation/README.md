# Validation records

Each directory is one validation round: its `README.md` states the rules (written before the run), the results, and
what failed. Records are compressed (`.gz`) where large; raw numerical traces are not committed. Paths here are cited by
commit messages and by other records, so they do not move. The method is described in
[docs/validation.md](../docs/validation.md).

## Text: correctness

| Round | Title |
|---|---|
| [g2-text-v1](g2-text-v1/) | G1–G3: native LFM3 text frontend, bit-exact against the reference |
| [text-native-v1](text-native-v1/) | Native text frontend wired into the engine binaries |
| [gemm-probe-v1](gemm-probe-v1/) | G4 stage 1: native BF16 GEMM against cuBLAS, every production shape |
| [g4-native-gemm-v1](g4-native-gemm-v1/) | G4 stage A, attempt 1: the engine on lfm::gemm. The §9 bar FAILS at 257 and 32,768 |
| [g4-native-gemm-v2](g4-native-gemm-v2/) | G4 stage A, attempt 2 (FP32 scores): the §9 bar FAILS at 257. G4 stage A is parked |
| [g4-native-gemm-v3](g4-native-gemm-v3/) | G4 stage A, attempt 3 (FP32 logits): the §9 bar PASSES at all four lengths |
| [g4-native-gates-v1](g4-native-gates-v1/) | G4 stage A: the runtime gates on the native-GEMM build, and what the failing ones measure |
| [g4-gates-v2](g4-gates-v2/) | G4 stage A done: the default engine runs without cuBLAS and passes every gate |
| [g4-prefill-heads-v1](g4-prefill-heads-v1/) | G4 prefill: query heads per attention call — measured (step 1 of journal/g4-prefill-memory_synth.md) |
| [g4-prefill-budget-v1](g4-prefill-budget-v1/) | G4 prefill: heads per call by a 256 MiB budget — verified; and a speed claim withdrawn |

## Text: speed

| Round | Title |
|---|---|
| [batch-decode-v1](batch-decode-v1/) | Batched decode for the LFM3 text engine (P0 speed): all gates PASS |
| [tree-verify-v1](tree-verify-v1/) | Tree-verify decode (single-request speed, step 2): exact, and up to 4.1x on one request |
| [lossless-v1](lossless-v1/) | Lossless BF16 weights (single-request speed, decision 5): exact, not yet faster |
| [drafting-v1](drafting-v1/) | Drafts for exact speculation (speed detour, parked): low-bit self-drafts and the trained head |

## Serving

| Round | Title |
|---|---|
| [native-server-v1](native-server-v1/) | G7: one executable for LFM3 text serving |
| [native-server-v2](native-server-v2/) | G7 red-team remediation: liquid-lfm3-server v2 |
| [text-wait-v1](text-wait-v1/) | Text no longer waits for speech to drain |

## Speech out

| Round | Title |
|---|---|
| [tts-trace-v1](tts-trace-v1/) | 2 step 1: measured trace of the HF text → text + audio path, and its ASR baseline |
| [codec-v1](codec-v1/) | 2 step 2: native codec decoder (A2): PASS |
| [depth-v1](depth-v1/) | 2 step 3, part 1: native depth decoder (A1, B1): PASS |
| [audio-control-v1](audio-control-v1/) | 2 step 3, part 2: the sampler (B2) and the mode machine (A4): PASS |
| [speech-forced-v1](speech-forced-v1/) | 2 step 4, the deterministic gate: the native speech loop with forced inputs (A3 and the LM in speech context) |
| [speech-e2e-v1](speech-e2e-v1/) | 2 step 4: native text in -> text and speech out, against HF (C1, C2, C4) |
| [speech-long-form-v1](speech-long-form-v1/) | 2 step 4: long-form TTS, the failure's mechanism and its fix |
| [speech-split-v1](speech-split-v1/) | 2 step 4: choosing the splitter's length rule by test (2026-11-09) |
| [speech-audio2b-v1](speech-audio2b-v1/) | Is LFM3-Audio-2B a better reader of long text than Omni? (2026-11-09) No. |
| [speech-normalize-v1](speech-normalize-v1/) | Numbers and abbreviations: rules, the model, or both, chosen by test |
| [multilingual-speech-v1](multilingual-speech-v1/) | Speech in every locale: native against HF beyond American English |

## Speech serving

| Round | Title |
|---|---|
| [speech-server-v1](speech-server-v1/) | 2: speech in liquid-lfm3-server (POST /speech) |
| [speech-stream-v1](speech-stream-v1/) | Streaming speech out of POST /speech |
| [speech-batch-v1](speech-batch-v1/) | Concurrent speech: many /speech requests at once, batched on the GPU |
| [speech-speed-v1](speech-speed-v1/) | Faster concurrent speech, every output bit-identical |
| [speech-speed-v2](speech-speed-v2/) | Faster concurrent speech, round 2, every output bit-identical |

## Audio in

| Round | Title |
|---|---|
| [audio-in-v1](audio-in-v1/) | 3 audio in: speech input on LFM3-Omni-2B, natively |
| [audio-assurance-v1](audio-assurance-v1/) | Audio in and out: assurance before images |

## Images in

| Round | Title |
|---|---|
| [images-v1](images-v1/) | 4 images in: image understanding on LFM3-Omni-2B, natively |

## Records without a README (2026-11-06)

Described by their commits and by [audit/](../audit/).

| Record | Contents |
|---|---|
| [g2-classes-v1](g2-classes-v1/) | pre-tokenizer character classes (G2) |
| [g6-v1](g6-v1/) | output ids without a token (G6) |
| [g6-v2](g6-v2/) | G6 red-team |
| [hf-text-nsys-v1](hf-text-nsys-v1/) | HF text path, Nsight Systems profile |
| [hf-text-trace-v2](hf-text-trace-v2/) | HF text path trace |
| [ppl-v1](ppl-v1/) | perplexity harness against HF SDPA |
| [prefill-v9-probe-v1](prefill-v9-probe-v1/) | prefill probe |
| [profile-32k-v1](profile-32k-v1/) | 32K profile |
| [thin-hybrid-v1](thin-hybrid-v1/) | thin-hybrid projection runtime candidate |
| [thin-hybrid-v3](thin-hybrid-v3/) | thin-hybrid projection runtime candidate |
| [thin-hybrid-v4](thin-hybrid-v4/) | thin-hybrid projection runtime candidate |
| [thin-probe-v1](thin-probe-v1/) | thin projection probe |

## Text-serving bring-up (2026-11-04 and earlier)

388 records from the text engine's first acceptance — forward-pass parity, 32K context, sanitizers, batching,
cancellation, soak and the cuBLAS policy studies — summarized in [audit/ACCEPTANCE.md](../audit/ACCEPTANCE.md) and the
other notes in [audit/](../audit/). By prefix:

`remediated*` (128), `wmma*` (76), `host*` (36), `projection*` (16), `http*` (15), `redteam*` (14), `decode*` (12), `forward*` (11), `bedrock*` (7), `context32k*` (6), `tma*` (6), `boundary*` (5), `racecheck32k*` (5), `worker*` (5), `context*` (4), `cublas*` (4), `grouped*` (4), `lt*` (4), `blas*` (3), `cancellation*` (3), `operator*` (3), `sampling*` (3), `acceptance*` (2), `benchmark*` (2), `chat*` (2), `native*` (2), `sanitizer32k*` (2), `32k*` (1), `frontend*` (1), `initcheck*` (1), `remediation*` (1), `resident*` (1), `sanitizer*` (1), `serving*` (1), `thor*` (1)
