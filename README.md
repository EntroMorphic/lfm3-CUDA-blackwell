# liquid-CUDA

A native C++/CUDA inference engine for [Liquid AI's LFM3-Omni-2B](https://huggingface.co/LiquidAI), built to run
every modality of the model in a single binary on an NVIDIA Jetson AGX Thor (Blackwell, `sm_110`) — at the same or
better quality than the Hugging Face reference implementation.

No Python, PyTorch or cuBLAS at run time: the tokenizer, chat template, image and audio front ends, encoders, language
model, depth decoder and audio codec are all native, served behind a loopback HTTP server, `liquid-lfm3-server`.

> **Status: research software.** Image understanding is new (November 2026); its listening and reading review is
> still open. Everything below is measured on one machine (a Jetson AGX Thor) against
> pinned Hugging Face references; see [Validation](#how-it-is-validated). It is not a general-purpose serving stack.

## What works

| Modality | In | Out | Evidence |
|---|---|---|---|
| Text | ✅ | ✅ | Greedy tokens identical to Transformers eager on the fixtures; per-token NLL 12.8–14.4% closer to an FP32 model than HF SDPA's at every length tested, 256 to 32K tokens ([g4-gates-v2](validation/g4-gates-v2/)); native tokenizer and chat template bit-exact ([g2-text-v1](validation/g2-text-v1/)) |
| Speech | ✅ | ✅ | ASR WER 2.59% vs HF's 2.61% on LibriSpeech test-clean ([audio-in-v1](validation/audio-in-v1/)); text-to-speech matches HF by ear and beats it on long-form text ([speech-e2e-v1](validation/speech-e2e-v1/), [speech-long-form-v1](validation/speech-long-form-v1/)) |
| Images | ✅ | — | Decoding and preprocessing bit-identical to HF on 1,982 images; vision encoder more accurate than HF BF16 on all 31 test images; POPE accuracy 89.06 vs HF 89.11; on MMStar, answers closer to the FP32 model than HF BF16's (94.8% vs 93.9% agreement) ([images-v1](validation/images-v1/)) |
| Video | — | — | Not started |

Serving: concurrent speech requests batched on the GPU, each byte-identical to its reply alone
([speech-batch-v1](validation/speech-batch-v1/)); streamed speech ([speech-stream-v1](validation/speech-stream-v1/));
text served alongside speech without waiting for it ([text-wait-v1](validation/text-wait-v1/)).

Text decode throughput on Thor (decode loop, greedy, [batch-decode-v1](validation/batch-decode-v1/)):

| Batch | 1 | 8 | 32 | 64 |
|---|---:|---:|---:|---:|
| tokens/s | 62 | 487 | 1,754 | 3,224 |

## Requirements

- NVIDIA Jetson AGX Thor (`sm_110`), JetPack 7 / CUDA 13. Other Blackwell parts may work but are untested.
- Docker, for the pinned build and reference containers ([docs/building.md](docs/building.md)).
- The [LFM3-Omni-2B](https://huggingface.co/LiquidAI) checkpoint (safetensors). Weights are not included in this
  repository and are subject to Liquid AI's license.

## Quick start

Build (on Thor, in the pinned compiler image — see [docs/building.md](docs/building.md) for the image and flags):

```bash
cmake -S native -B build/native -DLIQUID_LFM3_NATIVE_GEMM=ON
cmake --build build/native --target liquid-lfm3-server -j4
```

Run the server (loopback only; a bearer token is created on first start at `~/.config/liquid-lfm3/token`):

```bash
build/native/liquid-lfm3-server --experimental /path/to/LFM3-Omni-2B --port 8080
```

Ask for speech:

```bash
TOKEN=$(cat ~/.config/liquid-lfm3/token)
curl -s localhost:8080/speech -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Tell me about the moon."}],"mode":"interleaved","max_new_tokens":512}' \
  | python3 -c 'import sys,json,base64; r=json.load(sys.stdin); print(r["text"]); open("reply.wav","wb").write(base64.b64decode(r["audio_wav_base64"]))'
```

The full API — sessions, text generation, speech in and out, streaming — is in [docs/server-api.md](docs/server-api.md).

## How it is validated

Every capability is added as a *round* with its own directory under [`validation/`](validation/), and every round
follows the same discipline ([docs/validation.md](docs/validation.md)):

1. **Rules before data.** Pass/fail rules are written and committed before the first native run.
2. **Exactness where it is possible, measured bounds where it is not.** Integer and fixed-point stages must be
   bit-identical to the reference; floating-point stages must be at least as close to an FP32 reference as HF's own
   BF16 GPU path is.
3. **Coherency is a gate.** Real generations are read (and listened to) end to end after every engine change.
4. **Nothing else changes.** Each round re-runs the earlier rounds' outputs and requires them byte-identical.
5. **Red-team, remediate, verify** — then commit. Failures are recorded, not erased.

## Repository layout

| Path | Contents |
|---|---|
| [`native/`](native/) | The engine: headers (`include/lfm/`), executables (`src/`), native tests and probes, vendored third-party code |
| [`tools/`](tools/) | Python and shell tooling: HF reference runners, comparison and gate scripts, Thor job scripts ([catalog](tools/README.md)) |
| [`tests/`](tests/) | Host-only test suite (no GPU needed) |
| [`validation/`](validation/) | One directory per validation round: rules, results and compressed records |
| [`audit/`](audit/) | Engineering notes from the text-serving bring-up |
| [`docs/`](docs/) | Architecture, building, server API, validation method |

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Notable changes are listed in
[CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE). Vendored and ported third-party code keeps its own license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are not part of this repository.
