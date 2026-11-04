# Experimental text CLI

`tools/lfm3_chat.py` uses the **trusted local official AutoProcessor** for chat
formatting/tokenization, invokes the custom CUDA executable for all inference,
then uses that processor's tokenizer to decode generated text. It never loads a
Transformers language model or calls its forward/generation methods. The official
processor does load its bundled audio codec; no audio input is supplied or
processed in this text path. This is a Python-fronted native
engine, **not a fully native tokenizer**; native chat-tokenizer parity remains
unestablished. Only single-turn text input and greedy native decoding are exposed.

From `liquid-CUDA`, in the already validated checkpoint/processor environment:

```bash
python3 tools/lfm3_chat.py \
  --checkpoint ../models/LFM3-Omni-2B \
  --executable build/cmake/liquid-lfm3 \
  --prompt 'Say hello in one short sentence.' \
  --max-new-tokens 32 --capacity 512 \
  --output ../tmp/chat-smoke-001
```

The **GPU smoke passed on Thor**, producing `Hello! How can I help you today?`
through native CUDA inference. Its artifacts are in project-root
`tmp/chat-smoke-001/`. The output path
must not exist. For interactive typing, omit `--prompt`, type UTF-8 text on stdin,
then EOF. The native interface is exactly:

```text
liquid-lfm3 --experimental MODEL_DIRECTORY request.json output.json
```

Artifacts are `request.json`, untouched native `output.json`, decoded
`response.json`, and `text.txt`. Stdout is decoded text. `response.json` explicitly
records official preprocessing and `native_tokenizer_parity: false`. Copy actual
GPU outputs back to the local project root `tmp/` when running remotely; never
present mocked host-test text as inference evidence.

## Bounds and failure policy

- Prompt plus system text: 65,536 UTF-8 bytes, nonblank user prompt. Stdin is read
  with a byte bound. No implicit input truncation.
- Capacity and generation count: positive integers up to 4,096; prefill plus
  `max_new_tokens - 1` must fit capacity (the runner does not forward the final
  generated token). IDs must be real integers, not bools, and in vocabulary.
- Omni placeholder/mixed/audio-start IDs are suppressed explicitly in requests;
  EOS and text-end stop explicitly. Placeholder inputs are rejected. Responses
  with suppressed IDs, tokens after stop, unexplained early termination, wrong
  echoed input, wrong backend, invalid IDs or invalid timings fail before decoding.
  A final stop token remains in raw evidence but is removed before text decoding.
- Existing files, directories and symlinks are never reused as the output
  directory. It is created exclusively with mode 0700. No overwrite switch exists.
  Failed runs retain their fresh request/raw output; they cannot produce decoded
  success artifacts. A retry must use another fresh directory.
- Native invocation has no shell, a finite timeout (default 300 seconds), checked
  exit status, and discarded stdout/stderr rather than unbounded captured logs.
  To diagnose native errors, rerun the recorded request directly with a **new**
  raw-output path. Executable and checkpoint code must be trusted: this is not a
  sandbox against a malicious executable or checkpoint Python.
- JSON inputs/responses are bounded to 1 MiB and must be regular nonsymlink files.
  Duplicate keys, nonfinite constants, excess nesting, malformed JSON and invalid
  response schema fail closed. Decoded text is also bounded to 1 MiB.

## Host validation and dependency caveat

```bash
python3 -m pytest tests/test_lfm3_chat.py -q
PYTHONPATH=tests python3 -m pytest tests audit/test_host_contracts.py -q
```

Observed on this host: **68 passed in 0.62s** for the launcher suite;
**209 passed in 18.93s** for the combined host suite (including concurrent
context-validation additions in the working tree).

Tests use a fake processor and an actual fake subprocess, with no weights or GPU.
They exercise the exact argv contract, text-end/EOS and length stopping, malformed
responses, numeric/byte/token/cache limits, modality leakage, executable failures,
timeouts, symlink rejection and output overwrite protection.

A CPU-only real-processor preflight on this host failed in installed Transformers'
dynamic-module loader: the isolated module cache lacked transitive
`modeling_lfm3.py` while importing the official Omni processor. This is a dependency
environment blocker, not a native inference result. Use the pinned, previously
validated processor environment for the GPU smoke. The CLI intentionally does not
silently replace the official processor with a different tokenizer/template or
modify the checkpoint/cache to hide this failure. `local_files_only=True` is set;
only run trusted local checkpoint Python, since `trust_remote_code=True` is needed.
