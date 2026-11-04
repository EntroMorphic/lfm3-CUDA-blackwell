"""Local official processor -> experimental native CUDA -> decoded text.

No Transformers language model is loaded. The official processor may load its
bundled audio codec even for text-only use; it does not perform text inference.
Native chat-tokenizer parity is not claimed.
Only use trusted local checkpoint code (AutoProcessor requires remote-code trust).
"""
import argparse
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 65536
MAX_JSON_BYTES = 1024 * 1024
MODALITY_KEYS = (
    "audio_token_id", "audio_out_token_id", "image_token_id", "video_token_id",
    "mixed_start_token_id", "audio_start_token_id",
)


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return value


def token_ids(value, name, vocabulary, maximum):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{name}: expected bounded token list")
    return [integer(token, name, 0, vocabulary - 1) for token in value]


def read_json(path):
    # Reject response symlinks/FIFOs and oversized files before reading into RAM.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
            raise ValueError(f"{path}: expected regular JSON file <= {MAX_JSON_BYTES} bytes")
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("JSON file too large")
    # Do not rely on Python-version-dependent parser recursion limits. Scan the
    # UTF-8 text outside strings before asking the JSON decoder to allocate it.
    quoted = escaped = False
    depth = 0
    for char in raw.decode("utf-8"):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 64:
                raise ValueError("JSON nesting exceeds supported depth")
        elif char in "]}":
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds supported depth") from exc
    if not isinstance(result, dict):
        raise ValueError("expected JSON object")
    return result


def policy(config):
    omni = config.get("model_type") == "lfm3_omni"
    text = config.get("text_config") if omni else config
    if not isinstance(text, dict) or text.get("model_type") != "lfm3":
        raise ValueError("checkpoint must be LFM3 or LFM3-Omni")
    vocabulary = integer(text.get("vocab_size"), "vocab_size", 1, 2**31 - 1)
    eos = text.get("eos_token_id")
    stop = [] if eos is None else eos if isinstance(eos, list) else [eos]
    suppress = []
    if omni:
        suppress = [config.get(key) for key in MODALITY_KEYS]
        stop = stop + [config.get("text_end_token_id")]
    suppress = token_ids(suppress, "modality token", vocabulary, 6)
    stop = token_ids(stop, "stop token", vocabulary, vocabulary)
    if set(stop) & set(suppress) or len(set(suppress)) == vocabulary:
        raise ValueError("conflicting text generation policy")
    return vocabulary, sorted(set(suppress)), sorted(set(stop))


def load_processor(checkpoint):
    # Import lazily: host-only contract tests need neither Transformers nor torch.
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=True,
    )


def make_request(processor, prompt, system, config, capacity, count):
    integer(capacity, "capacity", 1, 4096)
    integer(count, "max_new_tokens", 1, 4096)
    if not prompt.strip() or len((prompt + system).encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError("prompt must be nonblank; combined prompt/system limit is 65536 UTF-8 bytes")
    vocabulary, suppress, stop = policy(config)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    encoded = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
    )
    ids = encoded.get("input_ids")
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    ids = token_ids(ids, "input_ids", vocabulary, capacity)
    if not ids or len(ids) + count - 1 > capacity:
        raise ValueError("prompt plus generation exceeds native cache capacity; no truncation performed")
    if set(ids) & set(suppress):
        raise ValueError("multimodal placeholder input is unsupported in text-only mode")
    return dict(input_ids=ids, max_new_tokens=count, capacity=capacity,
                suppress_tokens=suppress, stop_tokens=stop)


def validate_response(response, request, vocabulary):
    if response.get("backend") != "native-cuda-experimental":
        raise ValueError("unexpected native backend")
    echoed = token_ids(response.get("input_ids"), "response input_ids", vocabulary, request["capacity"])
    if echoed != request["input_ids"]:
        raise ValueError("native input_ids do not match request")
    generated = token_ids(response.get("generated_ids"), "generated_ids", vocabulary, request["max_new_tokens"])
    if not generated or set(generated) & set(request["suppress_tokens"]):
        raise ValueError("empty generation or forbidden modality token in native response")
    stops = set(request["stop_tokens"])
    if any(token in stops for token in generated[:-1]):
        raise ValueError("native response continues after stop token")
    if len(generated) < request["max_new_tokens"] and generated[-1] not in stops:
        raise ValueError("native response ended early without stop token")
    for key in ("prefill_seconds", "decode_seconds"):
        value = response.get(key)
        if (type(value) not in (int, float) or value < 0 or value > sys.float_info.max
                or not math.isfinite(value)):
            raise ValueError(f"invalid {key}")
    return generated[:-1] if generated[-1] in stops else generated


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run(args, processor=None):
    checkpoint = args.checkpoint.resolve(strict=True)
    executable = args.executable.resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("native executable is not an executable file")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 86400:
        raise ValueError("timeout must be finite and in (0, 86400] seconds")
    # Never reuse a directory, even if empty, or follow an output-directory symlink.
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output directory already exists; choose a fresh path")
    config = read_json(checkpoint / "config.json")
    integer(args.capacity, "capacity", 1, 4096)
    integer(args.max_new_tokens, "max_new_tokens", 1, 4096)
    prompt = args.prompt
    if prompt is None:
        raw = sys.stdin.buffer.read(MAX_TEXT_BYTES + 1)
        if len(raw) > MAX_TEXT_BYTES:
            raise ValueError("stdin prompt exceeds 65536 bytes")
        prompt = raw.decode("utf-8")
    if not prompt.strip() or len((prompt + args.system).encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError("blank or oversized prompt/system text")
    policy(config)
    if processor is None:
        processor = load_processor(checkpoint)
    request = make_request(processor, prompt, args.system, config, args.capacity, args.max_new_tokens)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    output = args.output.resolve()
    write_json(output / "request.json", request)
    # No shell and no unbounded captured subprocess logs. Failure artifacts remain
    # for diagnosis, but no decoded success report is written on any failure.
    subprocess.run([str(executable), "--experimental", str(checkpoint),
                    str(output / "request.json"), str(output / "output.json")],
                   check=True, timeout=args.timeout, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    response = read_json(output / "output.json")
    ids = validate_response(response, request, policy(config)[0])
    text = processor.tokenizer.decode(ids, skip_special_tokens=True)
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("invalid or oversized decoded text")
    write_json(output / "response.json", dict(
        backend="native-cuda-experimental", preprocessing="official-local-AutoProcessor",
        native_tokenizer_parity=False, generated_ids=response["generated_ids"], text=text,
        finish_reason="stop" if len(ids) < len(response["generated_ids"]) else "length",
    ))
    with (output / "text.txt").open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="trusted local checkpoint directory")
    parser.add_argument("--executable", type=Path, default=ROOT / "build/cmake/liquid-lfm3")
    parser.add_argument("--output", type=Path, required=True, help="new artifact directory (must not exist)")
    parser.add_argument("--prompt", help="user text; if omitted, read bounded UTF-8 stdin")
    parser.add_argument("--system", default="Respond with text.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        text = run(args)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        print(f"lfm3_chat: {exc}", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
