"""Host-only launcher contracts: fake processor and executable, no GPU/model."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools/lfm3_chat.py"
SPEC = importlib.util.spec_from_file_location("lfm3_chat", TOOL)
chat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat)


class Processor:
    def __init__(self, ids=None):
        self.ids = [1, 2] if ids is None else ids
        self.tokenizer = self
        self.decoded = None

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "system", "content": "Respond with text."},
                            {"role": "user", "content": "Hello"}]
        assert kwargs == dict(add_generation_prompt=True, tokenize=True, return_dict=True)
        return {"input_ids": [self.ids]}

    def decode(self, ids, **kwargs):
        assert kwargs == dict(skip_special_tokens=True)
        self.decoded = ids
        return "Hello native!"


@pytest.fixture
def config():
    result = dict(model_type="lfm3_omni", text_config=dict(model_type="lfm3", vocab_size=100, eos_token_id=9),
                  text_end_token_id=10)
    result.update(zip(chat.MODALITY_KEYS, range(11, 17)))
    return result


@pytest.fixture
def args(tmp_path, config):
    checkpoint = tmp_path / "model with spaces"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(config))
    executable = tmp_path / "native executable"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "assert len(sys.argv) == 5 and sys.argv[1] == '--experimental'\n"
        "request = json.loads(pathlib.Path(sys.argv[3]).read_text())\n"
        "result = dict(backend='native-cuda-experimental', input_ids=request['input_ids'],\n"
        "              generated_ids=[3, 10], prefill_seconds=0.1, decode_seconds=0.2)\n"
        "pathlib.Path(sys.argv[4]).write_text(json.dumps(result))\n"
    )
    executable.chmod(0o700)
    return argparse.Namespace(checkpoint=checkpoint, executable=executable, output=tmp_path / "new output",
                              prompt="Hello", system="Respond with text.", max_new_tokens=4,
                              capacity=8, timeout=10)


def test_native_subprocess_and_decoding(args):
    processor = Processor()
    assert chat.run(args, processor) == "Hello native!"
    assert processor.decoded == [3]  # Text-end is validated then removed, not displayed.
    request = json.loads((args.output / "request.json").read_text())
    assert request == dict(input_ids=[1, 2], capacity=8, max_new_tokens=4,
                           suppress_tokens=list(range(11, 17)), stop_tokens=[9, 10])
    result = json.loads((args.output / "response.json").read_text())
    assert result["native_tokenizer_parity"] is False
    assert result["finish_reason"] == "stop"
    assert result["generated_ids"] == [3, 10]
    assert (args.output / "text.txt").read_text() == "Hello native!\n"


@pytest.mark.parametrize("ids", [[], [True], [-1], [100], [11], [1] * 9, [1] * 6])
def test_bad_or_overflowing_prompt_tokens(args, ids):
    with pytest.raises(ValueError):
        chat.run(args, Processor(ids))
    assert not args.output.exists()


@pytest.mark.parametrize("field,value", [
    ("capacity", 0), ("capacity", 4097), ("capacity", True),
    ("max_new_tokens", 0), ("max_new_tokens", -1), ("max_new_tokens", 4097),
    ("timeout", float("nan")), ("timeout", float("inf")), ("timeout", 0),
    ("prompt", " "), ("prompt", "é" * 32769), ("system", "x" * 65536),
])
def test_preflight_limits(args, field, value):
    setattr(args, field, value)
    with pytest.raises(ValueError):
        chat.run(args, Processor())
    assert not args.output.exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling"])
def test_existing_output_never_overwritten(args, kind):
    if kind == "directory":
        args.output.mkdir()
        (args.output / "keep").write_text("user work")
    elif kind == "file":
        args.output.write_text("user work")
    else:
        args.output.symlink_to(args.checkpoint if kind == "symlink" else args.output.parent / "missing")
    with pytest.raises(ValueError, match="already exists"):
        chat.run(args, Processor())
    if kind == "directory":
        assert (args.output / "keep").read_text() == "user work"


@pytest.mark.parametrize("mode", ["missing", "nonexecutable", "exit", "timeout", "no_output"])
def test_executable_failures(args, mode):
    if mode == "missing":
        args.executable.unlink()
    elif mode == "nonexecutable":
        args.executable.chmod(0o600)
    else:
        body = {"exit": "raise SystemExit(7)", "timeout": "import time; time.sleep(5)",
                "no_output": "pass"}[mode]
        args.executable.write_text(f"#!{sys.executable}\n{body}\n")
        if mode == "timeout":
            args.timeout = 0.05
    with pytest.raises((OSError, ValueError, subprocess.SubprocessError)):
        chat.run(args, Processor())
    assert not (args.output / "response.json").exists()
    assert not (args.output / "text.txt").exists()


def response(ids):
    return dict(backend="native-cuda-experimental", input_ids=[1, 2], generated_ids=ids,
                prefill_seconds=0.1, decode_seconds=0.0)


@pytest.mark.parametrize("ids", [[], [True], [-1], [100], [11, 10], [9, 3], [10, 10], [3], [3] * 5])
def test_reject_bad_generated_tokens(config, ids):
    request = chat.make_request(Processor(), "Hello", "Respond with text.", config, 8, 4)
    with pytest.raises(ValueError):
        chat.validate_response(response(ids), request, 100)


@pytest.mark.parametrize("field,value", [
    ("backend", "Transformers"), ("input_ids", [2]), ("input_ids", [True, 2]),
    ("prefill_seconds", -1), ("decode_seconds", float("nan")),
    ("decode_seconds", float("inf")), ("decode_seconds", True), ("decode_seconds", 10**1000),
])
def test_reject_bad_response_metadata(config, field, value):
    request = chat.make_request(Processor(), "Hello", "Respond with text.", config, 8, 4)
    value_response = response([3, 10])
    value_response[field] = value
    with pytest.raises(ValueError):
        chat.validate_response(value_response, request, 100)


@pytest.mark.parametrize("ids,expected", [([9], []), ([10], []), ([3, 3, 3, 3], [3] * 4)])
def test_valid_stop_and_length(config, ids, expected):
    request = chat.make_request(Processor(), "Hello", "Respond with text.", config, 8, 4)
    assert chat.validate_response(response(ids), request, 100) == expected


@pytest.mark.parametrize("raw", [b"[]", b"{", b'{"a":1,"a":2}', b'{"a":NaN}',
                                  b'{"a":Infinity}', b" " * (chat.MAX_JSON_BYTES + 1), b"\xff"])
def test_malformed_json(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises((ValueError, UnicodeError)):
        chat.read_json(path)


def test_response_symlink_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_text("{}")
    path = tmp_path / "output.json"
    path.symlink_to(target)
    with pytest.raises(OSError):
        chat.read_json(path)


def test_nested_json_has_controlled_error(tmp_path):
    path = tmp_path / "nested.json"
    path.write_text('{"a":' + '[' * 2000 + '0' + ']' * 2000 + '}')
    with pytest.raises(ValueError, match="nesting"):
        chat.read_json(path)


@pytest.mark.parametrize("payload", ["{}", "[]", "not json", '{"generated_ids":[11]}'])
def test_malformed_executable_response_has_no_success_artifact(args, payload):
    args.executable.write_text(
        f"#!{sys.executable}\nimport pathlib, sys\n"
        f"pathlib.Path(sys.argv[4]).write_text({payload!r})\n"
    )
    with pytest.raises(ValueError):
        chat.run(args, Processor())
    assert not (args.output / "response.json").exists()
    assert not (args.output / "text.txt").exists()


def test_stdin_limit_checked_before_processor(args, monkeypatch):
    import io

    args.prompt = None
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"x" * (chat.MAX_TEXT_BYTES + 1))))
    with pytest.raises(ValueError, match="stdin"):
        chat.run(args, Processor())
    assert not args.output.exists()


def test_standalone_lfm3_policy():
    assert chat.policy(dict(model_type="lfm3", vocab_size=100, eos_token_id=[9, 10])) == (100, [], [9, 10])


def test_exact_cache_boundary(config):
    request = chat.make_request(Processor([1] * 5), "Hello", "Respond with text.", config, 8, 4)
    assert len(request["input_ids"]) + request["max_new_tokens"] - 1 == 8


def test_cli_help_has_no_transformers_dependency():
    result = subprocess.run([sys.executable, str(TOOL), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Native chat-tokenizer parity is not claimed" in result.stdout


def test_cli_error_has_no_traceback(args):
    result = subprocess.run([sys.executable, str(TOOL), "--checkpoint", str(args.checkpoint),
                             "--output", str(args.output), "--executable", "/nonexistent"],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "lfm3_chat:" in result.stderr and "Traceback" not in result.stderr


def test_complete_cli_uses_only_local_auto_processor(args, tmp_path):
    # Fake module deliberately has no model classes: any model loading fails.
    module_dir = tmp_path / "python modules"
    module_dir.mkdir()
    (module_dir / "transformers.py").write_text(
        "class AutoProcessor:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, checkpoint, **kwargs):\n"
        "        assert kwargs == dict(local_files_only=True, trust_remote_code=True)\n"
        "        obj = cls(); obj.tokenizer = obj; return obj\n"
        "    def apply_chat_template(self, messages, **kwargs):\n"
        "        assert messages[-1]['content'] == 'Hello'\n"
        "        return dict(input_ids=[[1, 2]])\n"
        "    def decode(self, ids, **kwargs):\n"
        "        assert ids == [3]\n"
        "        return 'Mock CLI text (not inference evidence)'\n"
    )
    result = subprocess.run([
        sys.executable, str(TOOL), "--checkpoint", str(args.checkpoint),
        "--executable", str(args.executable), "--output", str(args.output),
        "--max-new-tokens", "4", "--capacity", "8",
    ], input="Hello", capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(module_dir)))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "Mock CLI text (not inference evidence)\n"
    assert (args.output / "response.json").is_file()
