"""G6: token choice is bounded by the ids the tokenizer defines; both implementations verify, never assume."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from lfm3_tokenizer_info import tokenizer_token_count

CHECKPOINT = ROOT.parent / "models" / "LFM3-Omni-2B"


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path_factory.mktemp("tok") / "probe"
    subprocess.run(["g++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/tests/tokenizer_count_probe.cpp"), "-o", str(binary)], check=True)
    return binary


def write(tmp_path, vocab, added):
    d = tmp_path / "ckpt"
    d.mkdir(exist_ok=True)
    (d / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": vocab},
                                                  "added_tokens": [{"id": i, "content": f"<{i}>"} for i in added]}))
    return d


def both(probe, checkpoint, model_vocab):
    native = subprocess.run([str(probe), str(checkpoint / "tokenizer.json"), str(model_vocab)], capture_output=True, text=True)
    try:
        py = tokenizer_token_count(checkpoint, model_vocab)
    except ValueError as e:
        py = f"error: {e}"
    return native.stdout.strip(), py


def test_contiguous_ids_give_the_count(probe, tmp_path):
    ck = write(tmp_path, {"a": 0, "b": 1, "c": 2}, [3, 4])
    assert both(probe, ck, 8) == ("5", 5)


def test_added_token_reusing_a_base_id_is_the_same_id(probe, tmp_path):
    ck = write(tmp_path, {"a": 0, "b": 1, "c": 2}, [2, 3])
    assert both(probe, ck, 8) == ("4", 4)


def test_gap_is_rejected(probe, tmp_path):
    ck = write(tmp_path, {"a": 0, "b": 1, "c": 3}, [4])
    native, py = both(probe, ck, 8)
    assert "not contiguous" in native and "not contiguous" in str(py)


def test_id_beyond_model_vocabulary_is_rejected(probe, tmp_path):
    ck = write(tmp_path, {"a": 0, "b": 1}, [2, 9])
    native, py = both(probe, ck, 8)
    assert "outside the model vocabulary" in native and "outside the model vocabulary" in str(py)


def test_empty_vocabulary_is_rejected(probe, tmp_path):
    ck = write(tmp_path, {}, [0])
    native, py = both(probe, ck, 8)
    assert native.startswith("error") and str(py).startswith("error")


@pytest.mark.skipif(not (CHECKPOINT / "tokenizer.json").exists(), reason="checkpoint not present")
def test_real_checkpoint_has_125033_tokens_of_128000_rows(probe):
    assert both(probe, CHECKPOINT, 128000) == ("125033", 125033)
