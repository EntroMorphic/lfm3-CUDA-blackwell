"""Exact FP32 logit traces (G4): rounding, all-or-none presence, nonfinite rejection, and gate wiring."""
import importlib.util
from pathlib import Path
import random
import struct
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from lfm3_trace_exact import bf16_rounding_of, exact_logit_steps  # noqa: E402


def rne(word):
    return ((word + 0x7FFF + ((word >> 16) & 1)) >> 16) & 0xFFFF


def test_rounding_matches_round_to_nearest_even_on_finite_values():
    rng = random.Random(0)
    words = [rng.getrandbits(32) for _ in range(20000)]
    words += [0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x3F808000, 0x3F818000, 0x3F80C000,
              0x7F7FFFFF, 0xFF7FFFFF, 0x7F7F8000, 0x7F7F7FFF]  # zeros, subnormals, ties, near overflow
    finite = [w for w in words if (w >> 23) & 0xFF != 0xFF]
    got = struct.unpack(f"<{len(finite)}H", bf16_rounding_of(struct.pack(f"<{len(finite)}I", *finite)))
    assert list(got) == [rne(w) for w in finite]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_rounding_matches_torch():
    import torch
    rng = random.Random(1)
    finite = [w for w in (rng.getrandbits(32) for _ in range(20000)) if (w >> 23) & 0xFF != 0xFF]
    values = torch.tensor(finite, dtype=torch.int64).to(torch.int32).view(torch.float32)
    want = values.bfloat16().view(torch.int16).numpy().astype("<u2").tobytes()
    assert bf16_rounding_of(struct.pack(f"<{len(finite)}I", *finite)) == want


@pytest.mark.parametrize("word", [0x7F800000, 0xFF800000, 0x7FC00000, 0x7FFFFFFF, 0xFFFFFFFF, 0x7F800001])
def test_rounding_rejects_nonfinite(word):
    with pytest.raises(ValueError, match="nonfinite"):
        bf16_rounding_of(struct.pack("<If", word, 2.0))


def test_rounding_rejects_partial_words():
    with pytest.raises(ValueError, match="multiple of 4"):
        bf16_rounding_of(b"\0\0\0")


def write(directory, step, exact, rounded=None):
    raw = struct.pack(f"<{len(exact)}f", *exact)
    (directory / f"{step}.logits.f32").write_bytes(raw)
    (directory / f"{step}.logits.bf16").write_bytes(bf16_rounding_of(raw) if rounded is None else rounded)


def test_exact_steps_accepts_consistent_traces_and_absence(tmp_path):
    for step in range(3):
        write(tmp_path, step, [1.0, 16.0 - 0.03, -2.5])
    assert exact_logit_steps(tmp_path, range(3)) is True
    other = tmp_path / "bf16-only"
    other.mkdir()
    (other / "0.logits.bf16").write_bytes(b"\0\0")
    assert exact_logit_steps(other, [0]) is False


@pytest.mark.parametrize("damage", ["partial", "not-rounding", "nan", "short-bf16"])
def test_exact_steps_rejects_inconsistent_traces(tmp_path, damage):
    for step in range(2):
        write(tmp_path, step, [1.0, 16.0 - 0.03])
    if damage == "partial":
        (tmp_path / "1.logits.f32").unlink()
    if damage == "not-rounding":
        (tmp_path / "1.logits.bf16").write_bytes(struct.pack("<HH", 0x3F80, 0x417F))
    if damage == "nan":
        (tmp_path / "1.logits.f32").write_bytes(struct.pack("<ff", 1.0, float("nan")))
    if damage == "short-bf16":
        (tmp_path / "1.logits.bf16").write_bytes(b"\x80\x3f")
    with pytest.raises(ValueError):
        exact_logit_steps(tmp_path, range(2))


@pytest.mark.parametrize("tool", ["check_lfm3_context.py", "check_lfm3_forward.py"])
def test_gates_check_exact_logit_traces(tool):
    # Both gates that read native .bf16 logit traces must run the all-or-none rounding check.
    text = (TOOLS / tool).read_text()
    assert "from lfm3_trace_exact import exact_logit_steps" in text and "exact_logit_steps(" in text.split("import exact_logit_steps")[1]


def test_forward_tolerance_mode_requires_exact_logit_traces(tmp_path):
    spec = importlib.util.spec_from_file_location("forward_gate", TOOLS / "check_lfm3_forward.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    (tmp_path / "0.logits.bf16").write_bytes(struct.pack("<HH", 0x3F80, 0x4000))
    assert gate.check_exact_logits(tmp_path, False, "case") is False  # bitwise mode: BF16-only traces are fine
    with pytest.raises(AssertionError, match="requires the engine's exact"):
        gate.check_exact_logits(tmp_path, True, "case")
    (tmp_path / "0.logits.f32").write_bytes(struct.pack("<ff", 1.0, 2.0))
    assert gate.check_exact_logits(tmp_path, True, "case") is True
    (tmp_path / "0.logits.f32").write_bytes(struct.pack("<ff", 1.0, 2.5))
    with pytest.raises(ValueError, match="not the rounding"):
        gate.check_exact_logits(tmp_path, True, "case")
