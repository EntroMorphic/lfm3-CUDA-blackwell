"""lfm::Json::parse made faster (2026-11-17: a 13 MB image request was parsed twice at ~50 ms each): every value and
error identical to the frozen reference parser, native/tests/json_parse_check.cpp, plain and under ASan+UBSan."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("flags", [["-O2"], ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]])
def test_json_parse_matches_reference(tmp_path, flags):
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path / "json_parse_check"
    subprocess.run(["g++", "-std=c++20", *flags, "-I", str(ROOT / "native" / "include"),
                    str(ROOT / "native" / "tests" / "json_parse_check.cpp"), "-o", str(binary)], check=True)
    r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.mark.parametrize("variant", ["plain", "asan", "tsan"])
def test_strict_json_matches_reference_v2(tmp_path, variant):
    """validation/ttft-rest-v2 stage 1 rule 1: strict_json and Json::parse (a view parsed in place, plain runs scanned 8
    bytes at a time, the result moved) against main e41e36c's, frozen in json_reference_v2.hpp, on a million documents:
    the same tree or the same refusal; parse_messages with copy_media false against true on 100,000 message lists."""
    compiler, flags = {"plain": ("g++", ["-O2"]),
                       "asan": ("g++", ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]),
                       "tsan": ("clang++", ["-O1", "-g", "-fsanitize=thread"])}[variant]
    if shutil.which(compiler) is None:
        pytest.skip(f"{compiler} required")
    binary = tmp_path / "strict_json_check"
    subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror", *flags, "-I", str(ROOT / "native" / "include"),
                    str(ROOT / "native" / "tests" / "strict_json_check.cpp"), "-o", str(binary)], check=True)
    r = subprocess.run([str(binary), "1000000", "100000"], capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = json.loads(r.stdout)
    assert d["ok"] and d["documents"] == 1_000_000 and d["differ"] == 0, d
    assert d["accepted"] > 1000 and d["refused"] > 1000 and d["messages_accepted"] > 1000 and d["messages_refused"] > 1000, d
