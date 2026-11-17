"""Host tests for the Thor tools of validation/image-cache-v1's red-team fixes C3-C5: tools/make_lfm3_c3_images.py,
tools/measure_lfm3_image_stall.py's --prime-images, and tools/check_lfm3_image_cache_fixes.py, driven against
tests/fake_lfm3_image_server.py (the server's image path without a model: where it decodes shows in /proc as on Thor).
Linux only (/proc)."""
import importlib
import json
import os
import random
import struct
import sys
import zlib
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
pytestmark = pytest.mark.skipif(not Path("/proc/self/task").exists(), reason="needs /proc")

import check_lfm3_image_cache_fixes as fixes  # noqa: E402
import fake_lfm3_image_server as fake  # noqa: E402
import make_lfm3_c3_images as mk  # noqa: E402
import measure_lfm3_image_stall as stall  # noqa: E402
from check_lfm3_image_server import as_client  # noqa: E402

MIB = 1 << 20


def noise_png(path, width, height, seed, noise=True):
    rng = random.Random(seed)
    if noise:
        Image.frombytes("RGB", (width, height), rng.randbytes(width * height * 3)).save(path, "PNG")
    else:  # compresses to almost nothing: its cost is its rows
        Image.new("RGB", (width, height), (rng.randrange(256), 90, 200)).save(path, "PNG")
    return path


def wrapper(path, env):
    """An executable server: the fake, with ENV set (the tools start servers by path, as on Thor)."""
    path.write_text("#!" + sys.executable + "\nimport os, runpy, sys\n"
                    + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
                    + f"sys.argv[0] = {str(ROOT / 'tests' / 'fake_lfm3_image_server.py')!r}\n"
                    + f"runpy.run_path(sys.argv[0], run_name='__main__')\n")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def world(tmp_path):
    """A stage-0-shaped round (req/, out/ with the fake's replies), a 4K stand-in "cherry" (256 x 144 noise), primes
    made from it by make_lfm3_c3_images, and one fresh image."""
    img = tmp_path / "images"
    img.mkdir()
    sizes = dict(cherry=(256, 144), small=(64, 64), wide=(512, 256), tall=(256, 512), china=(1024, 1024),
                 flower=(1024, 1024))
    paths = {n: noise_png(img / f"{n}.png", w, h, i, noise=n in ("cherry", "china", "flower"))
             for i, (n, (w, h)) in enumerate(sizes.items())}
    rd = tmp_path / "round"
    (rd / "req").mkdir(parents=True)
    (rd / "out").mkdir()
    def conv(images, text="Describe this image."):
        return [{"role": "system", "content": "Describe the provided image."},
                {"role": "user", "content": [{"type": "image", "image": str(p)} for p in images] + [{"type": "text", "text": text}]}]
    reqs = {n: conv([paths[n]]) for n in ("cherry", "small", "wide", "tall", "china")}
    reqs["two"] = conv([paths["china"], paths["flower"]], "What is different between these two images?")
    for n, messages in reqs.items():
        req = dict(messages=messages, mode="text", max_new_tokens=128, record_image=True)
        (rd / "req" / f"{n}.json").write_text(json.dumps(req))
        (rd / "out" / f"{n}.json").write_text(json.dumps(dict(text=fake.reply_text(as_client(req)))))
    mk.main(str(paths["cherry"]), str(tmp_path / "prime"), 1, 16)
    mk.main(str(paths["cherry"]), str(tmp_path / "fresh"), 17, 17)
    return dict(tmp=tmp_path, round=str(rd), prime=str(tmp_path / "prime"), fresh=str(tmp_path / "fresh" / "c3-17.png"),
                cherry=paths["cherry"])


# ---------------------------------------------------------------------------------------------------- images
def test_images_are_distinct_valid_pngs_changed_from_one_row_down(tmp_path):
    src = noise_png(tmp_path / "src.png", 320, 200, 7)
    base = Image.open(src).convert("RGB")
    seen = {Path(src).read_bytes()}
    for index in (1, 2, 3, 300):
        png, info = mk.make(Path(src).read_bytes(), index)
        assert png not in seen
        seen.add(png)
        out = tmp_path / f"{index}.png"
        out.write_bytes(png)
        got = Image.open(out).convert("RGB")
        assert got.size == base.size
        diff = [y for y in range(200) if any(got.getpixel((x, y)) != base.getpixel((x, y)) for x in range(0, 320))]
        assert diff and diff[0] == info["changed_row"] == (index * 71) % 200
    with pytest.raises(ValueError):
        mk.make(Path(src).read_bytes(), 0)


def test_token_counts_and_costs_follow_the_server():
    assert mk.tokens(3840, 2160) == 1980            # cherry: 7,920 patches in HF's dump, 4 per token
    assert mk.tokens(256, 256) == 64 and mk.tokens(512, 512) == 256 and mk.tokens(2048, 1024) == 2048
    assert mk.tokens(16, 16) == 64                  # scaled up to the minimum
    c = mk.costs(b"x" * 9833788, 3840, 2160)
    assert c["key_bytes"] == 13111720 and c["row_bytes"] == 1980 * 2048 * 2
    assert c["engine_cost"] == c["key_bytes"] + c["row_bytes"] and c["front_end_cost"] == c["key_bytes"] + 4


def test_a_png_with_ancillary_chunks_keeps_them(tmp_path):
    src = noise_png(tmp_path / "src.png", 64, 48, 3)
    data = Path(src).read_bytes()
    body = b"sRGB-like test chunk"
    extra = struct.pack(">I", len(body)) + b"tEXt" + body + struct.pack(">I", zlib.crc32(b"tEXt" + body) & 0xFFFFFFFF)
    data = data[:33] + extra + data[33:]  # after IHDR (8 + 25 bytes)
    png, _ = mk.make(data, 5)
    assert [k for k, _ in mk.chunks(png)][:2] == [b"IHDR", b"tEXt"]


# ---------------------------------------------------------------------------------------------------- stall tool
def test_stall_tool_arguments_without_the_option_are_unchanged():
    assert stall.parse_args(["S", "M", "R", "O"]) == (["S", "M", "R", "O"], None)
    assert stall.parse_args(["S", "M", "R", "O", "3"]) == (["S", "M", "R", "O", 3], None)
    assert stall.parse_args(["S", "M", "R", "O", "3", "--image-cache-mb", "0"]) == \
        (["S", "M", "R", "O", 3, "--image-cache-mb", "0"], None)
    assert stall.parse_args(["S", "M", "R", "O", "3", "--prime-images", "D", "--image-cache-mb", "0"]) == \
        (["S", "M", "R", "O", 3, "--image-cache-mb", "0"], "D")
    assert stall.parse_args(["S", "M", "R", "O", "3", "--image-cache-mb", "0", "--prime-images", "D"]) == \
        (["S", "M", "R", "O", 3, "--image-cache-mb", "0"], "D")


@pytest.mark.parametrize("primed", [False, True])
def test_stall_tool_runs_with_and_without_priming(world, monkeypatch, primed):
    monkeypatch.setattr(stall, "ROWS", 2)
    monkeypatch.setattr(stall, "START_IMAGE", 0.3)
    monkeypatch.setattr(stall, "WINDOW", 0.6)
    server = wrapper(world["tmp"] / "server", dict(FAKE_STREAM_SECONDS="1.5", FAKE_DEFAULT_CACHE_MB="1"))
    out = world["tmp"] / f"stall-{primed}.json"
    req = Path(world["round"]) / "req" / "cherry.json"
    if primed:
        stall.main(server, "model", str(req), str(out), 1, prime_dir=world["prime"])
    else:
        stall.main(server, "model", str(req), str(out), 1)
    r = json.loads(out.read_text())
    assert [x["kind"] for x in r["runs"]] == ["without", "with"]
    assert all(x["image_status"] in (None, 200) for x in r["runs"])
    if primed:
        assert r["prime_images"] == [f"c3-{i:02d}.png" for i in range(1, 17)]
        assert all(x["primed_images"] == 16 and x["prime_seconds"] > 0 for x in r["runs"])
    else:
        assert set(r) == {"runs", "median_worst_gap", "stall_seconds"}
        assert all(set(x) == {"worst_gap", "row_gaps", "image_status", "image_seconds", "started", "kind"} for x in r["runs"])


# ---------------------------------------------------------------------------------------------------- c3
def c3(world, monkeypatch, env, primes=4):
    monkeypatch.setattr(fixes, "DEFAULT_BUDGET", 1 * MIB)
    pdir = world["tmp"] / f"prime{primes}"
    pdir.mkdir(exist_ok=True)
    m = json.loads((Path(world["prime"]) / "manifest.json").read_text())
    names = sorted(m["images"])[:primes]
    for n in names:
        (pdir / n).write_bytes((Path(world["prime"]) / n).read_bytes())
    (pdir / "manifest.json").write_text(json.dumps(dict(m, images={n: m["images"][n] for n in names})))
    server = wrapper(world["tmp"] / "server-c3", dict(FAKE_DEFAULT_CACHE_MB="1", **env))
    out = world["tmp"] / "c3.json"
    code = fixes.c3(server, "model", world["round"], str(pdir), world["fresh"], str(out))
    return code, json.loads(out.read_text())


def test_c3_sees_the_handler_decode_the_evicted_image(world, monkeypatch):
    code, r = c3(world, monkeypatch, {})
    assert r["precondition"]["engine_evicts_cherry"] and r["precondition"]["front_end_keeps_all"]
    assert all(r["controls"].values()), r["controls"]
    assert r["decoded_on"] == "handler" and r["cherry_texts_equal_lfm3_speech"] and code == 0 and r["passed"]


def test_c3_sees_the_engine_decode_without_the_fix(world, monkeypatch):
    code, r = c3(world, monkeypatch, dict(FAKE_C3_UNFIXED="1"))
    assert r["decoded_on"] == "engine" and code == 1 and not r["passed"]


def test_c3_refuses_images_too_few_to_evict(world, monkeypatch):
    code, r = c3(world, monkeypatch, {}, primes=1)
    assert code == 1 and not r["passed"] and not r["precondition"]["engine_evicts_cherry"] and "rows" not in r


# ---------------------------------------------------------------------------------------------------- c4
@pytest.mark.parametrize("leak", [0, 5])
def test_c4_subtracts_the_caches_and_sees_memory_left_behind(world, monkeypatch, leak):
    monkeypatch.setattr(fixes, "DEFAULT_BUDGET", 16 * MIB)
    monkeypatch.setattr(fixes, "SETTLE", 1.0)
    monkeypatch.setattr(fixes, "LATER", 1.0)
    server = wrapper(world["tmp"] / f"server-c4-{leak}", dict(FAKE_DEFAULT_CACHE_MB="16", FAKE_LEAK_MB=str(leak)))
    out = world["tmp"] / "c4.json"
    fixes.c4(server, "model", world["round"], world["prime"], str(out), "test")
    r = json.loads(out.read_text())
    assert r["statuses"] == [200] * 16 and r["server_exit"] == 0
    m = json.loads((Path(world["prime"]) / "manifest.json").read_text())["images"]
    held = [m[n] for n in sorted(m)][:16]
    assert r["host_cache_growth"] == sum(2 * c["key_bytes"] for c in held[1:])
    assert r["device_cache_growth"] == sum(c["row_bytes"] for c in held[1:])
    if leak:
        assert r["rss_excess"] >= 15 * leak * MIB * 0.9 and not r["passed"]
    else:
        assert abs(r["rss_excess"]) < 32 * MIB, r["rss_excess"]  # the device stand-in is this busy host's: not asserted


# ---------------------------------------------------------------------------------------------------- c5
@pytest.mark.parametrize("fault_env, ok", [(dict(FAKE_FAULT_KEEP="1"), True), ({}, False)])
def test_c5_fault_server_replies_as_cache_off_and_keeps_nothing(world, fault_env, ok):
    server = wrapper(world["tmp"] / "server-c5", {})
    fault = wrapper(world["tmp"] / "fault-c5", fault_env)
    out = world["tmp"] / "c5.json"
    code = fixes.c5(server, fault, "model", world["round"], str(out))
    r = json.loads(out.read_text())
    assert r["checks"]["fault_equals_off"] and r["checks"]["fault_stays_up"] and r["checks"]["calibration_on_hits"]
    assert r["checks"]["fault_caches_nothing"] is ok and r["passed"] is ok and code == (0 if ok else 1)


# ---------------------------------------------------------------------------------------------------- memcheck
def fake_sanitizer(path, errors):
    path.write_text("#!" + sys.executable + "\nimport subprocess, sys\n"
                    "a = sys.argv[1:]\nlog = None\nwhile a[0].startswith('--'):\n"
                    "    if a[0] == '--log-file': log = a[1]\n    a = a[2:]\n"
                    "code = subprocess.call(a)\n"
                    f"open(log, 'w').write('========= ERROR SUMMARY: {errors} error' + ('s' if {errors} != 1 else '') + '\\n')\n"
                    f"sys.exit(99 if {errors} else code)\n")
    path.chmod(0o755)
    return str(path)


def test_memcheck_sequence_covers_hits_misses_evictions_and_oversized(world):
    bodies = {n: fixes.case(world["round"], n)[0] for n in ("small", "wide", "tall", "china", "two")}
    keys_of = {n: [(k, fixes.key_costs(k)) for k in fixes.image_keys(b)] for n, b in bodies.items()}
    sim = fixes.simulate(fixes.MEMCHECK_SEQUENCE, keys_of, fixes.MEMCHECK_BUDGET_MB * MIB)
    flat = [e for _, ev in sim for e in ev]
    assert sim[0] == ("small", ["miss"]) and sim[1] == ("small", ["hit"])
    assert any(e.startswith("miss, evicts") for e in flat) and "over" in flat


@pytest.mark.parametrize("errors", [0, 2])
def test_memcheck_runs_under_a_sanitizer_and_stops_the_server_itself(world, errors):
    server = wrapper(world["tmp"] / "server-mc", {})
    out = world["tmp"] / "mc"
    assert fixes.memcheck("-", server, "model", world["round"], str(out)) == 0
    san = fake_sanitizer(world["tmp"] / f"sanitizer-{errors}", errors)
    code = fixes.memcheck(san, server, "model", world["round"], str(out))
    r = json.loads((out / "memcheck.json").read_text())
    assert r["statuses_200"] and r["texts_equal_plain"] is True
    assert (code, r["passed"], r["exit"]) == ((0, True, 0) if not errors else (1, False, 99))


def test_speech_list_writes_requests_and_both_lists(world, tmp_path):
    fixes.speech_list(world["round"], str(tmp_path / "sp"), "cherry", "small", "small")
    reqs = sorted((tmp_path / "sp" / "req").glob("*.json"))
    assert [p.name for p in reqs] == ["00-cherry.json", "01-small.json", "02-small.json"]
    r = json.loads(reqs[0].read_text())
    assert r["max_new_tokens"] == 16 and "record_image" not in r
    lines = (tmp_path / "sp" / "list-memcheck.txt").read_text().splitlines()
    assert lines[1] == f"{tmp_path / 'sp' / 'req' / '01-small'}.json {tmp_path / 'sp' / 'memcheck' / '01-small'}"


def test_the_tools_import_cleanly():
    for m in ("check_lfm3_image_cache_fixes", "measure_lfm3_image_stall", "make_lfm3_c3_images"):
        importlib.import_module(m)
    assert os.access(ROOT / "tools" / "check_lfm3_image_cache_fixes.py", os.R_OK)
