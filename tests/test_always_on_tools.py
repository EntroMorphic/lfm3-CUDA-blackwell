"""Host tests for validation/always-on-v1's deployment pieces and Thor tool: deploy/install-user-service.sh and the unit
template deploy/liquid-lfm3.service (filled in, installed under a temporary XDG_CONFIG_HOME with a stand-in systemctl,
checked by `systemd-analyze --user verify` where it exists), and tools/measure_lfm3_first_request.py against
tests/fake_lfm3_first_request_server.py (a server whose first request of each kind is slow unless --warmup ran)."""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
INSTALL = ROOT / "deploy" / "install-user-service.sh"

import measure_lfm3_first_request as tool  # noqa: E402


# ------------------------------------------------------------------ the unit and its installer

@pytest.fixture
def world(tmp_path):
    server = tmp_path / "bin" / "liquid-lfm3-server"
    server.parent.mkdir()
    server.write_text("#!/bin/sh\nexit 0\n")
    server.chmod(0o755)
    model = tmp_path / "models" / "LFM3-Omni-2B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    calls = tmp_path / "systemctl.log"
    fake = tmp_path / "bin" / "systemctl"
    fake.write_text(f"#!/bin/sh\necho \"$@\" >> {calls}\n")
    fake.chmod(0o755)
    sudo = tmp_path / "bin" / "sudo"  # on PATH first: the installer must never call it
    sudo.write_text(f"#!/bin/sh\necho sudo \"$@\" >> {calls}\nexit 1\n")
    sudo.chmod(0o755)
    env = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path / "config"), SYSTEMCTL=str(fake),
               PATH=f"{fake.parent}:{os.environ['PATH']}", USER="tester")
    return dict(server=server, model=model, calls=calls, env=env, unit=tmp_path / "config/systemd/user/liquid-lfm3.service")


def install(world, *extra, check=True):
    return subprocess.run(["bash", str(INSTALL), "--server", str(world["server"]), "--model", str(world["model"]), *extra],
                          env=world["env"], capture_output=True, text=True, check=check)


def directives(text):
    out = {}
    for line in text.splitlines():
        if line and not line.startswith(("#", "[")):
            key, _, value = line.partition("=")
            out.setdefault(key, []).append(value)
    return out


def test_install_fills_the_template_reloads_and_enables(world):
    r = install(world, "--port", "8123", "--library-path", "/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia",
                "--speech-rows", "8")
    unit = world["unit"].read_text()
    d = directives(unit)
    assert not re.search(r"@[A-Z_]+@", unit)  # every template value filled
    assert d["Type"] == ["notify"] and d["NotifyAccess"] == ["main"]
    assert d["ExecStart"] == [f'"{world["server"]}" --experimental "{world["model"]}" --port 8123 --warmup --speech-rows 8']
    assert d["Environment"] == ['"LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia"']
    assert d["Restart"] == ["on-failure"] and d["KillSignal"] == ["SIGTERM"] and d["TimeoutStopSec"] == ["30"]
    assert d["SendSIGKILL"] == ["no"] and d["RestartForceExitStatus"] == ["0"]
    assert d["WantedBy"] == ["default.target"]
    # SIGKILL appears only where the unit forbids it
    assert all("SIGKILL" not in line or line.startswith(("#", "SendSIGKILL=no")) for line in unit.splitlines())
    assert world["calls"].read_text().splitlines() == ["--user daemon-reload", "--user enable liquid-lfm3.service"]
    assert "sudo loginctl enable-linger tester" in r.stdout and "does not use sudo" in r.stdout


def test_no_library_path_means_no_environment_line(world):
    install(world)
    d = directives(world["unit"].read_text())
    assert "Environment" not in d and d["ExecStart"][0].endswith("--port 8080 --warmup")


def test_dry_run_installs_nothing(world):
    r = install(world, "--dry-run")
    assert "ExecStart=" in r.stdout and not world["unit"].exists() and not world["calls"].exists()


@pytest.mark.parametrize("extra,why", [
    (["--port", "0"], "--port"), (["--port", "x"], "--port"), (["--speech-rows", "8;rm"], "whole number"),
    (["--library-path", "/a%b"], "cannot carry"), (["--library-path", "/a|b"], "cannot carry"), (["--bogus"], "unknown"),
])
def test_bad_arguments_are_refused(world, extra, why):
    r = install(world, *extra, check=False)
    assert r.returncode == 2 and why in r.stderr and not world["unit"].exists(), r.stderr


def test_bad_paths_are_refused(world, tmp_path):
    for server, model, why in ((tmp_path / "missing", world["model"], "not an executable"),
                               (world["server"], tmp_path, "no config.json"),
                               ("relative/server", world["model"], "absolute")):
        r = subprocess.run(["bash", str(INSTALL), "--server", str(server), "--model", str(model)], env=world["env"],
                           capture_output=True, text=True)
        assert r.returncode == 2 and why in r.stderr, r.stderr


@pytest.mark.skipif(not shutil.which("systemd-analyze"), reason="systemd-analyze required")
def test_systemd_analyze_verifies_the_installed_unit(world):
    install(world, "--library-path", "/opt/cuda/lib")
    env = dict(os.environ)
    runtime = Path(f"/run/user/{os.getuid()}")
    if runtime.is_dir():
        env["XDG_RUNTIME_DIR"] = str(runtime)  # the user manager's own (verify loads the user's other units too)
    elif "XDG_RUNTIME_DIR" not in env:
        pytest.skip("no user runtime directory for systemd-analyze --user")
    r = subprocess.run(["systemd-analyze", "--user", "verify", str(world["unit"])], env=env, capture_output=True, text=True)
    mine = [line for line in (r.stdout + r.stderr).splitlines() if "liquid-lfm3" in line]
    assert r.returncode == 0 and not mine, r.stdout + r.stderr


# ------------------------------------------------------------------ tools/measure_lfm3_first_request.py

def wrapper(path, env):
    """An executable server: the fake, with ENV set (the tool starts servers by path, as on Thor)."""
    path.write_text("#!" + sys.executable + "\nimport os, runpy, sys\n"
                    + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
                    + f"sys.argv[0] = {str(ROOT / 'tests' / 'fake_lfm3_first_request_server.py')!r}\n"
                    + "runpy.run_path(sys.argv[0], run_name='__main__')\n")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def build(tmp_path):
    b = tmp_path / "build"
    (b / "vimg-e2e-a1/req").mkdir(parents=True)
    (b / "audio-in-listen").mkdir()
    png = tmp_path / "cherry.png"
    png.write_bytes(tool.warmup_png()[:64])  # the fake reads no pixels; the tool sends the bytes as they are
    (b / "vimg-e2e-a1/req/cherry.json").write_text(json.dumps(dict(
        messages=[{"role": "user", "content": [{"type": "image", "image": str(png)}, {"type": "text", "text": "?"}]}],
        mode="text", max_new_tokens=128, record_image="x")))
    (b / "audio-in-listen/q0-16k.wav").write_bytes(tool.warmup_wav())
    return b


def test_the_tool_measures_cold_and_warm_first_requests(tmp_path, build):
    server = wrapper(tmp_path / "server", {"FAKE_COLD": "0.3"})
    out = tmp_path / "out.json"
    assert tool.main([server, str(tmp_path), str(build), str(out), "--servers", "2", "--settle", "0.1"]) == 0
    r = json.loads(out.read_text())
    assert r["passed_requests"] is True
    for part in ("a", "b"):
        for kind in tool.KINDS:
            row = r["rule0"]["per_kind"][kind][part]
            assert 0.25 < row["first_minus_third"] < 0.6, (part, kind, row)
    assert r["rule0"]["proceed"] is False  # 0.3 s is under rule 0's 1 s
    assert len(r["c"]) == 2 and all(s["exit"] == 0 and s["warmup"]["warmup_seconds"] >= 1.2 for s in r["c"])
    assert r["rule1"]["passed"] is True and set(r["rule1"]["per_kind"]) == set(tool.KINDS)
    assert r["rule4_startup_seconds"]["median"] >= 1.2
    assert r["rule3_rss"]["cold_after_four"] > 0 and r["rule3_rss"]["warm"] > 0 and "passed" in r["rule3_rss"]
    # the server's own timings are carried for the /speech kinds
    assert all("prefill_seconds" in x for k in ("tts", "image", "audio") for x in r["a"]["requests"][k])


def test_rule0_proceeds_when_a_first_request_is_over_a_second_slower(tmp_path, build):
    server = wrapper(tmp_path / "server", {"FAKE_COLD": "1.1"})
    out = tmp_path / "out.json"
    assert tool.main([server, str(tmp_path), str(build), str(out), "--parts", "a"]) == 0
    r = json.loads(out.read_text())
    assert r["rule0"]["proceed"] is True and "c" not in r


def test_the_warmup_replica_is_the_servers_requests():
    bodies = tool.warmup_bodies()
    assert list(bodies) == list(tool.KINDS)
    image = base64.b64decode(bodies["image"]["messages"][1]["content"][0]["image"])
    assert image.startswith(b"\x89PNG") and int.from_bytes(image[16:20], "big") == 3840
    # the pixels and the clip against the C++ generator: tests/test_lfm3_http_warmup.py (needs the harness)
