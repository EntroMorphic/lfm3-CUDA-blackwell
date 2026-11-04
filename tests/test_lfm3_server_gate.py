"""Counterexamples for the HTTP/native smoke assertions, without GPU/listener."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


@pytest.mark.parametrize("damage", [None, "control", "A-reuse", "B-recall", "reset", "invalidation", "binary-change"])
def test_http_gate_rejects_broken_acceptance(tmp_path, monkeypatch, damage):
    tool = Path(__file__).resolve().parents[1] / "tools/check_lfm3_server.py"
    spec = importlib.util.spec_from_file_location("http_gate", tool)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    (tmp_path / "config.json").write_text("{}")
    binary = tmp_path / "worker"
    binary.write_text("original")
    reference = tmp_path / "control.json"
    reference.write_text(json.dumps(dict(generated_ids=[1, 2])))
    output = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", [str(tool), "--checkpoint", str(tmp_path), "--worker", str(binary),
                                     "--control-result", str(reference), "--output", str(output)])
    workers = []
    class Worker:
        def __init__(self, *args, **kwargs):
            self.process = SimpleNamespace(pid=12345, wait=lambda timeout: 0)
            self.closed = False
            workers.append(self)
        def close(self):
            self.closed = True
    class Service:
        def __init__(self, *args):
            self.sessions = {"orphan": {}} if damage == "invalidation" else {}
        def close(self):
            workers[0].close()
    class Server:
        def __init__(self, *args):
            self.server_address = ("127.0.0.1", 1)
        def serve_forever(self): pass
        def shutdown(self): pass
        def server_close(self): pass
    monkeypatch.setattr(gate, "WorkerClient", Worker)
    monkeypatch.setattr(gate, "Service", Service)
    monkeypatch.setattr(gate, "LocalHTTPServer", Server)
    monkeypatch.setattr(gate, "load_frontend", lambda path: object())
    monkeypatch.setattr(gate.os, "killpg", lambda *args: None)
    # Each expected status and reply is fed through the harness's real checks.
    sequence = [
        (200, dict(ready=True)), (401, {}),
        (200, dict(session_id="a" * 32)), (200, dict(session_id="b" * 32)), (429, {}),
        (200, dict(generated_ids=[9] if damage == "control" else [1, 2], text="control", gpu_batching=False)),
        (400, {}), (400, {}), (400, {}),
        (200, dict(text="remembered")),
        (200, dict(reused_tokens=0 if damage == "A-reuse" else 1)),
        (200, dict(text="wrong" if damage == "B-recall" else "cobalt", reused_tokens=1)),
        (200, {}),
        (200, dict(generated_ids=[3] if damage == "reset" else [1, 2], reused_tokens=0)),
        (200, {}), (404, {}), (503, {}), (503, dict(ready=False)),
    ]
    class HTTP:
        def __init__(self, *args, **kwargs): pass
        def request(self, *args, **kwargs): pass
        def getresponse(self):
            status, body = sequence.pop(0)
            if not sequence and damage == "binary-change":
                binary.write_text("changed")
            return SimpleNamespace(status=status, read=lambda limit: json.dumps(body).encode())
        def close(self): pass
    monkeypatch.setattr(gate.http.client, "HTTPConnection", HTTP)
    if damage:
        with pytest.raises(AssertionError):
            gate.main()
        assert json.loads((output / "summary.json").read_text())["status"] == "failed"
    else:
        gate.main()
        assert not sequence
        assert json.loads((output / "summary.json").read_text())["status"] == "passed"
    assert workers[0].closed
