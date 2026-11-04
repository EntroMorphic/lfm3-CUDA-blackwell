"""Fail-closed sanitizer-report tests using a fake tool, not GPU coverage."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/check_lfm3_sanitizer.py"
FAKE = '''#!/usr/bin/env python3
import os, pathlib, sys
a=sys.argv[1:]; log=pathlib.Path(a[a.index('--log-file')+1]); mode=os.environ['FAKE_MODE']
name=log.stem; code=0; text='ERROR SUMMARY: 0 errors'; stdout=''
if name=='initcheck-missing-control':
    text='Uninitialized __global__ memory read'; code=86
    if mode=='bad-control': code=0
if name=='initcheck-cublas-producer' and mode in ('blocked','bad-numerics'):
    text='Uninitialized __global__ memory read'; code=86
    stdout='PASS: 128 sampled outputs' if mode=='blocked' else 'incorrect projection output'
if name=='memcheck' and mode=='mem-error': code=86
if name=='initcheck-cublas-producer' and mode=='internal': text='Internal Sanitizer Error'
log.write_text(text); print(stdout); sys.exit(code)
'''


@pytest.mark.parametrize("mode,status,passed", [
    ("clean", "passed", True),
    ("blocked", "blocked-cuBLAS-output-tracking", False),
    ("bad-control", "failed-preflight-controls", False),
    ("bad-numerics", "failed-preflight-controls", False),
    ("internal", "failed-preflight-controls", False),
    ("mem-error", "passed", False),
])
def test_cannot_credit_incomplete_or_failed_checks(tmp_path, monkeypatch, mode, status, passed):
    sanitizer = tmp_path / "fake-sanitizer"
    sanitizer.write_text(FAKE)
    sanitizer.chmod(0o755)
    for name in ("probe", "binary", "request"):
        (tmp_path / name).write_text("fixture")
    monkeypatch.setenv("FAKE_MODE", mode)
    output = tmp_path / "evidence"
    command = [sys.executable, str(TOOL), "--sanitizer", str(sanitizer),
               "--probe", str(tmp_path / "probe"), "--binary", str(tmp_path / "binary"),
               "--checkpoint", str(tmp_path), "--request", str(tmp_path / "request"), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == (0 if passed else 86), result.stderr
    report = json.loads((output / "summary.json").read_text())
    assert report["passed"] is passed
    assert report["initcheck_status"] == status
    assert report["synccheck_nvjet_excluded"] is False
