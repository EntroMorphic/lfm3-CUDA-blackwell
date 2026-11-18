#!/usr/bin/env python3
"""validation/always-on-v1 rule 2, the text part: each of the soak's three text bodies (check_lfm3_audio_soak.TEXTS) as
the first request after startup, on a fresh server with --warmup and on a fresh server without it; the /generate
replies must be identical (text, generated_ids, finish_reason: everything but timings). Six fresh servers.
Exits 0 when all three are identical, 1 otherwise.
Usage: check_lfm3_warm_text.py SERVER MODEL_DIR OUT_JSON [--server-arg ARG]...
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_audio_soak import TEXTS  # noqa: E402
from measure_lfm3_first_request import Fresh, text_request  # noqa: E402

TIMING_KEYS = {"seconds", "prefill_seconds", "generate_seconds", "queue_seconds"}


def reply_of(binary, model, scratch, extra, body):
    f = Fresh(binary, model, scratch, extra)
    try:
        status, _, reply = text_request(f.s, body)
    finally:
        clean = f.stop()
    return status, {k: v for k, v in reply.items() if k not in TIMING_KEYS}, clean


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("server")
    ap.add_argument("model")
    ap.add_argument("out_json")
    ap.add_argument("--server-arg", action="append", default=[])
    a = ap.parse_args(argv)
    rows, ok = [], True
    with tempfile.TemporaryDirectory() as scratch:
        for i, body in enumerate(TEXTS):
            cold = reply_of(a.server, a.model, scratch, a.server_arg, body)
            warm = reply_of(a.server, a.model, scratch, [*a.server_arg, "--warmup"], body)
            same = cold[0] == warm[0] == 200 and cold[1] == warm[1] and cold[2] and warm[2]
            rows.append(dict(body=i, cold_status=cold[0], warm_status=warm[0], identical=same,
                             cold=cold[1], warm=warm[1], exits_clean=[cold[2], warm[2]]))
            print(f"text {i}: {'identical' if same else 'DIFFER'} ({cold[0]}, {warm[0]})", flush=True)
            ok &= same
    Path(a.out_json).write_text(json.dumps(dict(rows=rows, passed=ok), indent=1))
    print("WARM TEXT PASS" if ok else "WARM TEXT FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
