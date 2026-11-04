"""Matched warmed batch-one manual-greedy measurement, after forward correctness gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--gates", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    args = p.parse_args()
    gate = json.loads(args.gates.read_text())
    assert gate["binary_sha256"] == hashlib.sha256(args.binary.read_bytes()).hexdigest(), "binary differs from accepted gates"
    assert all(x["greedy_exact"] and x["teacher_argmax_exact"] and x["reset_exact_logits"] for x in gate["cases"])
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    tool = Path(__file__).with_name("lfm3_reference.py")
    with (args.output / "reference.log").open("w") as stream:
        subprocess.run([sys.executable, str(tool), "--checkpoint", str(args.checkpoint),
                        "--output", str(args.output / "oracle"), "--tokens", str(args.tokens),
                        "--attention", args.attention, "--no-trace", "--benchmark"],
                       env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    reference = json.loads((args.output / "oracle/reference.json").read_text())
    request = json.loads((args.output / "oracle/request.json").read_text())
    request.pop("teacher_ids")
    request.pop("trace_directory")
    request.update(max_new_tokens=args.tokens, warmup_runs=1, check_reset=False)
    (args.output / "request.json").write_text(json.dumps(request, indent=2))
    samples = []
    for i in range(3):
        with (args.output / f"native-{i}.log").open("w") as stream:
            subprocess.run([str(args.binary), "--experimental", str(args.checkpoint),
                            str(args.output / "request.json"), str(args.output / f"native-{i}.json")],
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        samples.append(json.loads((args.output / f"native-{i}.json").read_text()))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), trust_remote_code=True)
    (args.output / "native-response.txt").write_text(tokenizer.decode(samples[0]["generated_ids"], skip_special_tokens=True) + "\n")
    (args.output / "reference-response.txt").write_text(reference["text"] + "\n")
    expected = reference["generated_ids"]
    exact = all(x["generated_ids"] == expected for x in samples)
    ref_samples = reference["benchmark"]["samples"]
    exact &= all(x["generated_ids"] == expected for x in ref_samples)
    def stats(rows):
        return dict(prefill_seconds_median=statistics.median(x["prefill_seconds"] for x in rows),
                    decode_tok_s_median=statistics.median(x["decode_forward_calls"] / x["decode_seconds"] for x in rows),
                    generated_tokens=len(rows[0]["generated_ids"]))
    mismatch = next((i for i, (a, b) in enumerate(zip(samples[0]["generated_ids"], expected)) if a != b), None)
    report = dict(greedy_exact=bool(exact), first_mismatch=mismatch, input_tokens=len(request["input_ids"]),
                  max_new_tokens=args.tokens, reference_attention=args.attention, native_attention="eager-bf16",
                  native=stats(samples), transformers=stats(ref_samples), samples=3, warmup_runs_per_sample=1,
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  timing_scope="synchronized batch-1 manual greedy; prefill separate; decode forward calls/sec; no traces; excludes loading and text decoding",
                  limitations="native uses contiguous cache and host logits/argmax; reference dynamic cache; no CUDA graphs; sequential, not simultaneous GPU workloads")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not exact:
        raise SystemExit("Greedy sequence mismatch; timings retained but not an accepted parity benchmark")


if __name__ == "__main__":
    main()
