"""Checkpoint integration gates. Run in the pinned PyTorch CUDA container on Thor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tools = Path(__file__).resolve().parent
    env = dict(os.environ)
    # Use precisely the reference container's cuBLAS/runtime, not another image's version.
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")

    def run(command, log):
        with log.open("w") as stream:
            result = subprocess.run([str(x) for x in command], env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}): see {log}")

    reports = []
    for name, extra, steps in [
        ("baseline", [], 16),
        ("arithmetic", ["--prompt", "What is 17 plus 25? Answer briefly."], 16),
        ("window257", ["--length", "257"], 4),
        ("single", ["--length", "1"], 4),
    ]:
        directory = args.output / name
        run([sys.executable, tools / "lfm3_reference.py", "--checkpoint", args.checkpoint,
             "--output", directory, "--tokens", steps, *extra], args.output / (name + ".reference.log"))
        run([args.binary, "--experimental", args.checkpoint, directory / "request.json", directory / "native.json"],
            directory / "native.log")
        run([sys.executable, tools / "compare_lfm3_traces.py", directory, "--require-exact"], directory / "comparison.log")
        request = json.loads((directory / "request.json").read_text())
        expected = json.loads((directory / "reference.json").read_text())["generated_ids"]
        request.pop("teacher_ids")
        request.pop("trace_directory")
        request["max_new_tokens"] = len(expected)
        # Fill the last legal slot on the final decode (no spare cache capacity).
        request["capacity"] = len(request["input_ids"]) + len(expected) - 1
        (directory / "generation-request.json").write_text(json.dumps(request))
        run([args.binary, "--experimental", args.checkpoint, directory / "generation-request.json", directory / "generation.json"],
            directory / "generation.log")
        generation = json.loads((directory / "generation.json").read_text())
        assert generation["generated_ids"] == expected, f"free-running token mismatch: {name}"
        assert generation["reset_exact_logits_ok"], name
        assert not set(expected) & set(request["suppress_tokens"])
        comparison = json.loads((directory / "comparison.json").read_text())
        reports.append(dict(case=name, activation_checks=comparison["checks"], tokens=len(expected),
                            max_relative_rms=max(x["relative_rms"] for x in comparison["results"]),
                            max_logit_abs=max(x["max_abs"] for x in comparison["results"] if ".logits." in x["name"]),
                            teacher_argmax_exact=True, greedy_exact=True, reset_exact_logits=True))
        print(name, "PASS", flush=True)
    # Deterministic stopping and suppression: use known first argmax as stop.
    directory = args.output / "baseline"
    request = json.loads((directory / "generation-request.json").read_text())
    first = json.loads((directory / "generation.json").read_text())["generated_ids"][0]
    request["stop_tokens"] = [first]
    (directory / "stop-request.json").write_text(json.dumps(request))
    run([args.binary, "--experimental", args.checkpoint, directory / "stop-request.json", directory / "stop.json"], directory / "stop.log")
    assert json.loads((directory / "stop.json").read_text())["generated_ids"] == [first]
    request["suppress_tokens"].append(first)
    request["max_new_tokens"] = 1
    (directory / "suppress-request.json").write_text(json.dumps(request))
    run([args.binary, "--experimental", args.checkpoint, directory / "suppress-request.json", directory / "suppress.json"], directory / "suppress.log")
    assert json.loads((directory / "suppress.json").read_text())["generated_ids"][0] != first

    # Request contract failures must happen before loading checkpoint weights.
    invalid = [dict(input_ids=[]), dict(input_ids=[-1]), dict(input_ids=[128000]),
               dict(input_ids=[1], capacity=0), dict(input_ids=[1], capacity=4097),
               dict(input_ids=[1], max_new_tokens=True), dict(input_ids=[1], capacity=1, teacher_ids=[2]),
               dict(input_ids=[1], max_new_tokens=1, teacher_ids=[2])]
    for i, req in enumerate(invalid):
        path = args.output / f"invalid-{i}.json"
        path.write_text(json.dumps(req))
        result = subprocess.run([str(args.binary), "--experimental", str(args.checkpoint), str(path), str(args.output / f"invalid-{i}-output.json")],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 1, (i, result.stderr)
        assert not (args.output / f"invalid-{i}-output.json").exists()
    report = dict(cases=reports, stop_check=True, suppression_check=True, invalid_requests_rejected=len(invalid),
                  final_cache_slot_checked=True,
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  scope="actual BF16 checkpoint activations/logits, teacher forcing and free-running greedy; eager attention only")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
