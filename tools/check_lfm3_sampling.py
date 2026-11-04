"""Native sampling integration gates; not a PyTorch RNG parity claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--request",type=Path,required=True,help="Accepted tokenized text request")
    p.add_argument("--oracle",type=Path,required=True,help="Standalone text-sampling-oracle executable")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    base=json.loads(args.request.read_text())
    for key in ["teacher_ids","trace_directory","warmup_runs"]:base.pop(key,None)
    base.update(max_new_tokens=32,capacity=4096,check_reset=True)
    env=dict(os.environ)
    env["LD_LIBRARY_PATH"]=str(Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13/lib")+":"+env.get("LD_LIBRARY_PATH","")
    digest=hashlib.sha256(args.binary.read_bytes()).hexdigest()
    results={}
    def run(name,changes):
        request=dict(base,**changes)
        if name=="sample42":
            request.update(trace_logits_only=True,trace_directory=str(args.output/"sample42-trace"))
        path=args.output/(name+".request.json")
        path.write_text(json.dumps(request,indent=2)+"\n")
        destination=args.output/(name+".json")
        with (args.output/(name+".log")).open("w") as stream:
            subprocess.run([str(args.binary),"--experimental",str(args.checkpoint),str(path),str(destination)],
                           env=env,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        result=json.loads(destination.read_text())
        if result.get("reset_exact_logits_ok") is not True or not result.get("generated_ids"):
            raise ValueError(name+": missing reset/generation evidence")
        generated=result["generated_ids"]
        if set(generated)&set(request["suppress_tokens"]):raise ValueError("suppressed token sampled")
        if any(x in request["stop_tokens"] for x in generated[:-1]):raise ValueError("continued after stop")
        if len(generated)<request["max_new_tokens"] and generated[-1] not in request["stop_tokens"]:
            raise ValueError("unexplained termination")
        results[name]=result
        if name=="sample42":
            with (args.output/"sampling-oracle.log").open("w") as stream:
                subprocess.run([str(args.oracle),str(path),str(destination),request["trace_directory"],"128000"],
                               env=env,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=30)
        return generated
    greedy=run("greedy",{})
    if run("topk1",dict(do_sample=True,top_k=1,seed=42))!=greedy:
        raise ValueError("top-k1 differs from greedy")
    policy=dict(do_sample=True,temperature=0.7,top_k=42,top_p=0.963,min_p=0.01,repetition_penalty=1.1,seed=42)
    sampled=run("sample42",policy)
    if run("repeat42",policy)!=sampled:raise ValueError("seed reproducibility failed")
    if run("warm42",dict(policy,warmup_runs=1))!=sampled:raise ValueError("warmup consumed request RNG")
    run("sample43",dict(policy,seed=43)) # Difference is informative, not a flaky requirement.
    stub=args.output/"config-only";stub.mkdir();shutil.copyfile(args.checkpoint/"config.json",stub/"config.json")
    invalid=[dict(temperature=0),dict(temperature=True),dict(top_k=-1),dict(top_k=128001),dict(top_k=1.5),
             dict(top_p=0),dict(top_p=1.1),dict(min_p=-0.1),dict(min_p=1.1),dict(repetition_penalty=0),
             dict(do_sample=1),dict(seed=True),dict(seed=-1),dict(seed=2147483648)]
    for i,change in enumerate(invalid):
        request=dict(base,**change);path=args.output/f"invalid-{i}.request.json";path.write_text(json.dumps(request))
        destination=args.output/f"invalid-{i}.json"
        result=subprocess.run([str(args.binary),"--experimental",str(stub),str(path),str(destination)],
                              env=env,capture_output=True,text=True,timeout=30)
        (args.output/f"invalid-{i}.log").write_text(result.stdout+result.stderr)
        if result.returncode!=1 or destination.exists() or "cannot open" in result.stderr:
            raise ValueError(f"sampling metadata not rejected before checkpoint loading: {i}")
    if hashlib.sha256(args.binary.read_bytes()).hexdigest()!=digest:raise ValueError("binary changed")
    summary=dict(status="passed",binary_sha256=digest,seed_reproducible=True,warmup_rng_isolated=True,
                 independent_logit_replay=True,non_greedy_fixture_verified=True,
                 oracle_sha256=hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
                 topk1_equals_greedy=True,exact_reset_all_runs=True,preupload_sampling_rejections=len(invalid),
                 distinct_seeds_differ=results["sample42"]["generated_ids"]!=results["sample43"]["generated_ids"],
                 scope="finite native integration fixtures; sampler distribution tested separately; no torch RNG identity")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
