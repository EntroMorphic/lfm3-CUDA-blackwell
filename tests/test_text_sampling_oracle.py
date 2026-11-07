import json
from pathlib import Path
import shutil
import struct
import subprocess

import pytest


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    compiler=shutil.which("g++")
    if not compiler:pytest.skip("g++ required")
    root=Path(__file__).resolve().parents[1]
    directory=tmp_path_factory.mktemp("sampling-oracle")
    binary=directory/"oracle"
    subprocess.run([compiler,"-std=c++17","-O2","-I",str(root/"native/include"),
                    str(root/"native/tests/text_sampling_oracle.cpp"),"-o",str(binary)],check=True)
    return binary


@pytest.mark.parametrize("damage",[None,"greedy-bypass","wrong-id","short-trace","missing-trace","empty-output"])
def test_replay_gate_catches_greedy_bypass(oracle,tmp_path,damage):
    request=dict(do_sample=True,temperature=1.0,top_k=0,top_p=1.0,min_p=0.0,
                 repetition_penalty=1.0,seed=42,suppress_tokens=[])
    (tmp_path/"request.json").write_text(json.dumps(request))
    # mt19937_64 seed42 first draw maps to0.7551555...; uniform4-way support =>ID3.
    generated=[] if damage=="empty-output" else [0 if damage=="greedy-bypass" else 2 if damage=="wrong-id" else 3]
    (tmp_path/"result.json").write_text(json.dumps(dict(generated_ids=generated)))
    if damage!="missing-trace":
        (tmp_path/"0.logits.bf16").write_bytes(struct.pack("<"+"H"*(3 if damage=="short-trace" else 4),
                                                        *([0]*(3 if damage=="short-trace" else 4))))
    result=subprocess.run([str(oracle),str(tmp_path/"request.json"),str(tmp_path/"result.json"),str(tmp_path),"4"],capture_output=True,text=True)
    assert result.returncode==(0 if damage is None else 1),result.stderr


def _tokenizer(tmp_path,count):
    path=tmp_path/"tokenizer.json"
    path.write_text(json.dumps({"model":{"type":"BPE","vocab":{f"t{i}":i for i in range(count)}},"added_tokens":[]}))
    return str(path)


def _replay(oracle,tmp_path,generated,suppress=(),top_k=0,vocab="4",tokenizer=None):
    request=dict(do_sample=True,temperature=1.0,top_k=top_k,top_p=1.0,min_p=0.0,
                 repetition_penalty=1.0,seed=42,suppress_tokens=list(suppress))
    (tmp_path/"request.json").write_text(json.dumps(request))
    (tmp_path/"result.json").write_text(json.dumps(dict(generated_ids=[generated])))
    (tmp_path/"0.logits.bf16").write_bytes(struct.pack("<HHHH",0,0,0,0))
    args=[str(oracle),str(tmp_path/"request.json"),str(tmp_path/"result.json"),str(tmp_path),vocab]
    if tokenizer is not None:args.append(tokenizer)
    return subprocess.run(args,capture_output=True,text=True)


# G6: traces hold every output row; only the ids the tokenizer defines may be chosen. The oracle counts
# them with the engine's own lfm::tokenizer_token_count. Seed42's first draw is 0.755156...: over uniform
# 3-way support it selects ID 2; over 4-way, ID 3.
@pytest.mark.parametrize("generated,tokens,ok",[(2,3,True),(3,3,False),(3,4,True)])
def test_replay_respects_the_tokenizer_bound(oracle,tmp_path,generated,tokens,ok):
    result=_replay(oracle,tmp_path,generated,tokenizer=_tokenizer(tmp_path,tokens))
    assert result.returncode==(0 if ok else 1),result.stderr
    if ok:assert f"live_token_count {tokens}" in result.stdout


def test_replay_rejects_a_tokenizer_larger_than_the_trace(oracle,tmp_path):
    result=_replay(oracle,tmp_path,2,tokenizer=_tokenizer(tmp_path,5))
    assert result.returncode==1 and "outside the model vocabulary" in result.stderr


def test_replay_rejects_a_missing_tokenizer(oracle,tmp_path):
    result=_replay(oracle,tmp_path,2,tokenizer=str(tmp_path/"absent.json"))
    assert result.returncode==1 and "cannot open tokenizer" in result.stderr


@pytest.mark.parametrize("suppress,ok",[([3],True),([0],True),([4],False)])
def test_replay_drops_suppression_beyond_live_like_the_worker(oracle,tmp_path,suppress,ok):
    # Worker rule: suppression ids >= the token count are already excluded (dropped, not an error); ids
    # outside the trace's rows stay invalid. Support {0,1,2} or {1,2}; 0.755 selects ID 2 either way.
    result=_replay(oracle,tmp_path,2,suppress=suppress,tokenizer=_tokenizer(tmp_path,3))
    assert result.returncode==(0 if ok else 1),result.stderr


@pytest.mark.parametrize("top_k,ok",[(3,True),(4,True),(5,False)])
def test_replay_clamps_top_k_to_the_token_count_like_the_engines(oracle,tmp_path,top_k,ok):
    # The engines validate top_k against the output rows (4 here) and clamp it to the token count (3):
    # top_k 4 must replay exactly like "all tokens"; top_k 5 exceeds the rows and is invalid.
    result=_replay(oracle,tmp_path,2,top_k=top_k,tokenizer=_tokenizer(tmp_path,3))
    assert result.returncode==(0 if ok else 1),result.stderr
    if not ok:assert "top_k exceeds vocabulary size" in result.stderr


@pytest.mark.parametrize("vocab",["4.0","x","","-4","4abc"])
def test_replay_rejects_malformed_vocabulary_argument(oracle,tmp_path,vocab):
    result=_replay(oracle,tmp_path,2,vocab=vocab)
    assert result.returncode==1 and "invalid" in result.stderr,(result.returncode,result.stderr)


# G4: an FP32-logit engine (the native-GEMM build) samples from exact FP32 logits and writes them as
# STEP.logits.f32 next to the rounded BF16 trace. The oracle must replay the exact values. Here the BF16
# trace is uniform over 4 ids (seed 42 selects ID 3), but the exact trace puts ID 3 at -1000, leaving a
# uniform 3-way support (seed 42 selects ID 2): only a replay of the FP32 file accepts ID 2.
@pytest.mark.parametrize("generated,exact,ok",[(2,"good",True),(3,"good",False),(3,None,True),(2,"short",False)])
def test_replay_prefers_the_exact_fp32_trace(oracle,tmp_path,generated,exact,ok):
    if exact=="good":(tmp_path/"0.logits.f32").write_bytes(struct.pack("<ffff",0.0,0.0,0.0,-1000.0))
    if exact=="short":(tmp_path/"0.logits.f32").write_bytes(struct.pack("<fff",0.0,0.0,0.0))
    result=_replay(oracle,tmp_path,generated)
    assert result.returncode==(0 if ok else 1),result.stderr
    if exact=="short":assert "wrong-size FP32 trace" in result.stderr
