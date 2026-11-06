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


@pytest.mark.parametrize("generated,live,ok",[(2,"3",True),(3,"3",False),(3,"4",True),(2,"5",False),(2,"0",False)])
def test_replay_respects_live_token_bound(oracle,tmp_path,generated,live,ok):
    # G6: traces hold every output row; only the first LIVE ids (those the tokenizer defines) may be chosen.
    # Seed42's first draw 0.7551... over uniform 3-way support selects ID 2; over 4-way, ID 3.
    request=dict(do_sample=True,temperature=1.0,top_k=0,top_p=1.0,min_p=0.0,
                 repetition_penalty=1.0,seed=42,suppress_tokens=[])
    (tmp_path/"request.json").write_text(json.dumps(request))
    (tmp_path/"result.json").write_text(json.dumps(dict(generated_ids=[generated])))
    (tmp_path/"0.logits.bf16").write_bytes(struct.pack("<HHHH",0,0,0,0))
    result=subprocess.run([str(oracle),str(tmp_path/"request.json"),str(tmp_path/"result.json"),str(tmp_path),"4",live],capture_output=True,text=True)
    assert result.returncode==(0 if ok else 1),result.stderr
