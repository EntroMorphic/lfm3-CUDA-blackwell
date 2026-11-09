"""Long-form TTS by sentences, natively (native/include/lfm/speech_text.hpp) through lfm3-text-tool.

The reference is the validated Python prototype: tools/lfm3_speech_chunked_hf.py sentences() for the split, and the
HF processor's chat template for each sentence's prompt ids (validation/speech-long-form-v1: WER 0.000 on 30/30
native runs built from those ids). Equal ids mean an identical generation.
"""
import importlib.util
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT.parent / "models" / "LFM3-Omni-2B"
TOKENIZER = CHECKPOINT / "tokenizer.json"
SPEC = importlib.util.spec_from_file_location("chunked", ROOT / "tools/lfm3_speech_chunked_hf.py")
chunked = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chunked)

pytestmark = pytest.mark.skipif(not TOKENIZER.exists(), reason="checkpoint tokenizer required")


@pytest.fixture(scope="module")
def tool(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path_factory.mktemp("speech-text") / "lfm3-text-tool"
    subprocess.run(["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/tests/lfm3_text_tool.cpp"), "-o", str(binary)], check=True)
    return binary


def run(tool, tmp_path, mode, rows):
    src, dst = tmp_path / f"{mode}.in", tmp_path / f"{mode}.out"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    subprocess.run([str(tool), str(TOKENIZER), mode, str(src), str(dst)], check=True, capture_output=True)
    return [json.loads(line) for line in dst.read_text(encoding="utf-8").split("\n") if line]  # not splitlines(): U+2028


def test_whitespace_is_pythons_on_every_code_point(tool, tmp_path):
    native = run(tool, tmp_path, "spaces", [dict({"from": 0, "to": 0x110000})])[0]["spaces"]
    isspace = [c for c in range(0x110000) if chr(c).isspace()]
    regex = [c for c in range(0x110000) if re.fullmatch(r"\s", chr(c))]
    assert isspace == regex  # str.split()/strip() and the split regex agree on whitespace
    assert native == isspace


HAND = [
    "", "   ", "One.", "Hello world.", "a. b. c. d. e. f. g.",
    "This is a sentence of eight words in total. And here is another one that is long enough.",
    "Short one. This second sentence is comfortably long enough to stand. Tiny end.",
    "Wait... what? Really!   Yes.\tThat is how it works, every single time we try it out.",
    "Mr. Smith went to Washington on a very rainy day. He stayed for a week and then flew home.",
    "No terminal punctuation at all here but plenty of words to count",
    "Trailing spaces after the end of the sentence here.   \n\n",
    "　Ideographic space before. Not a break because of the nbsp? Actually it is one.",
    "Unicode — dashes, café, naïve, 漢字 and emoji 🙂 in a sentence. Then a second sentence with more words.",
    "Ends with a question? Ends with a bang! Ends with a period. And then one more full sentence here.",
    "x.y.z is not split. Neither is 3.14 or e.g. this, because no space follows those dots at all.",
    "Line one ends here.\nLine two starts here and keeps going for a while.\r\nLine three.",
    "A. B is separated by a line separator and has many words after it to count.",
]


def test_split_matches_python_on_edge_cases(tool, tmp_path):
    got = run(tool, tmp_path, "split", [{"text": t} for t in HAND])
    for t, g in zip(HAND, got):
        assert g["sentences"] == chunked.sentences(t), t


def test_short_last_sentence_joins_the_one_before(tool, tmp_path):
    # the prototype's first version crashed on two parts and overwrote the wrong one on three (never hit by a
    # validation prompt); both now give the intended join
    texts = ["This first sentence has plenty of words in it. Short end.",
             "This first sentence has plenty of words in it. The second one is also long enough. Short end."]
    want = [["This first sentence has plenty of words in it. Short end."],
            ["This first sentence has plenty of words in it.", "The second one is also long enough. Short end."]]
    assert [chunked.sentences(t) for t in texts] == want
    assert [g["sentences"] for g in run(tool, tmp_path, "split", [{"text": t} for t in texts])] == want


def test_split_matches_python_fuzzed(tool, tmp_path):
    rng = random.Random(20261109)
    words = ["the", "a", "lighthouse", "keeper", "café", "漢字", "3.14", "e.g.", "Mr.", "🙂", "x"]
    puncts = [".", "!", "?", "...", "?!", ",", ";", ""]
    spaces = [" ", "  ", "\t", "\n", " ", "　", " ", "\r\n", "\x1f", "\x85", "​"]
    texts = []
    for _ in range(5000):
        parts = []
        for _ in range(rng.randrange(0, 40)):
            parts.append(rng.choice(words) + (rng.choice(puncts) if rng.random() < 0.3 else ""))
            parts.append(rng.choice(spaces) if rng.random() < 0.9 else "")
        if rng.random() < 0.3:
            parts.insert(0, rng.choice(spaces))
        texts.append("".join(parts))
    got = run(tool, tmp_path, "split", [{"text": t} for t in texts])
    mismatches = [t for t, g in zip(texts, got) if g.get("sentences") != chunked.sentences(t)]
    assert not mismatches, mismatches[:3]


def prompt_cases():
    seen = {}
    for f in sorted((ROOT / "validation/speech-e2e-v1").glob("prompts*.json")):
        spec = json.loads(f.read_text())
        for c in spec["cases"]:
            seen[(spec["systems"][c["system"]], c["user"])] = c["max_new_tokens"]
    return seen


def test_every_validation_prompt_matches_hf_ids(tool, tmp_path):
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    cases = prompt_cases()
    rows = [dict(messages=[{"role": "system", "content": s}, {"role": "user", "content": u}], capacity=4096,
                 max_new_tokens=n) for (s, u), n in cases.items()]
    got = run(tool, tmp_path, "speech", rows)
    split_any = 0
    for ((system, user), _), g in zip(cases.items(), got):
        texts = chunked.sentences(user)
        assert g["texts"] == texts
        split_any += len(texts) > 1
        want = [processor.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": t}],
                                              add_generation_prompt=True, tokenize=True, return_dict=True,
                                              return_tensors="pt")["input_ids"][0].tolist() for t in texts]
        assert g["ids"] == want, user
    assert len(rows) >= 10 and split_any >= 3


@pytest.mark.parametrize("messages, why", [
    ([{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "x"}],
     "must be [system, user] or [user]"),
    ([{"role": "system", "content": "s"}], "must be [system, user] or [user]"),
    ([{"role": "user", "content": [{"type": "text", "text": "Hello there."}]}], "user content must be a string"),
    ([{"role": "user", "content": " \n "}], "user text is empty"),
])
def test_speech_split_refuses(tool, tmp_path, messages, why):
    got = run(tool, tmp_path, "speech", [dict(messages=messages, capacity=4096, max_new_tokens=16)])[0]
    assert why in got["error"]
