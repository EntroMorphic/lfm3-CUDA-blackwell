"""The native speech normalizer (native/include/lfm/speech_normalize.hpp) against its reference
(tools/speech_normalize.py), byte for byte, through native/tests/speech_normalize_tool.cpp built plain and with
ASan+UBSan.

Inputs: the experiment's 31 sentences (validation/speech-normalize-v1); every line of the local wikitext-2 and PG-19
test data that holds a digit, a symbol the rules read, or a period after a short word (when the data is present); and
30,000 generated strings mixing the rules' ingredients (numbers in every format they read, currency, degree, percent
and minus signs, dashes, times, dates, units, abbreviations, capitals, odd whitespace and punctuation, non-ASCII text).
Also: the rules' own expectations on a few sentences, and time on a long text (linear passes).
"""
import importlib.util
import json
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("speech_normalize", ROOT / "tools/speech_normalize.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)
VARIANTS = {"plain": ["-O2"], "asan": ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]}


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def tool(tmp_path_factory, request):
    if not shutil.which("g++"):
        pytest.skip("g++ required")
    binary = tmp_path_factory.mktemp("normalize") / f"speech-normalize-tool-{request.param}"
    subprocess.run(["g++", "-std=c++20", "-Wall", "-Wextra", "-Werror", *VARIANTS[request.param], "-I",
                    str(ROOT / "native/include"), str(ROOT / "native/tests/speech_normalize_tool.cpp"), "-o", str(binary)],
                   check=True)
    return binary


def native(tool, tmp_path, texts):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text("".join(json.dumps(t) + "\n" for t in texts), encoding="utf-8")
    subprocess.run([str(tool), str(src), str(dst)], check=True, capture_output=True, timeout=1800)
    return [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]


def compare(tool, tmp_path, texts):
    got = native(tool, tmp_path, texts)
    assert len(got) == len(texts)
    bad = [(t, g, reference.normalize(t)) for t, g in zip(texts, got) if g != reference.normalize(t)]
    assert not bad, "\n".join(f"in:  {t!r}\nC++: {g!r}\nref: {w!r}" for t, g, w in bad[:5]) + f"\n({len(bad)} differ)"


def test_experiment_sentences(tool, tmp_path):
    texts = [c["text"] for s in ("dev", "heldout")
             for c in json.load(open(ROOT / f"validation/speech-normalize-v1/cases-{s}.json"))["cases"]]
    compare(tool, tmp_path, texts)


def corpus_lines():
    data = ROOT.parent / "data"
    files = sorted((data / "wikitext-2-raw-v1").rglob("*")) + sorted((data / "pg19-test").rglob("*.txt"))[:20]
    want = re.compile(r"\d|[$£€%°#&+=×−–]|\b[A-Za-z]{1,6}\.")
    lines = []
    for f in files:
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if want.search(line) and len(line) < 4000:
                lines.append(line)
    return lines


def test_real_text(tool, tmp_path):
    lines = corpus_lines()
    if len(lines) < 1000:
        pytest.skip("corpus data not present")
    compare(tool, tmp_path, lines[:40000])


PIECES = [
    "", " ", "  ", "\t", "\n", ",", ".", ". ", ", ", ";", ":", "!", "?", "(", ")", "\"", "'", "“", "”", "-", "--", "–",
    "—", "/", "#", "%", "°", "°C", "°F", " °", "$", "£", "€", "−", "×", "&", "+", "=", "_", "x", "a", "A", "é", "日本",
    "0", "1", "7", "9", "12", "07", "389", "911", "1,500", "1,234,567", "12,34", "1,0000", "3.5", "0.25", "2.", ".5",
    "1984", "2026", "1500", "2000", "2005", "1905", "12345", "02139", "1234567", "99999999999999999999",
    "7:45", "9:00", "23:59", "24:00", "1:5", "12:30:45", "AM", "PM", "a.m.", "p.m.", "am", "pm",
    "1st", "2nd", "3rd", "11th", "22nd", "1/2", "3/4", "2/3", "11/9/2026", "13/45/2020", "25-30", "1990-1995",
    "555-0142", "(555) 123-4567", "555-123-4567", "km", "kg", "mph", "ft", "ft.", "lbs", "min", "hr", "GB",
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Mt.", "Jr.", "vs.", "vs", "etc.", "e.g.", "i.e.", "No.", "Inc.",
    "Co.", "Corp.", "Apt.", "Ave.", "Rd.", "min.", "Jan.", "Sept.", "Sep.", "Dec.", "Nov.",
    "$0.01", "$1.01", "£2.01", "€0.01", "$12.10", "4:10", "12:10", "0:10", "123456", "100000", "1050-1080", "1000-1099",
    "999-2001", "2099-2100", "12 Main St.", "123456 Main St.", "1600 Elm Way",
    "January", "March", "May", "December", "Main", "Elm", "Street", "Avenue", "Drive", "Way", "in", "since", "the",
    "zip", "ZIP", "code", "postal", "year", "by", "is", "Patel", "Smith", "Paul", "1990s", "2000s", "'90s",
]


def fuzz(seed, count):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        parts = [rng.choice(PIECES) for _ in range(rng.randint(1, 14))]
        seps = [rng.choice(["", " ", " ", " ", "  ", ", ", "\n"]) for _ in parts]
        out.append("".join(p + s for p, s in zip(parts, seps)))
    return out


def test_fuzz(tool, tmp_path):
    compare(tool, tmp_path, fuzz(11, 30000))


def test_rules_read_as_intended(tool, tmp_path):
    cases = {
        "It opens at 9:00 AM.": "It opens at nine AM.",
        "Call 911 now.": "Call nine one one now.",
        "The zip code is 12345.": "The zip code is one two three four five.",
        "It was 20° today.": "It was twenty degrees today.",
        "Elm Dr. is closed; see Dr. Who.": "Elm Drive is closed; see Doctor Who.",
        "I live at 12 Oak St. It is quiet.": "I live at twelve Oak Street. It is quiet.",
        "Bring fruit, nuts, etc. Then leave.": "Bring fruit, nuts, et cetera. Then leave.",
        "The year 1999 ended.": "The year nineteen ninety-nine ended.",
        "1500 people came in 1500.": "one thousand five hundred people came in fifteen hundred.",
    }
    assert native(tool, tmp_path, list(cases)) == list(cases.values())


def test_long_text_is_linear(tool, tmp_path):
    long = " ".join(["On Jan. 3, 2025 at 10:30 a.m., Dr. Lee paid $1,234.56 for 12 km of 1990s road, e.g. 45%."] * 4000)
    t0 = time.time()
    got = native(tool, tmp_path, [long])[0]
    assert got == reference.normalize(long) and time.time() - t0 < 30
