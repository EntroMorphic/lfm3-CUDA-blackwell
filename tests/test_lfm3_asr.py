import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lfm3_asr as asr  # noqa: E402


def test_word_errors():
    ref = "the quick brown fox".split()
    assert asr.word_errors(ref, ref) == 0
    assert asr.word_errors(ref, "the quick fox".split()) == 1            # deletion
    assert asr.word_errors(ref, "the quick brown red fox".split()) == 1  # insertion
    assert asr.word_errors(ref, "a quick brown box".split()) == 2         # substitutions
    assert asr.word_errors(ref, []) == 4 and asr.word_errors([], ["x"]) == 1


def test_intended_text():
    conv = [{"role": "system", "content": "s"}, {"role": "user", "content": "say this"}]
    assert asr.intended_text(dict(mode="audio", conversation=conv, text="")) == "say this"
    assert asr.intended_text(dict(mode="interleaved", conversation=conv, text="spoken reply")) == "spoken reply"


def test_ascii_punctuation():
    assert asr.ascii_punctuation("they’ve self‑reflection “yes”") == "they've self-reflection \"yes\""
    assert asr.ascii_punctuation("plain ascii") == "plain ascii"


def test_fold_ordinals():
    assert asr.fold_ordinals(["march", "3rd", "21st", "2024", "4th", "north", "3"]) == \
        ["march", "3", "21", "2024", "4", "north", "3"]
