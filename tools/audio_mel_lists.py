#!/usr/bin/env python3
"""validation/audio-frontend-v1 rule 1: the clip lists for audio-mel-check, in the container's paths.
- dump40.txt: audio-in-v1's 40 dumped clips, each against the engine's recorded features
  (build/audioin-v2/native-dump/CASE.audio0.mel.f32, written by lfm3-speech audioin-v2);
- edge.txt: the 9 edge cases' 12 clips, each against its recorded features (build/server-stream-v1-check/edge/native);
- all2620.txt: every LibriSpeech test-clean utterance (references written by the original log_mel).
Usage (Thor host, in liquid-CUDA/build): audio_mel_lists.py OUT_DIR
"""
import json
import sys
from pathlib import Path

C = "/workspace/liquid-CUDA/build"


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dump40.txt", "w") as f:
        for line in open("audioin-v2/list-dump.txt"):
            req, dst = line.split()
            r = json.load(open(req.replace(C + "/", "")))
            assert len(r["audio"]) == 1, req
            f.write(f"{dst.rsplit('/', 1)[1]} {r['audio'][0]} {dst}.audio0.mel.f32\n")
    with open(out / "edge.txt", "w") as f:
        for c in json.load(open("server-stream-v1-check/edge/cases.json"))["cases"]:
            for k, wav in enumerate(c["audio"]):
                name = f"{c['case']}.audio{k}"
                f.write(f"{name} {wav} {C}/server-stream-v1-check/edge/native/{name}.mel.f32\n")
    with open(out / "all2620.txt", "w") as f:
        for c in json.load(open("audioin-v2/cases.json"))["cases"]:
            assert len(c["audio"]) == 1, c["case"]
            f.write(f"{c['case']} {c['audio'][0]}\n")


if __name__ == "__main__":
    main(sys.argv[1])
