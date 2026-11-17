#!/usr/bin/env python3
"""validation/tcgen05-gemm-v1 rule 4b: the server soak with images and long text (Thor, lfm3-profile-v1; quiet).

tools/check_lfm3_audio_soak.py unchanged (its references alone, the timed soak under CLIENTS threads, idle memory
checkpoints, the crowd at the end, and its pass rule: every completed reply identical to its reference, no 5xx or
transport error, both processes alive, memory flat at idle, exit 0), with its case list extended by
  - the 11 image stage-0 conversations (tools/thor/vision_e2e_round.sh's requests; the two 4K photographs among them),
    images sent as base64 as a client sends them (check_lfm3_image_server.as_client), text mode, 128 new tokens;
  - 3 long text prompts through the same endpoint, text mode, 32 new tokens: about 300, 1,500 and 3,800 prompt tokens
    (the language model's prefill at M >= 257, on tcgen05, beside speech decode and image encodes).
These ride the soak's case draw (unstreamed, streamed, dropped), so speech streams, images and long prefills run
concurrently; each completed reply is compared with the same request served alone.
Usage: check_lfm3_tcgen05_soak.py SERVER MODEL_DIR BUILD_DIR LIBRI_WAV_DIR IMAGE_ROUND_DIR OUT_JSON [SECONDS [CLIENTS [TEXT_SHARE]]]
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("soak", HERE / "check_lfm3_audio_soak.py")
soak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soak)
sys.path.insert(0, str(HERE))
from check_lfm3_image_server import as_client  # noqa: E402

PARAGRAPH = ("The harbor town kept its old customs: at dawn the fishing boats left in a fixed order, the baker opened "
             "before the bell, and the lighthouse keeper logged the weather in a book older than anyone living. ")


def long_text(paragraphs):
    body = "".join(f"Note {i + 1}. {PARAGRAPH}" for i in range(paragraphs))
    return {"messages": [{"role": "user", "content": body + "\nSummarize these notes in two sentences."}],
            "mode": "text", "max_new_tokens": 32}


def extra_cases(image_round):
    out = []
    for req in sorted(Path(image_round, "req").glob("*.json")):
        out.append((f"image-{req.stem}", "image", as_client(json.loads(req.read_text()))))
    for name, n in (("long-300", 6), ("long-1500", 32), ("long-3800", 82)):
        out.append((f"text-{name}", "long-text", long_text(n)))
    return out


def main(binary, model, build, libri, image_round, out_json, *rest):
    original = soak.check.cases
    extras = extra_cases(image_round)
    soak.check.cases = lambda b, l: list(original(b, l)) + extras
    print(f"cases: {len(original(build, libri))} speech + {len(extras)} image and long-text", flush=True)
    return soak.main(binary, model, build, libri, out_json, *rest)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
