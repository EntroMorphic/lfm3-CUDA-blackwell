#!/usr/bin/env python3
"""#2 step 4, long-form TTS by sentences (the fix the sliding-window finding points to), prototyped in HF.

Why: the backbone's five sliding-attention layers (window 256) lose sight of the prompt text the model is reading
about 180 frames into an utterance (validation/speech-long-form-v1); from there only the six full-attention layers
keep the place, and long utterances skip and loop. Utterances under ~50 words never reach that point.

The text is split into sentences (at . ! ? followed by space); a sentence under MIN_WORDS joins the next one.
  fresh:    each chunk is its own request (system + user chunk), as the single-shot TTS request is built.
  continue: one conversation; each chunk is a new user turn after the previous turn exactly as generated
            (<|audio_start|>, its frames with the end-of-audio frame, <|im_end|>), so the earlier speech is in
            context (a whole prefill per chunk: simple, not fast).
Every chunk: torch.manual_seed(seed * 1000 + chunk), generate(generation_mode="audio", do_sample=False,
audio_temperature=0.4, audio_repetition_penalty=1.2). The accepted frames of all chunks are decoded by the codec in
one pass. Writes OUT_DIR/ENGINE-MODE/CASE-sSEED/{summary.json, response.wav, response.txt} (tools/lfm3_asr.py's
layout; the summary's conversation holds the whole text as one user turn, which ASR scores against).
Usage (lfm3-omni-dev): lfm3_speech_chunked_hf.py MODEL_DIR PROMPTS.json OUT_DIR fresh|continue|requests [bfloat16|float32]
  requests: write OUT_DIR/requests/CASE-sSEED.req.json for lfm3-speech ("chunks": each sentence's fresh prompt ids;
  the native engine seeds chunk k with SEED * 1000 + k and runs the codec stream on across chunks); no model loaded.
"""
import json
import re
import sys
import time
from pathlib import Path


MIN_WORDS = 6

# Candidate rules for the splitter's open gap (no length cap), prototyped here for the split experiment (validation/
# speech-split-v1) before the native splitter changes. LFM3_SPLIT_RULE selects one; the default is sentences(), the
# validated rule. LFM3_SPLIT_CAP is the cap in words.
#   cap-comma: sentences(), then a chunk over CAP words is cut at its last clause mark (, ; : or a dash) at or before
#              word CAP that leaves both sides MIN_WORDS long; with none, at the even cut below; repeated.
#   cap-hard:  sentences(), then a chunk over CAP words is cut into the fewest equal pieces of at most CAP words.
#   full:      paragraphs (blank lines) first; sentence ends also after a closing quote or bracket; titles (Mr. Dr. ...)
#              never end a sentence, other abbreviations (a.m. U.S. Jan. ...) only before a capital; then cap-comma.
TITLES = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "mt.", "no."}
ABBREVIATIONS = {"a.m.", "p.m.", "u.s.", "u.k.", "e.g.", "i.e.", "etc.", "vs.", "jan.", "feb.", "mar.", "apr.", "jun.",
                 "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec."}


def even_cut(n, cap):
    pieces = -(-n // cap)
    return -(-n // pieces)


def cap_comma(chunk, cap):
    words = chunk.split()
    if len(words) <= cap:
        return [chunk]
    marks = [i + 1 for i in range(len(words) - 1)
             if (words[i][-1] in ",;:\u2014\u2013" or words[i + 1] in ("-", "\u2014", "\u2013"))
             and MIN_WORDS <= i + 1 <= cap and len(words) - (i + 1) >= MIN_WORDS]
    cut = max(marks) if marks else even_cut(len(words), cap)
    return [" ".join(words[:cut])] + cap_comma(" ".join(words[cut:]), cap)


def cap_hard(chunk, cap):
    words = chunk.split()
    if len(words) <= cap:
        return [chunk]
    cut = even_cut(len(words), cap)
    return [" ".join(words[:cut])] + cap_hard(" ".join(words[cut:]), cap)


def merge_short(parts):
    out = []
    for p in parts:
        if out and len(out[-1].split()) < MIN_WORDS:
            out[-1] += " " + p
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1].split()) < MIN_WORDS:
        last = out.pop()
        out[-1] += " " + last
    return out


def full_sentences(paragraph):
    words, parts, cur = paragraph.split(), [], []
    for k, w in enumerate(words):
        cur.append(w)
        bare = w.rstrip("\"'\u201d\u2019)]")
        if k + 1 < len(words) and bare[-1:] in (".", "!", "?"):
            low = bare.lower()
            nxt = words[k + 1].lstrip("\"'\u201c\u2018([")
            if low in TITLES or (low in ABBREVIATIONS and not nxt[:1].isupper()):
                continue
            parts.append(" ".join(cur)); cur = []
    if cur:
        parts.append(" ".join(cur))
    return parts


def split_text(text, rule=None, cap=None):
    import os
    rule = rule or os.environ.get("LFM3_SPLIT_RULE", "sentences")
    cap = int(cap or os.environ.get("LFM3_SPLIT_CAP", "0"))
    if rule == "sentences":
        return sentences(text)
    assert cap >= 2 * MIN_WORDS, "LFM3_SPLIT_CAP must be at least 12"
    if rule in ("cap-comma", "cap-hard"):
        f = cap_comma if rule == "cap-comma" else cap_hard
        return [q for c in sentences(text) for q in f(c, cap)]
    if rule == "full":
        paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        chunks = merge_short([q for p in paragraphs for q in full_sentences(p)])
        return [q for c in chunks for q in cap_comma(c, cap)]
    raise SystemExit(f"unknown LFM3_SPLIT_RULE {rule}")


def sentences(text):
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    out = []
    for p in parts:
        if out and len(out[-1].split()) < MIN_WORDS:
            out[-1] += " " + p
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1].split()) < MIN_WORDS:
        last = out.pop()  # (out[-2] += out.pop() would read one item and write another)
        out[-1] += " " + last
    return out


def main(model_dir, prompts, out_dir, mode, dtype="bfloat16"):
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_grad_enabled(False)
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    assert mode in ("fresh", "continue", "requests")
    if mode == "requests":
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        spec = json.load(open(prompts))
        (Path(out_dir) / "requests").mkdir(parents=True, exist_ok=True)
        for c in spec["cases"]:
            system = spec["systems"][c["system"]]
            chunks = [processor.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": t}],
                                                    add_generation_prompt=True, tokenize=True, return_dict=True,
                                                    return_tensors="pt")["input_ids"][0].tolist() for t in split_text(c["user"])]
            for seed in spec["seeds"]:
                json.dump(dict(prompt_ids=chunks[0], chunks=chunks, chunk_texts=split_text(c["user"]), mode="audio", max_new_tokens=c["max_new_tokens"],
                               seed=seed, conversation=[{"role": "system", "content": system}, {"role": "user", "content": c["user"]}]),
                          open(Path(out_dir) / "requests" / f"{c['case']}-s{seed}.req.json", "w"))
        return
    if dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype), device_map="cuda",
                                                     trust_remote_code=True, attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    tok = processor.tokenizer
    codec = processor.audio_tokenizer.to(device="cuda").eval()
    ao, eos_code = model.config.audio_out_token_id, 2016
    engine = ("hf" if dtype == "bfloat16" else "hf-fp32") + "-" + mode
    spec = json.load(open(prompts))
    for c in spec["cases"]:
        system = spec["systems"][c["system"]]
        chunks = split_text(c["user"])
        for seed in spec["seeds"]:
            name = f"{c['case']}-s{seed}"
            d = Path(out_dir) / engine / name
            d.mkdir(parents=True, exist_ok=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            history_ids, history_codes, frames, per_chunk = None, [], [], []
            for k, chunk in enumerate(chunks):
                if mode == "fresh" or history_ids is None:
                    conv = [{"role": "system", "content": system}, {"role": "user", "content": chunk}]
                    ids = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                                        return_dict=True, return_tensors="pt")["input_ids"][0].tolist()
                    codes_in = []
                else:
                    suffix = tok.encode(f"\n<|im_start|>user\n{chunk}<|im_end|>\n<|im_start|>assistant\n<think></think>",
                                        add_special_tokens=False)
                    ids, codes_in = history_ids + suffix, history_codes
                inp = dict(input_ids=torch.tensor([ids], device="cuda"),
                           attention_mask=torch.ones(1, len(ids), dtype=torch.long, device="cuda"))
                if codes_in:
                    inp["audio_out_codes"] = torch.tensor(codes_in, device="cuda")
                torch.manual_seed(seed * 1000 + k)
                o = model.generate(**inp, max_new_tokens=c["max_new_tokens"], generation_mode="audio", do_sample=False,
                                   audio_temperature=0.4, audio_repetition_penalty=1.2, return_audio_eos=True)
                slots = o.sequences[0, len(ids):o.sequence_lengths[0]].tolist()
                n = int(o.audio_lengths[0])
                got = o.audio_codes[0, :, :n].T.tolist()  # generation order, end-of-audio frames kept
                speech = [f for f in got if eos_code not in f]
                frames += speech
                ended = bool(got) and eos_code in got[-1] and slots[-1] == tok.convert_tokens_to_ids("<|im_end|>")
                per_chunk.append(dict(text=chunk, frames=len(speech), slots=len(slots), ended_by_eos=ended))
                history_ids = ids + slots
                history_codes = codes_in + got
                assert history_ids.count(ao) == len(history_codes), (history_ids.count(ao), len(history_codes))
            wav = codec.decode(torch.tensor(frames, device="cuda").T[None]).audio_values[0].squeeze(0).float().cpu().numpy() \
                if frames else np.zeros(0, np.float32)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            sf.write(d / "response.wav", wav, 24_000)
            (d / "response.txt").write_text("\n")
            (d / "summary.json").write_text(json.dumps(dict(
                case=name, mode="audio", seed=seed, engine=engine,
                conversation=[{"role": "system", "content": system}, {"role": "user", "content": c["user"]}], text="",
                audio_seconds=len(wav) / 24_000, frames=len(frames), slots=sum(p["slots"] for p in per_chunk),
                chunks=per_chunk, generate_seconds=seconds,
                real_time_factor=(len(wav) / 24_000) / seconds if seconds else 0), indent=1) + "\n")
            print(name, engine, f"{len(chunks)} chunks, {len(frames)} frames, {len(wav) / 24_000:.1f}s audio in {seconds:.1f}s,",
                  "all ended by EOS" if all(p["ended_by_eos"] for p in per_chunk) else "NOT all ended by EOS", flush=True)


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    main(*sys.argv[1:])
