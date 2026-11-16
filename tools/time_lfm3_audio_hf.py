#!/usr/bin/env python3
"""validation/audio-speed-v1 rule 0: HF's time per stage on audio-in ASR requests, LFM3-Omni-2B in BF16 on the GPU
(SDPA, greedy, generation_mode "text"): audio-in-v1's reference configuration (tools/lfm3_audio_in_hf.py).
Per clip and pass: the clip read (soundfile, float32), the processor call (chat template, feature extraction and
tokenizing; CPU), the feature extractor alone (a separate call), the audio encoder and projector alone
(get_audio_features, a separate call), time to first token (generate with a streamer: from the call to the first new
token), and decode (first new token to the last). Every GPU stage ends with a CUDA synchronize. One untimed pass over
all clips, then PASSES timed passes; one JSONL row per clip and timed pass.
Usage: time_lfm3_audio_hf.py MODEL_DIR OUT_JSONL PASSES MAX_NEW_TOKENS NAME=WAV...
"""
import json
import platform
import sys
import time


def main(argv):
    if len(argv) < 6:
        sys.exit(__doc__)
    model_dir, out_jsonl, passes, max_new = argv[1], argv[2], int(argv[3]), int(argv[4])
    clips = [a.split("=", 1) for a in argv[5:]]
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from transformers.generation.streamers import BaseStreamer

    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda",
                                                     attn_implementation="sdpa", trust_remote_code=True).eval()
    sync = torch.cuda.synchronize

    class FirstToken(BaseStreamer):
        def __init__(self):
            self.calls, self.first = 0, None

        def put(self, value):
            self.calls += 1
            if self.calls == 2 and self.first is None:  # the first put is the prompt; the second, the first new token
                sync()
                self.first = time.perf_counter()

        def end(self):
            pass

    def run(wav):
        t0 = time.perf_counter()
        x, rate = sf.read(wav, dtype="float32", always_2d=False)
        if rate != 16_000 or x.ndim != 1:
            raise SystemExit(f"{wav}: expected 16 kHz mono")
        t1 = time.perf_counter()
        conv = [{"role": "system", "content": "Perform ASR in English."},
                {"role": "user", "content": [{"type": "audio", "audio": x}]}]
        inputs = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt").to("cuda")
        sync()
        t2 = time.perf_counter()
        processor.feature_extractor([x], sampling_rate=16_000, return_tensors="pt")
        t3 = time.perf_counter()
        model.model.get_audio_features(inputs["input_features"], inputs["feature_attention_mask"])
        sync()
        t4 = time.perf_counter()
        streamer = FirstToken()
        t5 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=max_new, generation_mode="text", do_sample=False,
                             streamer=streamer, return_dict_in_generate=True)
        sync()
        t6 = time.perf_counter()
        n = inputs["input_ids"].shape[1]
        end = int(out.sequence_lengths[0]) if getattr(out, "sequence_lengths", None) is not None else out.sequences.shape[1]
        first = streamer.first if streamer.first is not None else t6
        return dict(read_seconds=t1 - t0, processor_seconds=t2 - t1, feature_extractor_seconds=t3 - t2,
                    encode_seconds=t4 - t3, first_token_seconds=first - t5, decode_seconds=t6 - first,
                    time_to_first_token_seconds=(t2 - t0) + (first - t5),
                    end_to_end_seconds=(t2 - t0) + (t6 - t5), audio_seconds=len(x) / 16_000, prompt_tokens=n,
                    new_tokens=end - n)

    env = dict(machine=platform.machine(), transformers=transformers.__version__, torch=torch.__version__,
               device=torch.cuda.get_device_name(), dtype="bfloat16", attn="sdpa", passes=passes,
               max_new_tokens=max_new)
    with open(out_jsonl, "w") as f:
        f.write(json.dumps(dict(env=env)) + "\n")
        for _, wav in clips:  # untimed pass
            run(wav)
        for k in range(passes):
            for name, wav in clips:
                row = dict(name=name, pass_=k, **run(wav))
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main(sys.argv)
