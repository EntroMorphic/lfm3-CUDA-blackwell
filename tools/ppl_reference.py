"""Reference perplexity for LFM3-Omni-2B with the standard Hugging Face stack (teacher forced).

Builds the shared evaluation windows (reference tokenizer, so every engine scores identical ids) and
scores them with Transformers in each requested attention configuration. The language model is
called exactly as Lfm3OmniForConditionalGeneration._generate prefills (empty DynamicCache, all-ones
attention mask, explicit positions), so SDPA picks the same kernels as real use, including the
flash -> mem-efficient switch on sliding layers past 256 tokens. Logits come from the tied lm_head
in 256-row tiles; NLL = -log_softmax(logits.float())[next token] over all 128000 logits.

Run in the reference container (lfm3-omni-dev) with Thor's ptxas for the Triton conv kernels:
  TRITON_PTXAS_PATH=/workspace/tools/ptxas TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas \\
  python /workspace/liquid-CUDA/tools/ppl_reference.py --text /workspace/data/wikitext-2-raw-v1/test.txt \\
      --output /workspace/liquid-CUDA/build/ppl-v1 --lengths 256 257 4096 32768

Writes windows-L.i32 (int32, W x L ids), nll-hf-CONFIG-L.f32 (float32, W x (L-1)) with a
nll-hf-CONFIG-L.meta.json binding it to the windows it scored, and summary.json. ppl_compare.py
refuses any NLL file whose metadata does not match.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.cache_utils import DynamicCache


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_meta(path, **fields):
    path.write_text(json.dumps(fields, indent=2) + "\n")


def build_windows(tokens, bos, length, max_windows):
    step = length - 1
    count = len(tokens) // step
    if count < 1:
        raise ValueError(f"corpus has {len(tokens)} tokens, fewer than one window of length {length}")
    if max_windows:
        count = min(count, max_windows)
    rows = [[bos] + tokens[i * step:(i + 1) * step] for i in range(count)]
    return np.asarray(rows, dtype=np.int32)


def register_eager_blocked(block_rows):
    """Eager attention computed in blocks of query rows: the same math as the model's eager_attention_forward
    (scores, additive eager mask, FP32 softmax, P.V) without materializing all L x L scores at once. In FP32 at
    32,768 tokens, SDPA falls back to its unfused math kernel and needs over 96 GB (validation/g4-gates-v2/)."""
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    def eager_blocked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        groups = module.num_key_value_groups
        key = key.repeat_interleave(groups, dim=1) if groups > 1 else key
        value = value.repeat_interleave(groups, dim=1) if groups > 1 else value
        out = torch.empty_like(query[..., :value.shape[-1]])
        for r0 in range(0, query.shape[2], block_rows):
            r1 = min(query.shape[2], r0 + block_rows)
            weights = torch.matmul(query[:, :, r0:r1], key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                weights = weights + attention_mask[:, :, r0:r1, : key.shape[-2]]
            weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            out[:, :, r0:r1] = torch.matmul(weights, value)
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register("eager_blocked", eager_blocked)
    AttentionMaskInterface.register("eager_blocked", ALL_MASK_ATTENTION_FUNCTIONS["eager"])


@torch.inference_mode()
def score(model, windows, tile=256):
    lm = model.model.language_model
    text_config = model.config.text_config
    out = np.empty((windows.shape[0], windows.shape[1] - 1), dtype=np.float32)
    for w, row in enumerate(windows):
        ids = torch.from_numpy(row.astype(np.int64))[None].cuda()
        length = ids.shape[1]
        hidden = lm(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=torch.arange(length, device="cuda")[None],
                    past_key_values=DynamicCache(config=text_config), use_cache=True).last_hidden_state[0]
        targets = ids[0, 1:]
        for r0 in range(0, length - 1, tile):
            r1 = min(length - 1, r0 + tile)
            logits = model.lm_head(hidden[r0:r1]).float()
            nll = -torch.log_softmax(logits, dim=-1).gather(1, targets[r0:r1, None])[:, 0]
            out[w, r0:r1] = nll.cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--model", default="/workspace/models/LFM3-Omni-2B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[256, 257, 4096, 32768])
    p.add_argument("--configs", nargs="+", default=["sdpa", "eager"], choices=["sdpa", "eager", "eager_blocked"])
    p.add_argument("--eager-max-length", type=int, default=4096, help="eager materializes L x L attention in FP32")
    p.add_argument("--max-windows", type=int, default=0, help="cap windows per length (0 = all)")
    p.add_argument("--attention-block-rows", type=int, default=2048,
                   help="query rows per block for --configs eager_blocked (exact eager attention, bounded memory)")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="float32: a precision anchor (G4 tier 2b); engines are named hf-CONFIG-fp32, TF32 disabled")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if "eager_blocked" in a.configs:
        register_eager_blocked(a.attention_block_rows)
    if a.dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    suffix = "-fp32" if a.dtype == "float32" else ""

    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    tok = processor.tokenizer
    text = a.text.read_text(encoding="utf-8")
    tokens = tok(text, add_special_tokens=False)["input_ids"]
    bos = tok.convert_tokens_to_ids("<|startoftext|>")
    if bos in tokens:
        raise ValueError("corpus contains the BOS id; windows would not be [BOS] + content")
    summary = {"corpus": str(a.text), "corpus_sha256": sha256(a.text), "corpus_tokens": len(tokens), "bos_id": bos,
               "window_rule": "[BOS] + next L-1 corpus tokens, non-overlapping; score positions 1..L-1",
               "nll_definition": "-log_softmax(lm_head(final hidden).float(), -1)[next token], all 128000 logits",
               "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
               "dtype": a.dtype,
               "precision_note": ("FP32 SDPA runs PyTorch's memory-efficient attention, whose GEMMs use 3xTF32 (FP32-level "
                                  "accuracy) regardless of the TF32 flags (mem_eff_attention/gemm_kernel_utils.h); "
                                  "eager_blocked uses plain FP32 matmuls with TF32 disabled")
                                 if a.dtype == "float32" else None,
               "attention_block_rows": a.attention_block_rows if "eager_blocked" in a.configs else None,
               "precision": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                              "float32_matmul_precision": torch.get_float32_matmul_precision()},
               "windows": {}, "results": []}
    windows = {}
    for length in a.lengths:
        w = build_windows(tokens, bos, length, a.max_windows)
        path = a.output / f"windows-{length}.i32"
        w.tofile(path)
        windows[length] = w
        summary["windows"][str(length)] = {"count": int(w.shape[0]), "file": path.name, "sha256": sha256(path)}

    for config in a.configs:
        model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="cuda",
                                                         trust_remote_code=True, attn_implementation=config).eval()
        actual = model.model.language_model.config._attn_implementation
        if actual != config:
            raise RuntimeError(f"requested attn_implementation={config} but the language model runs {actual}")
        for length in a.lengths:
            if config == "eager" and length > a.eager_max_length:
                summary["results"].append({"engine": f"hf-{config}{suffix}", "length": length, "skipped": "length above eager limit"})
                continue
            t0 = time.perf_counter()
            nll = score(model, windows[length])
            seconds = time.perf_counter() - t0
            if not np.isfinite(nll).all():
                raise RuntimeError(f"non-finite NLL for hf-{config}{suffix} at length {length}")
            path = a.output / f"nll-hf-{config}{suffix}-{length}.f32"
            nll.tofile(path)
            write_meta(a.output / f"nll-hf-{config}{suffix}-{length}.meta.json", engine=f"hf-{config}{suffix}", length=length,
                       windows_file=f"windows-{length}.i32", windows_sha256=summary["windows"][str(length)]["sha256"],
                       nll_sha256=sha256(path), attn_implementation=actual, dtype=a.dtype,
                       versions=summary["versions"])
            mean = float(nll.astype(np.float64).mean())
            summary["results"].append({"engine": f"hf-{config}{suffix}", "length": length, "windows": int(nll.shape[0]),
                                       "scored_tokens": int(nll.size), "mean_nll": mean, "perplexity": math.exp(mean),
                                       "seconds": seconds, "file": path.name, "sha256": sha256(path)})
            print(json.dumps(summary["results"][-1]), flush=True)
        del model
        torch.cuda.empty_cache()
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
