#!/usr/bin/env python3
"""#2 step 4 report: C1 (intelligibility), C2 (duration and EOS), C4 (interleaved: speech against its own text),
native against HF over the fixed prompt set and its seeds (validation/speech-e2e-v1).

C1 reading of the bar ("native WER not worse than HF's beyond seed-to-seed variation"), per case: a case is FLAGGED
iff native's mean WER over the seeds exceeds HF's worst seed on that case. A flagged case escalates to more seeds
(validation/speech-e2e-v1/prompts-long.json) and a two-sided Mann-Whitney test with a bootstrap CI of the mean
difference. The pooled figure (word errors over all cases, per seed) is printed for reference only: the first
version's pooled rule (native's best seed <= HF's worst) passed while one case was 0.89 against 0.16.
C2: frames per utterance by case (mean, min, max over seeds), the native/HF ratio of means, and how each utterance
ended: by the model's own EOS (slots under the cap) or at max_new_tokens.
Usage: lfm3_speech_report.py PROMPTS.json NATIVE_ASR.json HF_ASR.json NATIVE_CASES HF_CASES OUT.json
"""
import json
import statistics as st
import sys
from pathlib import Path


def load(asr, cases_dir):
    out = {}
    for r in json.load(open(asr))["results"]:
        s = json.load(open(Path(cases_dir) / r["case"] / "summary.json"))
        out[r["case"]] = dict(r, **{k: s.get(k) for k in ("frames", "slots", "generate_seconds", "real_time_factor", "seed")})
    return out


def main(prompts, nat_asr, hf_asr, nat_dir, hf_dir, out_json):
    spec = json.load(open(prompts))
    eng = {"native": load(nat_asr, nat_dir), "hf": load(hf_asr, hf_dir)}
    seeds, rep = spec["seeds"], dict(cases={}, c1={}, c2={})
    print("| case | mode | native WER (mean over seeds) | HF WER (mean [worst]) | C1 | native frames mean [min-max] | HF frames | ratio | ended by EOS (native, HF) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in spec["cases"]:
        row = {}
        for e, rs in eng.items():
            runs = [rs[f"{c['case']}-s{s}"] for s in seeds]
            fr = [r["frames"] for r in runs]
            row[e] = dict(wer=st.mean(r["wer"] for r in runs), wer_max=max(r["wer"] for r in runs),
                          frames_mean=st.mean(fr), frames_min=min(fr), frames_max=max(fr),
                          eos=sum(r["slots"] < c["max_new_tokens"] for r in runs), rtf=st.mean(r["real_time_factor"] for r in runs))
        row["frames_ratio"] = row["native"]["frames_mean"] / row["hf"]["frames_mean"] if row["hf"]["frames_mean"] else None
        row["c1_flag"] = row["native"]["wer"] > row["hf"]["wer_max"]
        rep["cases"][c["case"]] = row
        n, h = row["native"], row["hf"]
        print(f"| {c['case']} | {c['mode']} | {n['wer']:.3f} | {h['wer']:.3f} [{h['wer_max']:.3f}] | "
              f"{'FLAG' if row['c1_flag'] else 'ok'} | {n['frames_mean']:.0f} [{n['frames_min']}-{n['frames_max']}] | "
              f"{h['frames_mean']:.0f} [{h['frames_min']}-{h['frames_max']}] | {row['frames_ratio']:.2f} | {n['eos']}/{len(seeds)}, {h['eos']}/{len(seeds)} |")
    for e, rs in eng.items():
        per_seed = {}
        for s in seeds:
            runs = [rs[f"{c['case']}-s{s}"] for c in spec["cases"]]
            per_seed[s] = sum(r["word_errors"] for r in runs) / sum(r["reference_words"] for r in runs)
        rep["c1"][e] = dict(per_seed_pooled_wer=per_seed, mean=st.mean(per_seed.values()))
    rep["c1"]["flagged"] = [k for k, v in rep["cases"].items() if v["c1_flag"]]
    rep["c2"]["frames_ratio_by_case"] = {k: v["frames_ratio"] for k, v in rep["cases"].items()}
    rep["c2"]["mean_frames_ratio"] = st.mean(v["frames_ratio"] for v in rep["cases"].values())
    rep["rtf"] = {e: st.mean(r["real_time_factor"] for r in rs.values()) for e, rs in eng.items()}
    print(f"\nC1 pooled WER per seed: native {rep['c1']['native']['per_seed_pooled_wer']}, HF {rep['c1']['hf']['per_seed_pooled_wer']}")
    print(f"C1 per case (native mean WER > HF's worst seed): flagged {rep['c1']['flagged'] or 'none'}")
    print(f"C2 mean frames ratio native/HF over cases: {rep['c2']['mean_frames_ratio']:.3f}")
    print(f"real-time factor (audio seconds per generation second): native {rep['rtf']['native']:.2f}, HF {rep['rtf']['hf']:.2f}")
    Path(out_json).write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    main(*sys.argv[1:])
