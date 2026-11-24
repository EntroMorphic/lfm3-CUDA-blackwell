#!/usr/bin/env python3
"""validation/encoder-wrap-v1 rule 2: the machine code of every kernel in a main build against a candidate build.

Usage: compare_sass.py MAIN_BINARY CANDIDATE_BINARY OUT_JSON [CUOBJDUMP] [CUFILT]

`cuobjdump -sass` of each binary is split into its functions (keyed by the code's architecture and the demangled
name). A candidate kernel that gained a trailing `false` template argument (encoder-wrap-v1's WRAP = false) is
matched to main's kernel without it. Each function of main is then: identical (the same instructions, every
encoding word included), different, or missing from the candidate; the candidate's functions that main lacks are
listed as extra. Exit 0 when no function of main differs (missing and extra are reported, not judged).
"""
import json
import re
import subprocess
import sys

def functions(binary, cuobjdump, cufilt):
    text = subprocess.run([cuobjdump, "-sass", binary], check=True, capture_output=True, text=True).stdout
    out, arch, name, body = {}, None, None, []

    def close():
        if name is not None:
            out[(arch, name)] = body

    for line in text.splitlines():
        m = re.match(r"\s*arch = (\S+)", line)
        if m:
            arch = m.group(1)
            continue
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            close()
            name, body = m.group(1), []
            continue
        if name is not None:
            s = line.strip()
            if s and not s.startswith(".") and not s.startswith("---"):
                body.append(re.sub(r"\s+", " ", s))
    close()
    names = sorted({n for _, n in out})
    demangled = subprocess.run([cufilt], input="\n".join(names) + "\n", check=True, capture_output=True,
                               text=True).stdout.splitlines()
    table = dict(zip(names, demangled))
    return {(a, table.get(n, n)): b for (a, n), b in out.items()}


def normalize(name):
    if any(w in name for w in ("wide_kernel<", "thin_kernel<", "gemm_2cta<")):  # cu++filt's names begin "void lfm::..."
        # the last template argument of WRAP = false (cu++filt prints it "(bool)0"; "false" kept for other demanglers):
        # "..., (bool)0>(...)" -> "...>(...)"
        name = re.sub(r", (?:\(bool\)0|false)>\(", ">(", name, count=1)
    return name


def main(main_bin, cand_bin, out_json, cuobjdump="cuobjdump", cufilt="cu++filt"):
    a = functions(main_bin, cuobjdump, cufilt)
    b_raw = functions(cand_bin, cuobjdump, cufilt)
    b = {}
    for (arch, n), body in b_raw.items():
        key = (arch, normalize(n))
        if key in a and (arch, n) not in a:
            b[key] = body
        else:
            b[(arch, n)] = body
    same = [k for k in a if k in b and a[k] == b[k]]
    differ = [k for k in a if k in b and a[k] != b[k]]
    missing = [k for k in a if k not in b]
    extra = [k for k in b if k not in a]
    gemm = [k for k in same if any(w in k[1] for w in ("wide_kernel<", "thin_kernel<", "gemm_2cta<"))]
    r = dict(main=main_bin, candidate=cand_bin, functions=len(a), identical=len(same), gemm_kernels_identical=len(gemm),
             differ=[f"{x} {n}" for x, n in differ], missing=[f"{x} {n}" for x, n in missing],
             extra=[f"{x} {n}" for x, n in extra])
    open(out_json, "w").write(json.dumps(r, indent=1) + "\n")
    print(f"{cand_bin}: {len(same)} of {len(a)} functions identical ({len(gemm)} of them wide/thin/gemm_2cta); "
          f"{len(differ)} differ, {len(missing)} missing, {len(extra)} extra")
    for k in differ:
        print("  DIFFER", k[0], k[1][:200])
    for k in missing:
        print("  missing", k[0], k[1][:200])
    for k in extra:
        print("  extra", k[0], k[1][:200])
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5, 6):
        sys.exit(__doc__)
    main(*sys.argv[1:])
