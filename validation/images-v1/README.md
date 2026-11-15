# #4 images in: image understanding on LFM3-Omni-2B, natively

**Draft rules, decisions made (below); set before any native run.**

Tripp: "the main model to be able to perceive txt, audio, images, and video". Audio is done (audio-in-v1,
audio-assurance-v1, text-wait-v1). This round brings the vision encoder and projector into the native engine and
checks them against HF on still images. Video, which reuses the encoder on frame pairs, is the next round.

## What it will be
- **Front end** (host): image decoding, `smart_resize` (pixel counts 65,536 to 2,097,152, multiples of 32, Python's
  round-half-to-even), the resize, rescale and normalize, and patching into 1536-wide rows grouped by 2x2 merge block.
- **Encoder** (`lfm/vision_encoder.cuh`): the patchifier, the embedding RMSNorm, 2-D axial RoPE, 14 layers (8 gated
  depthwise-conv, 6 bidirectional GQA attention, per image), and the projector (LayerNorm, 2x2 merge, Linear, exact
  GELU, Linear). FP32 activations over the stored BF16 weights from the start (`lfm/fp32_ops.cuh`), the convention
  audio-in-v1 settled on; the output is rounded to BF16, the language model's input.
- **Prompt:** the encoded rows replace the `<|image|>` placeholders at prefill, through the injection path audio input
  already uses. The native renderer expands each image part to `<|vision_start|>` + N x `<|image|>` +
  `<|vision_end|>`, with N = H'W'/1024.
- **Driver and server:** `lfm3-speech` takes image parts (paths); `POST /speech` takes
  `{"type": "image", "image": base64}` parts. Every output mode works with image input.

## Stage 0, done before these rules (2026-11-13)
`tools/dump_lfm3_vision_reference.py` on the workstation CPU, transformers 5.20.2, torch 2.10.0, torchvision 0.25.0:
11 cases (`build/vision-ref/hf-5.20.2/`, 4.1 GB), FP32 and BF16, every stage saved. HF's FP32 replies are coherent
and grounded on 5 of 6 generated cases; the sixth ("small", the pagoda at 200x133, 70 tokens) is a confident
hallucination ("the Jade Pagoda in Nanjing... a UNESCO World Heritage Site"). It is the model, and the port must
reproduce it, not fix it.

**Finding that shapes rule 2:** HF's BF16 encoder drifts from FP32 on a few tokens of the larger images. On the
pagoda (1,040 patches), per-token cosine to FP32 stays above 0.99 through layer 8, then falls; after layer 13, 18
tokens are below 0.99 and the lowest is 0.51. Those tokens are the ones the last layer leaves quiet (norm about 50
against a median of 755, with values up to 349): BF16's absolute error on the loud tokens, mixed in by attention, is
large next to them. Small images stay above 0.97. A BF16 port would inherit this; FP32 activations should not.

## Decisions (Tripp, 2026-11-14)
Asked which choices give the model the truest experience: "Proceed accordingly, my friend."
1. **Image decoding: vendored libjpeg-turbo**, the release PIL bundles on Thor, with PIL's default settings (islow
   DCT, fancy upsampling), so JPEG pixels are exactly the ones HF gives the model, not 1-2 levels off as `stb_image`
   would make them. PNG is lossless: `stb_image` decodes it exactly (8-bit RGB, RGBA, grayscale and palette; PIL applies
   no gamma). Versions found: Thor's container has Pillow 13.0.0 with libjpeg-turbo 3.2.0; the workstation's dump
   environment has Pillow 12.1.0 with libjpeg-turbo 3.1.3. The reference is Thor's (where HF's benchmark run is
   made); the dumped images are PNG, so stage 0 is unaffected, and the two decoders are compared on the JPEG cases
   and recorded before the native run.
2. **The understanding benchmarks: both.** MMStar, all 1,500 multiple-choice items (understanding: chosen by its
   authors so that the image is needed; scored by the answer letter), and POPE, its 3 standard splits of 3,000 yes/no
   items (random, popular, adversarial: does the model report only what is there; the "small" case's hallucination is
   that failure). Both are on the model card.

## Inputs fetched (2026-11-14)
- **libjpeg-turbo 3.2.0** source (`libjpeg-turbo-3.2.0.tar.gz`, sha256 `6f30092c...f09878e`, equal to GitHub's
  release digest; its GPG signature is good, from the project's signing key `0338 C8D8 D9FD A62C F9C4 21BD 7EC2 DBB6
  F4DB F434`). Thor's Pillow 13.0.0 bundles exactly this release, with the 6.2 API (`libjpeg.so.62`).
- **MMStar** (`Lin-Chen/MMStar` at `bc98d668`, `mmstar.parquet`, sha256 `29afd74b...`): 1,500 items, all JPEG
  (1,458 RGB, 42 grayscale). **POPE** (`lmms-lab/POPE` at `4db12766`, `data/test-0000{0,1,2}-of-00003.parquet`):
  adversarial, popular and random, 3,000 each, 1,500 yes and 1,500 no; all JPEG, about 2% grayscale in a 300-image
  sample of each split. Checksums: `dataset-checksums-vision.sha256` in the outer repo (paths under its `data/`).

## Groundwork (2026-11-14, before any native code)
- **The benchmarks as files.** `tools/extract_vision_benchmarks.py` writes each image's stored bytes (no decode, no
  re-encode), named by its sha256, and a manifest of 10,500 rows over 1,930 unique images (POPE's 9,000 questions
  share 500 COCO images). Manifest sha256 `6257d1e1...`. HF and the native engine read the same files; Thor needs no
  pyarrow.
- **HF on Thor.** transformers 5.20.2 in its own venv (`/workspace/venvs/hf-5.20.2`, layered on the container's
  torch 2.10.0+cu130, torchvision 0.25.0, Pillow 13.0.0); the container's own transformers (5.22.1) is untouched. The
  model's short convolution (`lfm3_conv.py`) runs a Triton kernel on CUDA, and Triton 3.6's bundled ptxas (CUDA 12.9)
  does not know Thor's sm_110a. Pointed at CUDA 13.0.88's ptxas (`TRITON_PTXAS_BLACKWELL_PATH`, the binary copied
  from the native build image, sha256 `a1941a04...`), so HF runs its default path rather than the PyTorch fallback.
- **Rule 1's cross-machine check: identical.** `tools/hash_lfm3_image_front.py` hashes, per image, the RGB pixels
  after HF's `load_image`, the grid, the token count and every tensor the processor returns. Thor (aarch64, Pillow
  13.0.0 with libjpeg-turbo 3.2.0, torch cu130) against the workstation (x86, Pillow 12.1.0 with libjpeg-turbo
  3.1.3, torch cu128): 0 differences over 2,445 images (the 1,930 benchmark images, the 10 dumped case images, 480
  distinct sizes of the 500-size sweep, and the first variant set) and over the 47 variants
  (`tools/make_image_variants.py`). Refusals agree too: 63 sweep sizes over the aspect limit (HF's `ValueError`)
  and the truncated JPEG. So HF's front end has no machine-dependent pixels for the native engine to chase.
- **What HF's loader does, read from Pillow 13.0.0's source** (the native front end must do the same):
  - JPEG: libjpeg-turbo's defaults (islow DCT, fancy upsampling); Pillow sets neither unless in draft mode, which
    HF never uses. 1 component is L, 3 are RGB, 4 are always read as Adobe-inverted CMYK ("CMYK;I": every byte
    inverted), whatever the markers say.
  - Orientation (`ImageOps.exif_transpose`, `Image.getexif`): EXIF's Orientation tag (several EXIF APP1 segments are
    concatenated; PNG's `eXIf` chunk or a "Raw profile type exif" text chunk); only when EXIF has none, XMP's
    `tiff:Orientation` (attribute or element form). Values 2-8 transpose; anything else does nothing.
  - To RGB (`Convert.c`): L and 1-bit replicate (1-bit as 0 or 255); LA and RGBA drop alpha (not composited); P
    through its palette; CMYK: `nk = 255 - K`, then `nk - (C*nk + 128 + ((C*nk + 128) >> 8)) >> 8` per channel,
    clipped. 16-bit RGB and RGBA keep the high byte; 16-bit LA is opened as RGBA.
  - **A quirk that matters: 16-bit grayscale PNG.** Pillow opens it as `I;16`, and `I16_RGB` turns every value over
    255 into 255. A 16-bit grayscale photograph reaches the model almost white (the variant's mean is 254.8 of 255).
    Matching HF here means showing the model a blank image; this is Tripp's call (below).
- **The understanding benchmarks' scoring, fixed before any native run.** The first HF run (stopped at 1,156 items,
  `build/images-v1/hf-bench/stopped-run1-16tok.jsonl`) gave MMStar 16 new tokens and took the first standalone
  letter: 28% of replies were prose cut off before naming an option, and a reply starting "A man ..." scored as A.
  Then 64 tokens (a full run, `64tok-run1.jsonl`): POPE F1 89.00 over all 9,000 (adversarial 86.63, popular
  88.72, random 91.81; yes-ratios 51.7%, 49.3%, 46.0%; every reply parsed), against the model card's 89.34, so the
  HF reference on Thor is sound; but MMStar 42.20% with 400 replies unanswered, 342 of them cut off at 64 tokens (the
  model reasons first; 346 questions ask for the letter "at the end"). Its second run was stopped. Now: 512 tokens
  for MMStar, and a parser modeled on VLMEvalKit's (without its LLM judge): a leading letter (followed by `)`, `.`,
  `:` or the end), else a last line that is only a letter, else "answer is X" / "Answer: X", else exactly one
  standalone option letter ("A" ignored in replies over 3 words), else exactly one option's text, else no answer.
  The power cycle of 2026-11-14 21:10 cut the 64-token second run mid-line; the partial line was dropped and the
  run resumed before it was stopped for this change. Options are read from both of MMStar's layouts ("Options: A: ...,"
  and "(A) ..." lines). The same requests and scoring go to both engines, so rule 3's comparison is unaffected by
  this choice; it only makes the accuracy mean what it says. The model card reports MMStar 55.67 (Liquid AI's own
  harness, not reproduced here).

- **The native decoder** (`native/src/image_decode.cpp`, `lfm/image_decode.hpp`; check `lfm3-image-decode-check`,
  built by `tools/build_image_decode_check.sh`): vendored libjpeg-turbo 3.2.0 and stb_image (PNG only), with the
  provenance and two backported upstream fixes recorded in `native/third_party/README.md`. JPEG is read through a
  suspending source as Pillow's is (a file that ends before its last row is refused; one missing only its EOI
  marker is accepted, as Pillow accepts it), not libjpeg's memory source (which pads with a fake EOI).
  - **Identical to HF on every image: 1,987 of 1,987 on each machine.** Workstation (x86, libjpeg-turbo's C path,
    no NASM) against the workstation's HF, and Thor (aarch64, NEON SIMD) against Thor's HF: the 1,930 benchmark
    images, the 10 dumped case images and the 47 variants, each RGB byte and size identical, every refusal matched.
    This includes CMYK, grayscale, progressive, restart markers, every EXIF orientation (both byte orders), XMP-only
    orientation in both forms, EXIF taking precedence over XMP, and every PNG kind (16-bit gray matched to HF's
    saturation for now: the open question below).
  - **Red-team, hostile input** (`tools/make_image_mutants.py`, 4,000 files from the variants by seeded mutation:
    bit flips, overwrites, truncation, insertion, deletion, corrupted segment and chunk lengths, swaps, trailing
    garbage, zeroed runs), the native check built with ASan and UBSan on Thor: **0 sanitizer reports, no crash**.
    Outcome against HF (Thor), file by file: 3,932 agree (2,121 both refuse, 1,811 identical pixels); 68 do not:
    - 7 JPEG: the pixels decode identically but the orientation differs. Pillow's marker loop (in Python, before
      libjpeg runs) recovers from damage before the APP1 segment differently from mine (it treats `FF FF` as padding,
      `FF 00` as escaped data, refuses an unknown marker code, and reads past EOI), so it found intact EXIF that mine
      missed. Pillow also refuses some files in Python before libjpeg runs (a short SOF, DQT or Adobe segment, the
      ICC fix-up, a truncated segment read).
    - 61 PNG: Pillow's own PNG reader (its chunk parser, CRC checks and zlib) and stb tolerate different damage: 36
      Pillow refuses and stb decodes ("broken data stream", "cannot identify"); 25 stb refuses and Pillow decodes (21
      "IDAT size limit", 3 "out of data", 1 unknown critical chunk, which Pillow ignores).
  - Not a rule as set (rule 1 covers valid images, rule 5 that refusals are clean); recorded as found. Closing it
    means porting Pillow's JPEG marker parse exactly, and for PNG replacing stb with a port of Pillow's PNG reader.

## Open question for Tripp
- **16-bit grayscale PNGs.** Match HF (the model sees almost white) or decode them correctly (the high byte, as
  16-bit RGB already gets) and record the difference from HF as a deliberate one. Benchmarks are unaffected (all
  JPEG); the native front end refuses nothing it can decode either way.
- **Damaged files.** 98.3% of 4,000 hostile files already get HF's outcome; the 68 that do not are damaged files
  where Pillow and the native decoder disagree on whether (or, for 7 JPEGs, which way up) to show them. Port Pillow's
  JPEG marker parse and PNG reader for full parity, or accept and record the difference (valid images are identical
  either way).

## Rules, set before any native run (draft)
1. **Decoding and preprocessing match HF exactly.** Decoded pixels are bit-identical to PIL's (Thor's) on every JPEG
   and PNG case: the dumped images, the MMStar and POPE images, and progressive, 4:2:0, 4:2:2, 4:4:4, grayscale, CMYK
   and EXIF-rotated JPEGs (all 8 orientations). HF's loader, checked (transformers 5.20.2, `image_utils.py:512-513`
   `load_image`), applies `PIL.ImageOps.exif_transpose` and then `convert("RGB")`; the native engine does the same,
   including PIL's grayscale, palette, RGBA (alpha dropped, not composited) and CMYK conversions to RGB, and the HF
   benchmark run loads every image through `load_image`. From those pixels: the grid (t, h, w) and the token count equal
   HF's on every case, and `pixel_values` are bit-identical to HF's (torchvision's uint8 antialiased bicubic is
   fixed-point integer arithmetic, so exact is achievable). Cases: the 11 dumped images, plus a sweep of 500 sizes
   (every width and height from 1 to 4,096 hit at random, rounding ties at x.5 of 32, both sides of the minimum and
   maximum pixel counts, aspect ratios up to 200). Before the native run, HF on Thor (aarch64) is checked against HF on
   the workstation (x86): if torchvision's two CPU paths disagree, Thor's HF is the reference and the disagreement is
   recorded.
2. **The encoder is at least as accurate as HF's.** On every case (the 11 dumped and 20 held-out MMStar images), the
   native projected features' relative L2 error to HF's FP32 features (same pixel_values) is at most HF BF16's
   (ratio <= 1.0), and the native per-token minimum cosine to FP32 is at least HF BF16's.
3. **Understanding is as good as HF's.** Greedy, HF BF16 on Thor's GPU (transformers 5.20.2, SDPA) against native,
   the same requests: over all 1,500 MMStar items and over all 9,000 POPE items, the accuracy difference (native minus
   HF, in points) has a paired-bootstrap 95% lower bound of at least -0.5. POPE's F1 and yes-ratio are reported per
   split.
4. **Coherency.** I read the replies to the 11 dumped cases, 30 MMStar items, and every item where the two engines
   answer differently; then images described in speech (interleaved), which Tripp listens to.
5. **Refusals are clean.** Aspect ratio over 200, an undecodable or truncated image, an unsupported format, zero-size
   or oversized images, an image/placeholder count mismatch, a span without its markers, image with audio-only modes
   that cannot take it, images in an assistant turn, and over capacity: each refused before any state changes, with
   its own message.
6. **Nothing else changes.** The text gate (control sha256 `7ea05068...`), lfm3-speech's 36-output regression, the
   audio-in replies and the server's TTS WAVs are all bit-identical to the current validated builds; host suites pass
   on plain, ASan+UBSan and TSan.
7. **Sanitizers.** memcheck, racecheck, synccheck and initcheck on an encoder check covering a small image, two images
   in one prompt and a maximum-size image: 0 errors, outputs identical to the plain run.

Speed is reported (encode milliseconds per image by size, prefill, end to end), not gated.

Video, multi-turn image caching and image tokens' share of the KV budget beyond 32k are out of scope.
