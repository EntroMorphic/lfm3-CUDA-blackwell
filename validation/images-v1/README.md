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
