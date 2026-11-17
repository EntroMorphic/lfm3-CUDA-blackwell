#!/usr/bin/env python3
"""validation/image-cache-v1, red-team fixes C3-C5 on Thor: distinct 4K photographs, made deterministically from one
(the stage-0 "cherry", 3840 x 2160 RGB PNG). Image INDEX is the source with 16 bytes of one scanline's filtered data
changed (row (INDEX * 71) % height, bytes 1..16 XOR (INDEX & 255), INDEX >= 1), recompressed with zlib level 6 into a
PNG of the source's own chunks (IHDR, iCCP and every other ancillary chunk before the image data kept as they are,
IDAT in 64 KiB pieces, IEND). The pixels differ from the source's (and each other's) from that row down the changed
columns, so each image is a distinct cache key, a distinct decode and a distinct encode; the size stays the source's
(level 6 gives 9,881,565 bytes against cherry's 9,833,788). Standard library only. zlib's output is deterministic for
one zlib version: the manifest records every file's sha256 where the images were made.
The manifest also gives each image's cache costs as the server counts them (validation/image-cache-v1 C3): the key is
the image part's base64 text; the engine keeps key + tokens x 2048 x 2 bytes (BF16 encoder rows), the front end the
key (and a 4-byte count). Tokens follow HF's smart_resize (factor 32, 65,536..2,097,152 pixels): 1,980 at 3840 x 2160.
Usage: make_lfm3_c3_images.py SOURCE_PNG OUT_DIR FIRST LAST   (writes OUT_DIR/c3-FIRST.png .. c3-LAST.png and
       OUT_DIR/manifest.json; indices zero-padded to 2 digits)
"""
import hashlib
import json
import math
import struct
import sys
import zlib
from pathlib import Path

SIGNATURE = b"\x89PNG\r\n\x1a\n"
HIDDEN, ROW_BYTES = 2048, 2  # the engine's encoder rows: tokens x hidden, BF16
FACTOR, MIN_PIXELS, MAX_PIXELS = 32, 65536, 2097152
PIECE = 65536


def chunks(data):
    if data[:8] != SIGNATURE:
        raise ValueError("not a PNG")
    at, out = 8, []
    while at < len(data):
        n, kind = struct.unpack(">I4s", data[at:at + 8])
        out.append((kind, data[at + 8:at + 8 + n]))
        at += 12 + n
    return out


def chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def tokens(width, height):
    """HF's smart_resize (Qwen2-VL's, factor 32), then one token per 32 x 32 block (lfm::image::tokens)."""
    h, w = max(FACTOR, round(height / FACTOR) * FACTOR), max(FACTOR, round(width / FACTOR) * FACTOR)
    if h * w > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        h, w = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR), max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h * w < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h, w = math.ceil(height * beta / FACTOR) * FACTOR, math.ceil(width * beta / FACTOR) * FACTOR
    return (h // FACTOR) * (w // FACTOR)


def costs(png_bytes, width, height):
    key = 4 * ((len(png_bytes) + 2) // 3)  # the base64 text's length
    t = tokens(width, height)
    rows = t * HIDDEN * ROW_BYTES
    return dict(key_bytes=key, tokens=t, row_bytes=rows, engine_cost=key + rows, front_end_cost=key + 4)


def make(source, index):
    """The source PNG's bytes with image INDEX's change (INDEX >= 1)."""
    if index < 1:
        raise ValueError("index must be >= 1 (0 is the source itself)")
    parts = chunks(source)
    width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", parts[0][1])
    if parts[0][0] != b"IHDR" or (depth, color, interlace) != (8, 2, 0):
        raise ValueError("expected an 8-bit RGB non-interlaced PNG")
    raw = bytearray(zlib.decompress(b"".join(body for kind, body in parts if kind == b"IDAT")))
    stride = 1 + width * 3
    if len(raw) != stride * height:
        raise ValueError("unexpected image data length")
    row = (index * 71) % height
    for i in range(1, 17):
        raw[row * stride + i] ^= index & 255 or 1
    data = zlib.compress(bytes(raw), 6)
    out = [SIGNATURE]
    wrote_data = False
    for kind, body in parts:
        if kind == b"IDAT":
            if not wrote_data:
                out += [chunk(b"IDAT", data[i:i + PIECE]) for i in range(0, len(data), PIECE)]
                wrote_data = True
        else:
            out.append(chunk(kind, body))
    return b"".join(out), dict(width=width, height=height, changed_row=row)


def main(source_png, out_dir, first, last):
    first, last = int(first), int(last)
    source = Path(source_png).read_bytes()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w, h = struct.unpack(">II", chunks(source)[0][1][:8])
    manifest = dict(source=str(source_png), source_sha256=hashlib.sha256(source).hexdigest(), zlib=zlib.ZLIB_RUNTIME_VERSION,
                    source_costs=costs(source, w, h), images={})
    seen = {hashlib.sha256(source).hexdigest()}
    for index in range(first, last + 1):
        png, info = make(source, index)
        digest = hashlib.sha256(png).hexdigest()
        if digest in seen:
            raise RuntimeError(f"image {index} is not distinct")
        seen.add(digest)
        name = f"c3-{index:02d}.png"
        (out / name).write_bytes(png)
        manifest["images"][name] = dict(index=index, bytes=len(png), sha256=digest, **info, **costs(png, info["width"], info["height"]))
        print(name, len(png), digest[:16], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
