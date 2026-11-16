# Security

## Scope

`liquid-lfm3-server` is designed for a single trusted machine:

- It listens on **loopback only** and every request needs a **bearer token** (by default a random token in an owned,
  mode-0600 file at `~/.config/liquid-lfm3/token`).
- Request bodies are bounded (see below), parsed strictly (duplicate keys, non-finite numbers, invalid UTF-8 and unknown
  fields are refused), and every engine job has a deadline.
- Media inputs are decoded natively from the request body, as encoded bytes only (never file paths). The audio
  decoders are fuzzed and sanitizer-checked (`validation/audio-assurance-v1`); the image decoders are checked against
  thousands of damaged and hostile files, refuse damaged PNGs, and cap a decoded image at 64 megapixels
  (`validation/images-v1`). Request bodies with images may be up to 16 MiB; all others 4 MiB.

Exposing the server beyond loopback (for example through a reverse proxy) is outside what has been validated.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's **Report a vulnerability** (Security → Advisories)
on this repository rather than in a public issue. Include the request or input that triggers it, the binary's sha256
and what happens. You can expect an acknowledgement within a week.
