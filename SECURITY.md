# Security

## Scope

`liquid-lfm3-server` is designed for a single trusted machine:

- It listens on **loopback only** and every request needs a **bearer token** (by default a random token in an owned,
  mode-0600 file at `~/.config/liquid-lfm3/token`).
- Request bodies are bounded (4 MiB), parsed strictly (duplicate keys, non-finite numbers, invalid UTF-8 and unknown
  fields are refused), and every engine job has a deadline.
- Audio inputs are decoded natively from the request body; the decoders are fuzzed and sanitizer-checked
  (`validation/audio-assurance-v1`). Image input to the server is in progress: it will take encoded bytes only, never
  file paths, through the decoders hardened in `validation/images-v1`.

Exposing the server beyond loopback (for example through a reverse proxy) is outside what has been validated.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's **Report a vulnerability** (Security → Advisories)
on this repository rather than in a public issue. Include the request or input that triggers it, the binary's sha256
and what happens. You can expect an acknowledgement within a week.
