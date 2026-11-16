# Contributing

Thank you for your interest. liquid-CUDA is research software with a strict evidence discipline: a change is accepted
when the evidence says it is right, not when it looks right. This page is how that works in practice.

## Ground rules

1. **Rules before data.** A change to the engine's behavior starts with a validation round: a directory under
   `validation/` whose `README.md` states what will be run and what counts as a pass — committed *before* the first
   native run. Rules may be amended only with the amendment recorded and dated, before the data it would affect.
2. **Exact where possible.** Integer and fixed-point paths (tokenization, image decoding and resizing, sampling
   decisions) must be bit-identical to the reference. Floating-point paths must be at least as close to an FP32
   reference as the reference's own BF16 GPU path is.
3. **Nothing else changes.** A change must leave every earlier round's outputs byte-identical unless the round says
   otherwise. The regression scripts under `tools/thor/` re-run them.
4. **Coherency is a gate.** Read (or listen to) real generations end to end after any engine change; mechanical checks
   are not enough.
5. **Record failures.** A failed rule stays in the round's README with what was learned. Claims are worded to the data.
6. **Speed is labeled.** Every number says whether it is the decode loop or end to end, and is measured on an idle
   machine.

## Development

- Host-only tests (no GPU): `PYTHONPATH=tests python3 -m pytest tests audit/test_host_contracts.py -q`
- Native builds and GPU checks run on the target machine in pinned containers; see
  [docs/building.md](docs/building.md).
- Style: C++20 and CUDA in `native/` follow the surrounding code (2-space indent, `lfm::` namespaces, headers in
  `native/include/lfm/`). Python tools use the standard library where they can and run as scripts.
- Comments say *why*, and cite the validation round that decided it (`validation/<round>` rule N).

## Commits

- One logical change per commit, with a message that states the result (for example
  `Concurrent speech: /speech requests batched on the GPU, each byte-identical to its reply alone`).
- Large evidence goes in `validation/<round>/` compressed (`.gz`); raw numerical traces stay out of git.

## Reporting problems

Open an issue with the request, the binary's sha256, the checkpoint revision and what you expected. For security
issues, see [SECURITY.md](SECURITY.md) instead.
