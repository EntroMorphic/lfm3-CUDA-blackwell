# Explicit grouped text serving: acceptance ledger

This increment adds native `generate_batch`, a correlated client group RPC and
authenticated HTTP `POST /generate_batch {"requests":[A,B]}`. It is a bounded
fixed-two-lane scheduler with serial prefill and genuine two-row dense decode;
it is **not automatic coalescing** of unrelated `/generate` requests. The original
single endpoint remains serialized and truthful about batch participation.

Each lane has its own lease, sampler/RNG, stop decision, committed transcript,
physical cache and cancellation identity. Group cancellation uses tagged frames
separate from legacy single-RPC cancellation, with one shared pipe/partial-frame
reader. A retired lane is not forwarded again. Pre-dispatch cancellation can
leave only single-lane work, which must report zero paired calls.

## Red-team and host evidence

- Native scheduler/reader host fakes, strict C++17/UBSan and legacy regressions:
  full suite **905 passed**, `validation/host-grouped-serving-v1.log`.
- Independent review reproduced seven acceptance-helper gaps: trace aliasing,
  vacuous references/coverage, impossible counts, boolean counts and invalid cache
  metadata. Original failures: `validation/redteam-group-gates-before.log`.
  All are now mandatory passing regressions. The stochastic native gate also
  requires independent per-step sampler replay and a demonstrated non-greedy
  choice, not merely agreement between two potentially greedy paths.
- Four service failures exposed wrong-session lifecycle acknowledgments and
  impossible paired counts. Original failures:
  `validation/redteam-service-batch-before.log`. Present-but-mismatched lifecycle
  IDs now fail closed; legacy omitted IDs remain accepted. Actual native RPC
  correlation remains mandatory. Group counts must agree with completed work.
- Independent native source review found no new concrete defect. This is not GPU
  proof. The Thor worker compiled as `184f14b7...`; GPU acceptance is pending.

## GPU verification and resource gates

`check_lfm3_batch_worker.py` requires independent serial/group trace identities,
all full-vocabulary BF16 bits, IDs, policies, counts and real pair calls. Its
fixtures cover mixed context/policy, both retirement orders, reuse, pre-mutation
rejection, EOS, observed live lane cancellation, peer continuation, recovery,
stale tags, and aggregate capacities3+32765. Stochastic logit replay is mandatory.

`check_lfm3_http_batch.py` checks authenticated explicit grouping, live targeted
cancellation and recovery, then the same exact32-token output per lane under
fixed warm RSS/FD/device-wide memory limits. Both HTTP policies are deterministic
greedy/top-k1 controls; non-greedy acceptance belongs to the native replay gate.
The planned extended run requires at least60 cycles and600 measured seconds,
with serial-pair timing as a same-workload observation, not a speed promise.
Post-close device free memory must recover against an empty-session warm baseline
within a fixed16MiB allowance. Device-wide CUDA free memory on Thor is not a
per-worker allocation or physical-residency attribution.

Neither these tools nor host tests waive sanitizer gates. Initcheck remains
unresolved; no supported vendor BSP/tool pairing or bug confirmation is claimed.
The checkpoint-free reproduction packet is prepared locally, not published.
Actual GPU results must be appended before this increment is accepted.
