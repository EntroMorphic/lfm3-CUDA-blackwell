# HTTP review remediation record

The initial review identified three P2 defects: startup SIGTERM could orphan the
separately grouped worker, an executing timed-out create could strand aggregate
quota, and contradictory nested sampling metadata was accepted. Fixes install
termination handling before worker creation, synchronize abandonment/completion
and reclaim native allocations on the owner thread, expire idle allocations
(default 300 seconds), and reject all three ambiguous sampling metadata fields.

Real socketpair tests add parser/framing/authentication, absolute header/body
deadlines, incomplete bodies, handler-slot admission, and non-reading response
peers. They do not replace native inference acceptance.

Independent checking found an additional defect in the startup fix: exec inherited
SIGTERM/SIGINT blocking (`SigBlk: 0000000000004002`). The first native HTTP run
reached its fault injection but failed to terminate the worker within ten seconds;
forced cleanup also exceeded its final bounded wait. Its failed summary, output
and traceback are retained in `http-native-remediated-v1/`, not relabeled a pass.
A subsequent process check found no remaining worker or test process.

The transport now restores the original signal mask through an exec-only Python
launcher, without unsafe threaded `preexec_fn`. A host regression verifies the
exact child/parent masks and ordinary SIGTERM exit within one second. The complete
host suite passes 552 tests (`host-http-remediated-v1.log`). The corrected native
HTTP run uses a fresh output directory; its final disposition is recorded separately.

Final cleanup hardening also closes pipes when final reaping exceeds its deadline
and preserves fail-closed503 responses if cleanup itself fails. Two new regressions
bring the full suite to554 passes (`host-http-remediated-v2.log`). The final-source
native gate passed in `http-native-remediated-v3/`, with normal SIGTERM termination,
no remaining worker/test process, and source hashes matched locally/remotely.

No assertion was relaxed to admit the failed native run. GPU batching, native
cancellation, soak acceptance and the integrated sanitizer blockers remain open.
