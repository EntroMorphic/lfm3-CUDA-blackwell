"""Bounded local resident-worker transport, independent of the GPU gate tool."""
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import threading
import time


MAX_FRAME = 4 * 1024 * 1024


def strict_json(raw):
    text = raw.decode("utf-8", errors="strict")
    quoted = escaped = False
    depth = 0
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 64:
                raise ValueError("JSON nesting exceeds limit")
        elif char in "]}":
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        raise ValueError("nonfinite JSON number")

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number")
        return result

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=finite, parse_float=number)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds limit") from exc
    if not isinstance(result, dict):
        raise ValueError("expected JSON object")
    return result


class WorkerFailure(RuntimeError):
    """Pipe/process/protocol failure: no cache state remains trustworthy."""


class WorkerClient:
    def __init__(self, binary, checkpoint, *, timeout=300, env=None):
        if not math.isfinite(timeout) or not 0 < timeout <= 86400:
            raise ValueError("invalid worker timeout")
        self.timeout = timeout
        self.closed = False
        self.lock = threading.Lock()
        self.buffer = bytearray()
        self.sequence = 0
        # No checkpoint writes, shell, inherited stdout or unbounded stderr pipe.
        self.process = None
        try:
            # Defer interruption until Popen's child is recorded; otherwise a
            # signal between spawn and assignment can orphan its separate group.
            blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
            try:
                self.process = subprocess.Popen(
                    # exec preserves blocked signals. A tiny exec-only launcher
                    # restores the original mask in the child without unsafe
                    # preexec_fn use in this multithreaded serving process.
                    [sys.executable, "-c",
                     "import os,signal,sys;signal.pthread_sigmask(signal.SIG_SETMASK,"
                     + repr(sorted(int(sig) for sig in blocked))
                     + ");os.execv(sys.argv[1],sys.argv[1:])",
                     str(binary), "--experimental", str(checkpoint)], env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    restore_signals=True,
                )
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            ready = self._read(time.monotonic() + timeout)
            if (ready.get("ready") is not True or type(ready.get("protocol")) is not int
                    or ready["protocol"] != 2 or ready.get("gpu_batching") is not False
                    or type(ready.get("max_sessions")) is not int or ready["max_sessions"] != 2
                    or type(ready.get("aggregate_capacity")) is not int
                    or ready["aggregate_capacity"] != 32768):
                raise WorkerFailure("incompatible worker handshake")
        except BaseException:
            self.close()
            raise

    def _read(self, deadline):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                end = self.buffer.find(b"\n")
                if end >= 0:
                    if end > MAX_FRAME:
                        raise WorkerFailure("worker frame exceeds limit")
                    raw = bytes(self.buffer[:end])
                    del self.buffer[:end + 1]
                    return strict_json(raw)
                if len(self.buffer) > MAX_FRAME:
                    raise WorkerFailure("worker frame exceeds limit")
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("worker response deadline")
                try:
                    block = os.read(self.process.stdout.fileno(), min(65536, MAX_FRAME + 1 - len(self.buffer)))
                except BlockingIOError:
                    continue
                if not block:
                    raise WorkerFailure("worker exited before reply")
                self.buffer.extend(block)

    def call(self, request):
        if "rpc_id" in request:
            raise ValueError("rpc_id is owned by the transport")
        # Validate size before admission without consuming an RPC sequence.
        raw = json.dumps(request, allow_nan=False, separators=(",", ":"))
        if len(raw.encode("utf-8")) + 32 > MAX_FRAME:
            raise ValueError("request frame exceeds limit")
        if not self.lock.acquire(blocking=False):
            raise WorkerFailure("concurrent worker pipe access is forbidden")
        try:
            if self.closed:
                raise WorkerFailure("worker closed")
            if self.sequence >= 2147483647:
                raise WorkerFailure("RPC sequence exhausted; restart required")
            self.sequence += 1
            sequence = self.sequence
            payload = (json.dumps(dict(request, rpc_id=sequence), allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            if self.buffer:
                raise WorkerFailure("unsolicited trailing worker data")
            deadline = time.monotonic() + self.timeout
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                offset = 0
                while offset < len(payload):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError("worker write deadline")
                    try:
                        written = os.write(self.process.stdin.fileno(), payload[offset:])
                    except BlockingIOError:
                        continue
                    if written <= 0:
                        raise WorkerFailure("worker pipe closed")
                    offset += written
            result = self._read(deadline)
            if (type(result.get("ok")) is not bool or type(result.get("rpc_id")) is not int
                    or result["rpc_id"] != sequence or self.buffer):
                raise WorkerFailure("invalid worker reply")
            del result["rpc_id"]
            return result
        except (OSError, ValueError, TimeoutError, WorkerFailure) as exc:
            self.close()
            raise WorkerFailure("worker RPC failed; all caches invalid") from exc
        finally:
            self.lock.release()

    def close(self):
        if getattr(self, "closed", False):
            return
        self.closed = True
        if getattr(self, "process", None) is None:
            return
        # Kill the owned group even if the direct child already exited: helpers
        # must not survive an EOF/protocol failure in their parent.
        try:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=2)
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            # Even an uninterruptible child exceeding the final reap deadline
            # must not leak the parent's pipe descriptors. Do not hide that
            # timeout or falsely report the process as reaped.
            self.process.stdin.close()
            self.process.stdout.close()
