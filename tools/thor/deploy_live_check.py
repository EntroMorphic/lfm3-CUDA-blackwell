#!/usr/bin/env python3
"""After a deploy (validation/always-on-v1): /health; one text generation (its session closed); one TTS; a clip's transcript
twice, with the time to first text from the send; against the running always-on service on 127.0.0.1:8080.
Standard library only. Usage (Thor host): deploy_live_check.py TOOLS_DIR CLIP.wav"""
import base64, json, socket, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, sys.argv[1])  # the tools directory, for lfm3_token
from lfm3_token import read_token  # noqa: E402

TOKEN = read_token(None)
URL = "http://127.0.0.1:8080"


def post(path, body):
    r = urllib.request.Request(URL + path, data=json.dumps(body).encode(), method="POST",
                               headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(r, timeout=300) as f:
            return f.status, json.loads(f.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def stream(body):
    payload = json.dumps(body).encode()
    head = (f"POST /speech HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nConnection: close\r\n"
            f"Content-Length: {len(payload)}\r\nAuthorization: Bearer {TOKEN}\r\n\r\n").encode()
    s = socket.create_connection(("127.0.0.1", 8080))
    t_send = time.monotonic(); s.sendall(head + payload)
    buf = b""; first = None; text = ""; done = None; status = None
    while True:
        d = s.recv(65536)
        if not d: break
        buf += d
        if status is None and b"\r\n\r\n" in buf:
            hdr, buf = buf.split(b"\r\n\r\n", 1); status = int(hdr.split()[1])
        if status is None: continue
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"{"): continue
            e = json.loads(line)
            if e.get("event") == "done":
                done = e
                continue
            if "text" in e:
                if first is None: first = time.monotonic() - t_send
                text += e["text"]
    s.close()
    return status, first, text, done


with urllib.request.urlopen(URL + "/health", timeout=10) as f:
    print("health:", f.status, f.read().decode()[:200])
st, sess = post("/sessions", {"capacity": 1024})
st, g = post("/generate", {"session_id": sess["session_id"], "max_new_tokens": 64,
                           "messages": [{"role": "user", "content": "In two sentences, why is the sky blue?"}]})
print("text:", st, json.dumps(g)[:600])
print("close:", post(f"/sessions/{sess['session_id']}/close", {}))
st, first, text, done = stream({"messages": [{"role": "system", "content": "Perform TTS in American English using a female voice."},
                                                    {"role": "user", "content": "The merged server is live on Thor this morning."}],
                                       "mode": "audio", "max_new_tokens": 1024, "seed": 1, "stream": True})
print(f"tts: {st}, {done.get('audio_seconds') if done else None} s of audio, ended by EOS "
      f"{all(c.get('ended_by_eos') for c in done.get('chunks', [])) if done else None}")
clip = base64.b64encode(Path(sys.argv[2]).read_bytes()).decode()
for i in range(2):
    st, first, text, done = stream({"messages": [{"role": "system", "content": "Perform ASR in English."},
                                                        {"role": "user", "content": [{"type": "audio", "audio": clip}]}],
                                           "mode": "text", "max_new_tokens": 400, "seed": 0, "stream": True})
    print(f"audio in #{i + 1}: {st}, first text {first:.4f} s after the send; streamed text equals done's: "
          f"{bool(done) and text == done.get('text')}")
print("transcript:", text)
