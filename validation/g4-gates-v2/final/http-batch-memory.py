#!/usr/bin/env python3
"""Extract the exact-memory figures from an http-batch summary: how memory was judged, sample count, and the
worker's engine-buffer bytes with sessions empty, at every soak sample (distinct values), and after close.
Usage: http-batch-memory.py SUMMARY.json   (produces pin/ and golden/ http-batch-memory.json)"""
import json
import sys

j = json.load(open(sys.argv[1]))
print(json.dumps(dict(status=j.get("status"), memory_judged_by=j.get("memory_judged_by"), samples=len(j["samples"]),
                      app_live=sorted({x["gpu"].get("app_live_bytes") for x in j["samples"]}),
                      empty=j["empty_session_gpu_baseline"].get("app_live_bytes"), post_close=j["post_close_gpu"].get("app_live_bytes"))))
