import json
import sys

M = 2 ** 20
for side in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{side}/soak-4b.readings.jsonl")]
    main = [r for r in rows if r["thread"] == "MainThread"]
    groups, cur = [], []
    for r in main:
        if cur and r["wall"] - cur[-1]["wall"] > 5:
            groups.append(cur)
            cur = []
        cur.append(r)
    if cur:
        groups.append(cur)
    print(f"== {side}: {len(groups)} checkpoints")
    for i, g in enumerate(groups):
        st = lambda r, k: (r.get("cgroup_stat") or {}).get(k, 0) / M
        med = lambda k: sorted(x[k] for x in g)[len(g) // 2] / M
        print(f"cp{i + 1}: host_used med {med('host_used'):.0f}  free med {med('free'):.0f}  buff/cache {g[0]['buff_cache'] / M:.0f}  "
              f"AnonPages {g[0]['AnonPages'] / M:.0f}  cgroup current {g[0]['cgroup_current'] / M:.0f} anon {st(g[0], 'anon'):.0f} "
              f"file {st(g[0], 'file'):.0f} pgscan {st(g[0], 'pgscan') * M:.0f} pgsteal {st(g[0], 'pgsteal') * M:.0f}")
        if max(x["host_used"] for x in g) - min(x["host_used"] for x in g) > 100 * M:
            for r in g:
                print(f"     host_used {r['host_used'] / M:.0f} free {r['free'] / M:.0f} buff/cache {r['buff_cache'] / M:.0f} "
                      f"AnonPages {r['AnonPages'] / M:.0f} Shmem {r['Shmem'] / M:.0f} cgroup current {r['cgroup_current'] / M:.0f} "
                      f"anon {st(r, 'anon'):.0f} file {st(r, 'file'):.0f}")
