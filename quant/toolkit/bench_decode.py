#!/usr/bin/env python3
"""测 decode 速度: 3 条代表样本 x 3 次, 流式统计 TTFT 和 decode tok/s"""
import json
import sys
import time

import requests

MODEL = sys.argv[1]
URL = 'http://127.0.0.1:30000/v1/chat/completions'

recs = [json.loads(l) for l in open('/mnt/lisiqi23/internship/zhangyikang/uiagent/press/press_pc_512.jsonl')]
recs = recs[:3]  # 固定前3条, A/B 一致

def run_once(rec):
    rec = dict(rec)
    rec['model'] = MODEL
    rec['max_tokens'] = 256
    rec['stream'] = True
    rec['stream_options'] = {"include_usage": True}
    t0 = time.time()
    ttft = None
    n = 0
    with requests.post(URL, json=rec, stream=True, timeout=300) as resp:
        for line in resp.iter_lines():
            if not line or not line.startswith(b'data:'):
                continue
            p = line[5:].strip()
            if p == b'[DONE]':
                break
            c = json.loads(p)
            ch = c.get('choices') or [{}]
            if ch and ch[0].get('delta', {}).get('content') and ttft is None:
                ttft = time.time() - t0
            if c.get('usage'):
                n = c['usage']['completion_tokens']
    total = time.time() - t0
    return ttft or total, n, total

# warmup
for _ in range(2):
    run_once(recs[0])

ttfts, dpss = [], []
for r in recs:
    for _ in range(3):
        ttft, n, total = run_once(r)
        tps = (n - 1) / max(total - ttft, 1e-6)
        ttfts.append(ttft)
        dpss.append(tps)
ttfts.sort(); dpss.sort()
med = len(dpss) // 2
print(f'BENCH {MODEL.split("/")[-1]}: TTFT_med={ttfts[med]:.2f}s decode_med={dpss[med]:.1f} tok/s '
      f'(range {dpss[0]:.1f}-{dpss[-1]:.1f})', flush=True)
