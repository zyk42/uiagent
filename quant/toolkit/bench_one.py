#!/usr/bin/env python3
"""单模型测速: 假设 sglang server 已在 127.0.0.1:30000 运行。
用法: python bench_one.py <model_name>"""
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import orjson
import requests

URL = "http://127.0.0.1:30000/generate"
NAME = sys.argv[1]

# 选 3 条代表样本: 按文本长度取 p10/p50/p90 (约 2k/4k/8k token)
rows = [orjson.loads(l) for l in open('/root/quant_calib/calib_pc_512.jsonl', 'rb')]
rows.sort(key=lambda r: len(r['text']))
samples = [rows[int(len(rows)*p)] for p in (0.1, 0.5, 0.9)]


def load_sample(r):
    # 截到 assistant 前, 相当于 add_generation_prompt=True
    text = r['text'].split('<|im_start|>assistant\n')[0] + '<|im_start|>assistant\n'
    with open(r['image_path'], 'rb') as f:
        img = base64.b64encode(f.read()).decode()
    return text, img


SAMPLES = [load_sample(r) for r in samples]


def gen(text, img, max_new=128):
    """流式请求, 返回 (ttft, n_tokens, total_s)"""
    t0 = time.time()
    ttft = None
    n_tokens = 0
    with requests.post(URL, json={
        "text": text,
        "image_data": img,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new},
        "stream": True,
    }, stream=True, timeout=300) as resp:
        for line in resp.iter_lines():
            if not line or not line.startswith(b'data:'):
                continue
            payload = line[5:].strip()
            if payload == b'[DONE]':
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            if chunk.get('text') and ttft is None:
                ttft = time.time() - t0
            mi = chunk.get('meta_info') or {}
            if mi.get('completion_tokens'):
                n_tokens = mi['completion_tokens']
    total = time.time() - t0
    return ttft or total, n_tokens, total


def fmt(x):
    return f"{x:.2f}"


# ---- warmup ----
for _ in range(3):
    gen(*SAMPLES[1], max_new=32)

# ---- 单流延迟: 3 档长度 x 3 次 ----
print(f"[{NAME}] === single-stream latency ===", flush=True)
lat_rows = []
for si, (text, img) in enumerate(SAMPLES):
    ttfts, tpss = [], []
    for _ in range(3):
        ttft, n, total = gen(text, img)
        decode_tps = (n - 1) / max(total - ttft, 1e-6) if n > 1 else 0
        ttfts.append(ttft)
        tpss.append(decode_tps)
    ttfts.sort(); tpss.sort()
    lat_rows.append((si, ttfts[1], tpss[1]))  # 中位数
    print(f"[{NAME}] sample{si} (~{['2k','4k','8k'][si]} tok): TTFT={fmt(ttfts[1])}s  decode={fmt(tpss[1])} tok/s", flush=True)

# ---- 并发吞吐: 16 并发 x 中档样本 ----
print(f"[{NAME}] === throughput (16 concurrent) ===", flush=True)
text, img = SAMPLES[1]
t0 = time.time()
with ThreadPoolExecutor(16) as ex:
    results = list(ex.map(lambda _: gen(text, img), range(16)))
wall = time.time() - t0
tot_tokens = sum(r[1] for r in results)
print(f"[{NAME}] throughput: {fmt(tot_tokens/wall)} tok/s (16 conc, {tot_tokens} tokens in {fmt(wall)}s)", flush=True)

print(f"BENCH_RESULT {json.dumps({'model': NAME, 'ttft': [fmt(r[1]) for r in lat_rows], 'decode_tps': [fmt(r[2]) for r in lat_rows], 'throughput_16c': fmt(tot_tokens/wall)})}", flush=True)
