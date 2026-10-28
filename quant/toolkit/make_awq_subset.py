#!/usr/bin/env python3
"""从 calib_pc_512 中抽 AWQ 专用减量子集: 128 条, <=4096 token, 按动作类型分层"""
import orjson, json, random, re
from collections import Counter, defaultdict
from PIL import Image
from transformers import AutoProcessor

SRC = '/root/quant_calib/calib_pc_512_v2.jsonl'
DST = '/root/quant_calib/calib_pc_awq_64.jsonl'
MODEL = '/mnt/lisiqi23/internship/zhangyikang/uiagent/GUI-Owl-1.5-32B-Instruct'
N, MAX_LEN = 64, 2048

processor = AutoProcessor.from_pretrained(MODEL)
pool = defaultdict(list)
for line in open(SRC, 'rb'):
    d = orjson.loads(line)
    ans = d['text'].split('<|im_start|>assistant\n')[-1]
    m = re.search(r'"Action":\s*"([^"(]+)', ans)
    at = m.group(1).strip() if m else '?'
    pool[at].append(d)

total = sum(len(v) for v in pool.values())
quota = {at: max(1, round(len(v)/total*N)) for at, v in pool.items()}
while sum(quota.values()) > N:
    big = max(quota, key=lambda a: quota[a]); quota[big] -= 1
while sum(quota.values()) < N:
    big = max(quota, key=lambda a: len(pool[a])/total*N - quota[a]); quota[big] += 1
print('quota:', dict(sorted(quota.items(), key=lambda x:-x[1])))

rng = random.Random(42)
kept = []
for at, q in quota.items():
    cands = pool[at][:]
    rng.shuffle(cands)
    got = 0
    for d in cands:
        if got >= q: break
        img = Image.open(d['image_path']).convert('RGB')
        n = len(processor(text=[d['text']], images=[img])['input_ids'][0])
        if n > MAX_LEN: continue
        kept.append(d); got += 1
    print(f'{at}: {got}/{q}')
with open(DST, 'wb') as f:
    for d in kept:
        f.write(orjson.dumps(d) + b'\n')
print(f'DONE: {len(kept)} -> {DST}')
