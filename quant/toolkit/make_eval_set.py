#!/usr/bin/env python3
"""
构建量化一致性评估集:
- 来源: distill_32b_snapshot (BF16 模型 temperature=0 的输出 = 参照答案)
- 排除校准集 calib_pc_512 的 sample_id (避免校准泄漏高估一致性)
- 分层按 action 类型 + task 去重, 300 条
- 图片从 tar 分片抽到 eval_images/
输出字段: sample_id, prompt, image_path, ref_output(带```json围栏), ref_action, ref_thought, ref_summary, gt_action
"""
import orjson
import random
import re
from collections import Counter, defaultdict

BASE = '/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill'
SNAPSHOT = f'{BASE}/distill_32b_snapshot_20261014.jsonl'
SOURCE = f'{BASE}/agentnet_win_mac_pcagent.jsonl'
TAR_DIR = f'{BASE}/win_mac_tar'
IMG_DIR = '/root/quant_calib/eval_images'
OUT = '/root/quant_calib/eval_pc_300.jsonl'
N = 300
SEED = 123

import os
os.makedirs(IMG_DIR, exist_ok=True)

# 校准集 sample_id (要排除)
calib_ids = set()
for f in ('/root/quant_calib/calib_pc_512.jsonl', '/root/quant_calib/calib_pc_awq_64.jsonl'):
    for line in open(f, 'rb'):
        calib_ids.add(orjson.loads(line)['sample_id'])
print(f'excluding {len(calib_ids)} calib sample_ids')

# ---- 1) 分层抽样 ----
pool = defaultdict(list)
with open(SNAPSHOT, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        if not d.get('parse_ok'):
            continue
        if not (d.get('model_thought') and d.get('model_action') and d.get('model_summary')):
            continue
        sid = d['sample_id']
        if sid in calib_ids:
            continue
        at = (d.get('model_action') or '').split('(')[0].strip() or '?'
        pool[at].append((sid, sid.rsplit('_s', 1)[0], d))

total = sum(len(v) for v in pool.values())
print(f'eligible: {total}')

raw_q = {at: len(v) / total * N for at, v in pool.items()}
quota = {at: max(1, int(q)) for at, q in raw_q.items() if len(pool[at]) >= 30}
rem = sorted(quota, key=lambda a: raw_q[a] - int(raw_q[a]), reverse=True)
while sum(quota.values()) < N:
    for at in rem:
        if sum(quota.values()) >= N: break
        quota[at] += 1
while sum(quota.values()) > N:
    big = max(quota, key=lambda a: quota[a] - max(1, int(raw_q[a])))
    if quota[big] > 1: quota[big] -= 1
    else: break
print('quota:', dict(sorted(quota.items(), key=lambda x: -x[1])), 'sum:', sum(quota.values()))

rng = random.Random(SEED)
picked = {}
picked_tasks = set()
for at, q in quota.items():
    cands = pool[at][:]
    rng.shuffle(cands)
    got = 0
    for sid, tid, d in cands:
        if got >= q: break
        if tid in picked_tasks: continue
        picked_tasks.add(tid)
        picked[sid] = d
        got += 1
    if got < q:
        for sid, tid, d in cands:
            if got >= q: break
            if sid in picked: continue
            picked[sid] = d
            got += 1
print(f'picked: {len(picked)}')

# ---- 2) 回源捞 prompt / image_tar ----
need = set(picked)
found = {}
ID_RE = re.compile(rb'"sample_id":\s*"([^"]+)"')
with open(SOURCE, 'rb') as f:
    for line in f:
        m = ID_RE.search(line[:200])
        if not m: continue
        sid = m.group(1).decode()
        if sid in need:
            d = orjson.loads(line)
            found[sid] = (d['prompt'], d['image_tar'], d.get('action', ''))
            if len(found) == len(need): break
print(f'found {len(found)}/{len(need)} in source')

# ---- 3) 抽图片 + 写评估集 ----
tar_handles = {}
n_out = 0
with open(OUT, 'wb') as out:
    for sid, d in picked.items():
        if sid not in found: continue
        prompt, (shard, off, size), gt_action = found[sid]
        if shard not in tar_handles:
            tar_handles[shard] = open(f'{TAR_DIR}/{shard}', 'rb')
        fh = tar_handles[shard]; fh.seek(off)
        img_path = f'{IMG_DIR}/{sid}.png'
        if not os.path.exists(img_path):
            with open(img_path, 'wb') as o:
                o.write(fh.read(size))
        resp = d['model_output']
        if not resp.strip().startswith('```'):
            resp = '```json\n' + resp.strip() + '\n```'
        out.write(orjson.dumps({
            'sample_id': sid, 'prompt': prompt, 'image_path': img_path,
            'ref_output': resp, 'ref_thought': d['model_thought'],
            'ref_action': d['model_action'], 'ref_summary': d['model_summary'],
            'gt_action': gt_action,
        }) + b'\n')
        n_out += 1
print(f'DONE: {n_out} -> {OUT}')
