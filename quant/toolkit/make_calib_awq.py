#!/usr/bin/env python3
"""
校准集 v2: 三平台 (win 170 + mac 171 + ubuntu 171) + 动作类型最大化覆盖
- 排除评估集 eval_pc_360.jsonl 的 360 条 (防评估污染)
- 每个动作类型(样本量>=30)保底 MIN_PER_TYPE 条, 其余按比例分配
- token <= 8192 (eager 注意力内存约束), task 去重
- 图片: win/mac 从 tar, ubuntu 从分卷 zip; 输出到 pc_images_v2/ 避免与旧文件混淆
"""
import io
import json
import orjson
import os
import random
import struct
import sys
import zlib
from collections import Counter, defaultdict

BASE = '/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill'
AGENTNET = '/mnt/lisiqi23/internship/zhangyikang/uiagent/AgentNet'
WINMAC_SRC = f'{BASE}/agentnet_win_mac_pcagent.jsonl'
WINMAC_DISTILL = f'{BASE}/distill_32b_dedup.jsonl'
UBUNTU_SRC = f'{BASE}/agentnet_ubuntu_5k_pcagent.jsonl'
UBUNTU_DISTILL = f'{BASE}/distill_ubuntu_32b_dedup.jsonl'
TAR_DIR = f'{BASE}/win_mac_tar'
IMG_DIR = '/root/quant_calib/pc_images_v2'
OUT = '/root/quant_calib/calib_pc_awq_64.jsonl'
MODEL = '/mnt/lisiqi23/internship/zhangyikang/uiagent/GUI-Owl-1.5-32B-Instruct'
QUOTA_PLAT = {'win': 21, 'mac': 22, 'ubuntu': 21}
MIN_PER_TYPE = 1
MAX_LEN = 3072  # AWQ 缓存敏感
OVERSAMPLE = 20
SEED = 42

SYSTEM_PROMPT = ("You are a helpful AI PC operating assistant. "
                 "You need to help me operate the PC to complete the user's instruction.")

os.makedirs(IMG_DIR, exist_ok=True)

# 排除评估集
eval_ids = {orjson.loads(l)['sample_id'] for l in open('/root/quant_calib/eval_pc_360.jsonl', 'rb')}
print(f'excluding {len(eval_ids)} eval ids')

def valid_distill(d):
    return d.get('parse_ok') and d.get('model_thought') and d.get('model_action') and d.get('model_summary')

def norm_action(at):
    return (at or '').split('(')[0].strip() or '?'

# ============ 数据装载 ============
print('scanning sources ...', flush=True)
wm_src = {}
with open(WINMAC_SRC, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid not in eval_ids:
            wm_src[sid] = (d.get('system', '?'), d['prompt'], d['image_tar'])
ub_src = {}
with open(UBUNTU_SRC, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid not in eval_ids:
            ub_src[sid] = (d['prompt'], d['image'])

pools = {'win': defaultdict(list), 'mac': defaultdict(list), 'ubuntu': defaultdict(list)}
with open(WINMAC_DISTILL, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid in wm_src and valid_distill(d):
            plat = 'win' if wm_src[sid][0] == 'Windows' else 'mac'
            pools[plat][norm_action(d['model_action'])].append((sid, d['model_output']))
with open(UBUNTU_DISTILL, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid in ub_src and valid_distill(d):
            pools['ubuntu'][norm_action(d['model_action'])].append((sid, d['model_output']))
for p in pools:
    print(f'{p}: {sum(len(v) for v in pools[p].values())} records, {len(pools[p])} action types')

# ============ 分层抽样: 保底 + 比例 ============
rng = random.Random(SEED)
cands = []   # (platform, action_type, sid, model_output)
for plat, Q in QUOTA_PLAT.items():
    sub = {at: v for at, v in pools[plat].items() if len(v) >= 30}
    total = sum(len(v) for v in sub.values())
    # 保底
    quota = {at: MIN_PER_TYPE for at in sub}
    rest = Q - sum(quota.values())
    for at in sub:  # 剩余按比例
        quota[at] += int(len(sub[at]) / total * rest)
    while sum(quota.values()) < Q:
        big = max(sub, key=lambda a: len(sub[a]))
        quota[big] += 1
    while sum(quota.values()) > Q:
        big = max(quota, key=lambda a: quota[a] - MIN_PER_TYPE)
        if quota[big] > MIN_PER_TYPE: quota[big] -= 1
        else: break
    print(f'{plat} quota: {dict(sorted(quota.items(), key=lambda x: -x[1]))}')
    tasks_seen = set()
    for at, q in quota.items():
        c = sub[at][:]
        rng.shuffle(c)
        got = 0
        for sid, mo in c:
            if got >= q * OVERSAMPLE: break
            tid = sid.rsplit('_s', 1)[0]
            if tid in tasks_seen: continue
            tasks_seen.add(tid)
            cands.append((plat, at, sid, mo))
            got += 1
print(f'candidates: {len(cands)}')

# ============ 抽图片 ============
print('extracting images ...', flush=True)
tar_handles = {}
sys.path.insert(0, '/mnt/lisiqi23/internship/zhangyikang/uiagent/tools')
from multizip import MultiFileReader
from scan_index import list_parts
ub_reader = io.BufferedReader(MultiFileReader(list_parts(f'{AGENTNET}/ubuntu_images')),
                              buffer_size=4*1024*1024)
ub_index = json.load(open(f'{AGENTNET}/ubuntu_images_index.json'))

def read_ub_png(name):
    off, method, cs, us, flags = ub_index[name]
    ub_reader.seek(off)
    hdr = ub_reader.read(30)
    nl, el = struct.unpack('<HH', hdr[26:30])
    ub_reader.seek(off + 30 + nl + el)
    data = ub_reader.read(cs)
    return zlib.decompress(data, -15) if method == 8 else data

rows = []
for plat, at, sid, mo in cands:
    img_path = f'{IMG_DIR}/{sid}.png'
    if not os.path.exists(img_path):
        if plat in ('win', 'mac'):
            shard, off, size = wm_src[sid][2]
            if shard not in tar_handles:
                tar_handles[shard] = open(f'{TAR_DIR}/{shard}', 'rb')
            fh = tar_handles[shard]; fh.seek(off)
            png = fh.read(size)
        else:
            png = read_ub_png(ub_src[sid][1].split('/')[-1])
        with open(img_path, 'wb') as o:
            o.write(png)
    rows.append({'platform': plat, 'action_type': at, 'sample_id': sid,
                 'image_path': img_path, 'model_output': mo})
print(f'images ready: {len(rows)}')

# ============ chat template + 长度过滤 ============
print('applying chat template & tokenize filter ...', flush=True)
from transformers import AutoProcessor
from PIL import Image
processor = AutoProcessor.from_pretrained(MODEL)

for r in rows:
    resp = r['model_output']
    if not resp.strip().startswith('```'):
        resp = '```json\n' + resp.strip() + '\n```'
    prompt = wm_src[r['sample_id']][1] if r['platform'] in ('win', 'mac') else ub_src[r['sample_id']][0]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]},
        {"role": "assistant", "content": resp},
    ]
    r['text'] = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

target = {(p, at): q for p, Q in QUOTA_PLAT.items()
          for at, q in {a: qq for a, qq in []}.items()}  # placeholder, 重建 below
# 重建 (plat, at) -> 目标数
target = {}
for plat, Q in QUOTA_PLAT.items():
    sub = {at: v for at, v in pools[plat].items() if len(v) >= 30}
    total = sum(len(v) for v in sub.values())
    quota = {at: MIN_PER_TYPE for at in sub}
    rest = Q - sum(quota.values())
    for at in sub:
        quota[at] += int(len(sub[at]) / total * rest)
    while sum(quota.values()) < Q:
        quota[max(sub, key=lambda a: len(sub[a]))] += 1
    while sum(quota.values()) > Q:
        big = max(quota, key=lambda a: quota[a] - MIN_PER_TYPE)
        if quota[big] > MIN_PER_TYPE: quota[big] -= 1
        else: break
    for at, q in quota.items():
        target[(plat, at)] = q

kept = []
kept_cnt = Counter()
dropped_long = 0
for r in rows:
    key = (r['platform'], r['action_type'])
    if kept_cnt[key] >= target.get(key, 0):
        continue
    img = Image.open(r['image_path']).convert('RGB')
    n = len(processor(text=[r['text']], images=[img])['input_ids'][0])
    if n > MAX_LEN:
        dropped_long += 1
        continue
    r['n_tokens'] = n
    kept.append(r)
    kept_cnt[key] += 1
print(f'kept {len(kept)} (dropped {dropped_long} overlong)')

with open(OUT, 'wb') as out:
    for r in kept:
        out.write(orjson.dumps({'text': r['text'], 'image_path': r['image_path'],
                                'sample_id': r['sample_id'], 'platform': r['platform'],
                                'action_type': r['action_type']}) + b'\n')
print(f'DONE -> {OUT}')
