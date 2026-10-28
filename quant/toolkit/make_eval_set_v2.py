#!/usr/bin/env python3
"""
三平台量化一致性评估集: win 120 + mac 120 + ubuntu 120 = 360 条
- 参照输出: BF16 模型蒸馏输出 (win/mac: distill_32b_dedup.jsonl, ubuntu: distill_ubuntu_32b_dedup.jsonl)
- 排除校准集 sample_id; 平台内按动作类型分层 + task 去重
- win/mac 图片从 tar 分片读; ubuntu 图片从分卷 zip 按扫描索引读
输出: eval_pc_360.jsonl, 字段同 eval_pc_300 + platform
"""
import io
import json
import orjson
import os
import random
import re
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
IMG_DIR = '/root/quant_calib/eval_images'
OUT = '/root/quant_calib/eval_pc_360.jsonl'
PER_PLATFORM = 120
SEED = 123

os.makedirs(IMG_DIR, exist_ok=True)

# 校准集排除
calib_ids = set()
for f in ('/root/quant_calib/calib_pc_512.jsonl', '/root/quant_calib/calib_pc_awq_64.jsonl'):
    for line in open(f, 'rb'):
        calib_ids.add(orjson.loads(line)['sample_id'])

def valid_distill(d):
    return d.get('parse_ok') and d.get('model_thought') and d.get('model_action') and d.get('model_summary')

def norm_action(at):
    return (at or '').split('(')[0].strip() or '?'

# ============ win/mac: 一次扫描源文件取 system/prompt/image_tar ============
print('scanning win/mac source ...', flush=True)
wm_src = {}   # sid -> (system, prompt, image_tar)
with open(WINMAC_SRC, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid in calib_ids:
            continue
        wm_src[sid] = (d.get('system', '?'), d['prompt'], d['image_tar'])
print(f'win/mac source records (excl calib): {len(wm_src)}')

# 蒸馏输出
wm_pool = defaultdict(list)   # (platform, action_type) -> [(sid, distill_rec)]
with open(WINMAC_DISTILL, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid not in wm_src or not valid_distill(d):
            continue
        plat = 'win' if wm_src[sid][0] == 'Windows' else 'mac'
        wm_pool[(plat, norm_action(d['model_action']))].append((sid, d))
print('win/mac pool sizes:', {k: len(v) for k, v in sorted(wm_pool.items()) if len(v) >= 20})

# ============ ubuntu ============
print('scanning ubuntu source ...', flush=True)
ub_src = {}
with open(UBUNTU_SRC, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid in calib_ids:
            continue
        ub_src[sid] = (d['prompt'], d['image'])  # image = ubuntu_images/<uuid>.png
ub_pool = defaultdict(list)
with open(UBUNTU_DISTILL, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid not in ub_src or not valid_distill(d):
            continue
        ub_pool[('ubuntu', norm_action(d['model_action']))].append((sid, d))
print('ubuntu pool sizes:', {k: len(v) for k, v in sorted(ub_pool.items()) if len(v) >= 20})

# ============ 分层抽样 (平台内按动作正比 + task 去重) ============
rng = random.Random(SEED)
picked = []   # (platform, sid, distill_rec)
for plat in ('win', 'mac', 'ubuntu'):
    pool = wm_pool if plat in ('win', 'mac') else ub_pool
    sub = {at: v for (p, at), v in pool.items() if p == plat}
    total = sum(len(v) for v in sub.values())
    quota = {at: max(1, int(len(v) / total * PER_PLATFORM)) for at, v in sub.items() if len(v) >= 30}
    rem = sorted(quota, key=lambda a: len(sub[a]) / total * PER_PLATFORM - quota[a], reverse=True)
    while sum(quota.values()) < PER_PLATFORM:
        for at in rem:
            if sum(quota.values()) >= PER_PLATFORM: break
            quota[at] += 1
    while sum(quota.values()) > PER_PLATFORM:
        big = max(quota, key=lambda a: quota[a])
        if quota[big] > 1: quota[big] -= 1
        else: break
    tasks_seen = set()
    for at, q in quota.items():
        cands = sub[at][:]
        rng.shuffle(cands)
        got = 0
        for sid, d in cands:
            if got >= q: break
            tid = sid.rsplit('_s', 1)[0]
            if tid in tasks_seen: continue
            tasks_seen.add(tid)
            picked.append((plat, sid, d))
            got += 1
    print(f'{plat}: picked {sum(1 for p,_,_ in picked if p==plat)}, quota={quota}')

# ============ 抽图片 ============
print('extracting images ...', flush=True)
tar_handles = {}
# ubuntu zip reader
sys.path.insert(0, '/mnt/lisiqi23/internship/zhangyikang/uiagent/tools')
from multizip import MultiFileReader
from scan_index import list_parts
ub_parts = list_parts(f'{AGENTNET}/ubuntu_images')
ub_reader = io.BufferedReader(MultiFileReader(ub_parts), buffer_size=4*1024*1024)
ub_index = json.load(open(f'{AGENTNET}/ubuntu_images_index.json'))

def read_ub_png(name):
    off, method, cs, us, flags = ub_index[name]
    ub_reader.seek(off)
    hdr = ub_reader.read(30)
    assert struct.unpack('<I', hdr[:4])[0] == 0x04034B50
    nl, el = struct.unpack('<HH', hdr[26:30])
    ub_reader.seek(off + 30 + nl + el)
    data = ub_reader.read(cs)
    content = zlib.decompress(data, -15) if method == 8 else data
    assert len(content) == us
    return content

n_out = 0
with open(OUT, 'wb') as out:
    for plat, sid, d in picked:
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
        prompt = wm_src[sid][1] if plat in ('win', 'mac') else ub_src[sid][0]
        resp = d['model_output']
        if not resp.strip().startswith('```'):
            resp = '```json\n' + resp.strip() + '\n```'
        out.write(orjson.dumps({
            'sample_id': sid, 'platform': plat, 'prompt': prompt, 'image_path': img_path,
            'ref_output': resp, 'ref_thought': d['model_thought'],
            'ref_action': d['model_action'], 'ref_summary': d['model_summary'],
            'gt_action': d.get('gt_action', ''),
        }) + b'\n')
        n_out += 1
print(f'DONE: {n_out} -> {OUT}')
