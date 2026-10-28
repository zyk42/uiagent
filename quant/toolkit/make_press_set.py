#!/usr/bin/env python3
"""
构建压测集: 512 条, 输入输出长度窄带 (prompt 3400-4200 tok, completion 90-150 tok),
三平台均衡 (win 170 / mac 171 / ubuntu 171), task 去重, 排除校准/评估集。
输出:
  press_pc_512.jsonl       OpenAI chat 格式 (press README §1.1), 图片 base64 内联
  press_pc_512_prompt.jsonl  vLLM custom 格式 (§1.2), 拍平成单条
  press_pc_512.meta.jsonl  sample_id/platform/token 数等元信息
"""
import base64
import io
import json
import orjson
import os
import random
import struct
import sys
import zlib
from collections import defaultdict

BASE = '/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill'
AGENTNET = '/mnt/lisiqi23/internship/zhangyikang/uiagent/AgentNet'
WINMAC_SRC = f'{BASE}/agentnet_win_mac_pcagent.jsonl'
WINMAC_DISTILL = f'{BASE}/distill_32b_dedup.jsonl'
UBUNTU_SRC = f'{BASE}/agentnet_ubuntu_5k_pcagent.jsonl'
UBUNTU_DISTILL = f'{BASE}/distill_ubuntu_32b_dedup.jsonl'
TAR_DIR = f'{BASE}/win_mac_tar'
OUT_DIR = '/mnt/lisiqi23/internship/zhangyikang/uiagent/press'
OUT = f'{OUT_DIR}/press_pc_512.jsonl'

P_LO, P_HI = 3400, 4200     # prompt_tokens 窄带
C_LO, C_HI = 90, 150        # completion_tokens 窄带
QUOTA_PLAT = {'win': 170, 'mac': 171, 'ubuntu': 171}
SEED = 7

SYSTEM_PROMPT = ("You are a helpful AI PC operating assistant. "
                 "You need to help me operate the PC to complete the user's instruction.")

# 排除校准+评估集
exclude = set()
for f in ('/root/quant_calib/calib_pc_512.jsonl', '/root/quant_calib/calib_pc_awq_64.jsonl',
          '/root/quant_calib/eval_pc_360.jsonl'):
    for line in open(f, 'rb'):
        exclude.add(orjson.loads(line)['sample_id'])
print(f'exclude {len(exclude)} calib/eval ids')

def valid(d):
    if not (d.get('parse_ok') and d.get('model_thought') and d.get('model_action') and d.get('model_summary')):
        return False
    pt, ct = d.get('prompt_tokens', 0), d.get('completion_tokens', 0)
    return P_LO <= pt <= P_HI and C_LO <= ct <= C_HI

# ============ 扫描蒸馏文件得候选 ============
cand = {'win': [], 'mac': [], 'ubuntu': []}   # sid -> (pt, ct)
winmac_ids = {}
with open(WINMAC_DISTILL, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid in exclude or not valid(d):
            continue
        winmac_ids[sid] = (d['prompt_tokens'], d['completion_tokens'])
ubuntu_ids = {}
with open(UBUNTU_DISTILL, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid in exclude or not valid(d):
            continue
        ubuntu_ids[sid] = (d['prompt_tokens'], d['completion_tokens'])
print(f'in-band candidates: winmac={len(winmac_ids)} ubuntu={len(ubuntu_ids)}')

# ============ 回源取 prompt/image 引用 + 平台 ============
print('scanning sources ...', flush=True)
wm_info = {}
with open(WINMAC_SRC, 'rb') as f:
    for line in f:
        m = line[:200]
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid in winmac_ids:
            wm_info[sid] = (d.get('system', '?'), d['prompt'], d['image_tar'])
        if len(wm_info) == len(winmac_ids):
            break
ub_info = {}
with open(UBUNTU_SRC, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        sid = d['sample_id']
        if sid in ubuntu_ids:
            ub_info[sid] = (d['prompt'], d['image'])
        if len(ub_info) == len(ubuntu_ids):
            break
print(f'source joined: winmac={len(wm_info)} ubuntu={len(ub_info)}')

for sid in winmac_ids:
    if sid in wm_info:
        cand['win' if wm_info[sid][0] == 'Windows' else 'mac'].append(sid)
for sid in ubuntu_ids:
    if sid in ub_info:
        cand['ubuntu'].append(sid)
print({k: len(v) for k, v in cand.items()})

# ============ 分层抽样 (task 去重) ============
rng = random.Random(SEED)
picked = []
for plat, Q in QUOTA_PLAT.items():
    sids = cand[plat][:]
    rng.shuffle(sids)
    seen_task = set()
    got = 0
    for sid in sids:
        if got >= Q:
            break
        tid = sid.rsplit('_s', 1)[0]
        if tid in seen_task:
            continue
        seen_task.add(tid)
        picked.append((plat, sid))
        got += 1
    print(f'{plat}: {got}/{Q}')

# ============ 抽图片 + base64 内联 ============
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

def get_png(plat, sid):
    if plat in ('win', 'mac'):
        shard, off, size = wm_info[sid][2]
        if shard not in tar_handles:
            tar_handles[shard] = open(f'{TAR_DIR}/{shard}', 'rb')
        fh = tar_handles[shard]; fh.seek(off)
        return fh.read(size)
    return read_ub_png(ub_info[sid][1].split('/')[-1])

with open(OUT, 'w') as out, open(OUT.replace('.jsonl', '.meta.jsonl'), 'w') as meta:
    for plat, sid in picked:
        prompt = wm_info[sid][1] if plat in ('win', 'mac') else ub_info[sid][0]
        b64 = base64.b64encode(get_png(plat, sid)).decode()
        rec = {
            "model": "base_model",
            "temperature": 0,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
        }
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        pt, ct = (winmac_ids if plat != 'ubuntu' else ubuntu_ids)[sid]
        meta.write(json.dumps({'sample_id': sid, 'platform': plat,
                               'prompt_tokens': pt, 'completion_tokens': ct}) + '\n')

# ============ vLLM custom 格式 (拍平) ============
with open(OUT) as f, open(OUT.replace('.jsonl', '_prompt.jsonl'), 'w') as g:
    for line in f:
        rec = json.loads(line)
        parts = []
        for m in rec['messages']:
            c = m['content']
            if isinstance(c, str):
                if c.strip():
                    parts.append({"type": "text", "text": c})
            else:
                parts.extend(c)
        g.write(json.dumps({"prompt": parts}, ensure_ascii=False) + '\n')
print(f'DONE -> {OUT} (+ _prompt.jsonl, .meta.jsonl)')
