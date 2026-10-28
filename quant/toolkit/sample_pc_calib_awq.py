#!/usr/bin/env python3
"""
从 pc_agent_distill 蒸馏快照抽 512 条校准样本 (长度感知版):
1) 从 distill_32b_snapshot 取 parse_ok=True 且三字段齐全的记录, 按 action 类型分层 + task 去重
   (每层取 4x 候选, 供长度过滤后递补)
2) 回源 agentnet_win_mac_pcagent.jsonl (13.9GB) 按 sample_id 捞 prompt/image_tar
3) 从 tar 分片随机读取图片存到 /root/quant_calib/pc_images/
4) 套 GUI-Owl chat_template
5) 逐条 tokenize, 只保留 <= MAX_LEN 的样本 (eager 注意力 O(L^2) 瞬存约束)
"""
import orjson
import random
import re
from collections import Counter, defaultdict

BASE = '/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill'
SNAPSHOT = f'{BASE}/distill_32b_snapshot_20261014.jsonl'
SOURCE = f'{BASE}/agentnet_win_mac_pcagent.jsonl'
TAR_DIR = f'{BASE}/win_mac_tar'
IMG_DIR = '/root/quant_calib/pc_images'
OUT = '/root/quant_calib/calib_pc_awq_64.jsonl'
MODEL = '/mnt/lisiqi23/internship/zhangyikang/uiagent/GUI-Owl-1.5-32B-Instruct'
N = 64
MAX_LEN = 3072      # AWQ 缓存敏感, 进一步压短
OVERSAMPLE = 15     # <=3072 token 的样本占比低, 加大候选
SEED = 42

SYSTEM_PROMPT = ("You are a helpful AI PC operating assistant. "
                 "You need to help me operate the PC to complete the user's instruction.")

# ---------- 1) 分层抽样 (候选) ----------
print('step1: sampling from snapshot ...', flush=True)
pool = defaultdict(list)   # action_type -> [(sample_id, task_id, model_output)]
with open(SNAPSHOT, 'rb') as f:
    for line in f:
        d = orjson.loads(line)
        if not d.get('parse_ok'):
            continue
        # parse_ok=True 也可能缺字段 (Thought/Action 后截断), 要求三字段齐全
        if not (d.get('model_thought') and d.get('model_action') and d.get('model_summary')):
            continue
        ma = d.get('model_action') or ''
        at = ma.split('(')[0].strip() or '?'
        sid = d['sample_id']
        tid = sid.rsplit('_s', 1)[0]
        pool[at].append((sid, tid, d['model_output']))

total = sum(len(v) for v in pool.values())
print(f'parse_ok+complete records: {total}, action types: {len(pool)}')

raw_q = {at: len(v) / total * N for at, v in pool.items()}
# 只保留样本量 >=30 的干净动作类型参与分层 (~12 类, 覆盖 99.8%), 长尾杂项不抽
quota = {at: max(1, int(q)) for at, q in raw_q.items() if len(pool[at]) >= 30}
rem = sorted(quota, key=lambda a: raw_q[a] - int(raw_q[a]), reverse=True)
while sum(quota.values()) < N:
    for at in rem:
        if sum(quota.values()) >= N:
            break
        quota[at] += 1
while sum(quota.values()) > N:
    big = max(quota, key=lambda a: quota[a] - max(1, int(raw_q[a])))
    if quota[big] > 1:
        quota[big] -= 1
    else:
        break
print('quota:', dict(sorted(quota.items(), key=lambda x: -x[1])), 'sum:', sum(quota.values()))

rng = random.Random(SEED)
cands = []           # [(action_type, sample_id, model_output)]
picked_tasks = set()
for at, q in quota.items():
    c = pool[at][:]
    rng.shuffle(c)
    got = 0
    for sid, tid, mo in c:
        if got >= q * OVERSAMPLE:
            break
        if tid in picked_tasks:
            continue
        picked_tasks.add(tid)
        cands.append((at, sid, mo))
        got += 1
print(f'candidates: {len(cands)}')

# ---------- 2) 回源捞 prompt / image_tar ----------
print('step2: scanning source jsonl ...', flush=True)
need = {sid for _, sid, _ in cands}
found = {}
ID_RE = re.compile(rb'"sample_id":\s*"([^"]+)"')
with open(SOURCE, 'rb') as f:
    for line in f:
        m = ID_RE.search(line[:200])
        if not m:
            continue
        sid = m.group(1).decode()
        if sid in need:
            d = orjson.loads(line)
            found[sid] = (d['prompt'], d['image_tar'])
            if len(found) == len(need):
                break
print(f'found {len(found)}/{len(need)}')

# ---------- 3) 抽图片 ----------
print('step3: extracting images ...', flush=True)
tar_handles = {}
rows = []
for at, sid, mo in cands:
    if sid not in found:
        continue
    prompt, (shard, off, size) = found[sid]
    if shard not in tar_handles:
        tar_handles[shard] = open(f'{TAR_DIR}/{shard}', 'rb')
    fh = tar_handles[shard]
    fh.seek(off)
    img_path = f'{IMG_DIR}/{sid}.png'
    with open(img_path, 'wb') as out:
        out.write(fh.read(size))
    rows.append({'action_type': at, 'sample_id': sid, 'prompt': prompt,
                 'image_path': img_path, 'model_output': mo})
print(f'extracted {len(rows)} images')

# ---------- 4) 套 chat template ----------
print('step4: applying chat template ...', flush=True)
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(MODEL)

for r in rows:
    # 按蒸馏 pipeline 说明 §2.3: 统一 ```json 围栏, 对齐 PC-Agent 解析习惯
    resp = r['model_output']
    if not resp.strip().startswith('```'):
        resp = '```json\n' + resp.strip() + '\n```'
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": r['prompt']},
            {"type": "image"},
        ]},
        {"role": "assistant", "content": resp},
    ]
    r['text'] = processor.apply_chat_template(messages, add_generation_prompt=False,
                                              tokenize=False)

# ---------- 5) 长度过滤 + 按层递补 ----------
print('step5: tokenize & length filter ...', flush=True)
from PIL import Image
kept, kept_at = [], Counter()
dropped_long = 0
for r in rows:
    at = r['action_type']
    if kept_at[at] >= quota[at]:
        continue
    img = Image.open(r['image_path']).convert('RGB')
    out = processor(text=[r['text']], images=[img])
    r['n_tokens'] = len(out['input_ids'][0])
    if r['n_tokens'] > MAX_LEN:
        dropped_long += 1
        continue
    kept.append(r)
    kept_at[at] += 1

short = {at: quota[at] - kept_at[at] for at in quota if kept_at[at] < quota[at]}
if short:
    print(f'WARN strata not filled: {short}')

# 注意: 不要删除未选中样本的图片! 与 calib_pc_512 共享 pc_images 目录,
# 误删会导致主校准集缺图 (2026-10-14 事故: 445 张被误删, 已从 tar 修复)

lens = sorted(r['n_tokens'] for r in kept)
print(f'kept {len(kept)} (dropped {dropped_long} overlong), '
      f'token len: min={lens[0]} p50={lens[len(lens)//2]} max={lens[-1]}')

with open(OUT, 'wb') as out:
    for r in kept:
        out.write(orjson.dumps({'text': r['text'], 'image_path': r['image_path'],
                                'sample_id': r['sample_id']}) + b'\n')
print(f'DONE -> {OUT}')
