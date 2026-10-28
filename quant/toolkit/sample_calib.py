#!/usr/bin/env python3
"""从 regen.jsonl 分层抽样校准集：function 分层 + episode 去重"""
import orjson, re, random
from collections import Counter, defaultdict

F = '/mnt/xiaoai-guiagent/guiagent/training-out/shot-navigation-rl-merge/20260806zwzv5/regen/regen-400k/regen.jsonl'
OUT = '/root/quant_calib/calib_512.jsonl'
N = 512
SEED = 42
TOOL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.S)

# ---- pass 1: 收集每条记录的 (行号, function, episode) ----
pool = defaultdict(list)   # function -> [(lineno, episode_id)]
pop_cnt = Counter()
with open(F, 'rb') as f:
    for i, line in enumerate(f):
        d = orjson.loads(line)
        m = TOOL_RE.findall(d.get('ground_truth', ''))
        if not m:
            continue
        try:
            nm = orjson.loads(m[0]).get('name')
        except Exception:
            continue  # 剔除解析错误
        pool[nm].append((i, d.get('episode_id', '')))
        pop_cnt[nm] += 1

total = sum(pop_cnt.values())
print(f'pass1 done: {total} valid records, functions: {dict(pop_cnt.most_common())}')

# ---- 配额分配: 正比 + 最大余数法, 每类保底1 ----
raw = {nm: pop_cnt[nm] / total * N for nm in pop_cnt}
quota = {nm: max(1, int(v)) for nm, v in raw.items()}
# 用最大余数补到 N
rem = sorted(raw, key=lambda nm: raw[nm] - int(raw[nm]), reverse=True)
while sum(quota.values()) < N:
    for nm in rem:
        if sum(quota.values()) >= N: break
        quota[nm] += 1
# 超出则从最大类减
while sum(quota.values()) > N:
    big = max(quota, key=lambda nm: quota[nm] - max(1, int(raw[nm])))
    if quota[big] > 1: quota[big] -= 1
    else: break
print(f'quota: {dict(sorted(quota.items(), key=lambda x: -x[1]))}  sum={sum(quota.values())}')

# ---- pass 2 选择: 每层内 shuffle 后按 episode 去重取满配额 ----
rng = random.Random(SEED)
picked = set()   # linenos
picked_ep = set()
sample_cnt = Counter()
for nm, q in quota.items():
    cands = pool[nm][:]
    rng.shuffle(cands)
    got = 0
    for lineno, ep in cands:
        if got >= q: break
        if ep in picked_ep: continue
        picked_ep.add(ep)
        picked.add(lineno)
        got += 1
    sample_cnt[nm] = got
    if got < q:
        print(f'WARN: {nm} only got {got}/{q} (episode dedup)')

# ---- pass 3: 按行号抽取原行写出 ----
written = 0
with open(F, 'rb') as f, open(OUT, 'wb') as out:
    for i, line in enumerate(f):
        if i in picked:
            out.write(line)
            written += 1
print(f'written {written} records -> {OUT}')
print(f'sample distribution: {dict(sample_cnt.most_common())}')
