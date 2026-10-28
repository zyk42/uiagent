import orjson, os, re
BASE = '/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill'
SOURCE = f'{BASE}/agentnet_win_mac_pcagent.jsonl'
TAR_DIR = f'{BASE}/win_mac_tar'

need = {}
for line in open('/root/quant_calib/calib_pc_512.jsonl', 'rb'):
    d = orjson.loads(line)
    if not os.path.exists(d['image_path']):
        need[d['sample_id']] = d['image_path']
print(f'missing images: {len(need)}')
if not need:
    raise SystemExit

ID_RE = re.compile(rb'"sample_id":\s*"([^"]+)"')
tar_handles = {}
fixed = 0
with open(SOURCE, 'rb') as f:
    for line in f:
        m = ID_RE.search(line[:200])
        if not m: continue
        sid = m.group(1).decode()
        if sid in need:
            d = orjson.loads(line)
            shard, off, size = d['image_tar']
            if shard not in tar_handles:
                tar_handles[shard] = open(f'{TAR_DIR}/{shard}', 'rb')
            fh = tar_handles[shard]; fh.seek(off)
            with open(need[sid], 'wb') as out:
                out.write(fh.read(size))
            fixed += 1
            if fixed == len(need): break
print(f'repaired {fixed}/{len(need)}')
