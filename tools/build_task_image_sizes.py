#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build task_id -> actual image size map by reading PNG IHDR headers from tar shards."""
import json
import struct
import sys
from multiprocessing import Pool

BASE = '/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill'
IDX = json.load(open(f'{BASE}/win_mac_tar/tar_index.json'))
_handles = {}


def get_handle(shard):
    if shard not in _handles:
        _handles[shard] = open(f'{BASE}/win_mac_tar/{shard}', 'rb')
    return _handles[shard]


def png_size(item):
    tid, name = item
    shard, off, size = IDX[name]
    f = get_handle(shard)
    f.seek(off + 16)  # PNG IHDR width/height
    w, h = struct.unpack('>II', f.read(8))
    return tid, w, h


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
        '/mnt/lisiqi23/internship/zhangyikang/uiagent/AgentNet/agentnet_win_mac_18k.jsonl'
    out = sys.argv[2] if len(sys.argv) > 2 else \
        f'{BASE}/task_image_sizes.json'

    task_img = {}
    with open(src) as f:
        for line in f:
            d = json.loads(line)
            if d['task_id'] not in task_img:
                task_img[d['task_id']] = d['traj'][0]['image']
    items = list(task_img.items())
    print(f'{len(items)} unique tasks')

    result = {}
    with Pool(48) as pool:
        for i, (tid, w, h) in enumerate(pool.imap_unordered(png_size, items, chunksize=64)):
            result[tid] = [w, h]
            if (i + 1) % 3000 == 0:
                print(f'  {i+1}/{len(items)}', flush=True)

    json.dump(result, open(out, 'w'))
    print(f'saved -> {out}')

    # 和 meta_data 的分辨率对比统计
    meta = {}
    with open('/mnt/lisiqi23/internship/zhangyikang/uiagent/AgentNet/meta_data_merged.jsonl') as f:
        for line in f:
            d = json.loads(line)
            meta[d['task_id']] = (d.get('screen_width'), d.get('screen_height'))
    same = diff = 0
    for tid, (w, h) in result.items():
        if meta.get(tid) == (w, h):
            same += 1
        else:
            diff += 1
    print(f'image size == meta resolution: {same}, differ: {diff}')


if __name__ == '__main__':
    main()
