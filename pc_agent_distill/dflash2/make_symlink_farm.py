#!/usr/bin/env python3
"""Build the /dev/shm symlink farm pointing at JuiceFS-extracted images.

The training pipeline references images as
    /dev/shm/pc_agent_images/<set>/<shard_stem>_<offset>.png
while the durable files live on JuiceFS at
    <jfs_root>/<set>/<shard_stem>/<offset>.png

This script (re)creates the symlink farm on /dev/shm (tmpfs, fast). If
/dev/shm gets wiped again, just rerun this script (~1 min).

Usage:
  python3 make_symlink_farm.py
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

JFS_ROOT = "/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill/images_extracted"
SHM_ROOT = "/dev/shm/pc_agent_images"


def link_shard(args):
    set_name, shard_dir = args
    src_dir = os.path.join(JFS_ROOT, set_name, shard_dir)
    dst_dir = os.path.join(SHM_ROOT, set_name)
    n = 0
    for fname in os.listdir(src_dir):
        offset = fname.removesuffix(".png")
        link = os.path.join(dst_dir, f"{shard_dir}_{offset}.png")
        target = os.path.join(src_dir, fname)
        if os.path.islink(link) or os.path.exists(link):
            continue
        os.symlink(target, link)
        n += 1
    return f"{set_name}/{shard_dir}: {n} links"


def main():
    jobs = []
    for set_name in sorted(os.listdir(JFS_ROOT)):
        set_dir = os.path.join(JFS_ROOT, set_name)
        if not os.path.isdir(set_dir):
            continue
        os.makedirs(os.path.join(SHM_ROOT, set_name), exist_ok=True)
        for shard_dir in sorted(os.listdir(set_dir)):
            jobs.append((set_name, shard_dir))

    total = 0
    with ProcessPoolExecutor(16) as ex:
        for res in ex.map(link_shard, jobs):
            print(res, flush=True)
            total += int(res.rsplit(":", 1)[1].strip().split()[0])
    print(f"DONE: {total} symlinks created under {SHM_ROOT}")


if __name__ == "__main__":
    main()
