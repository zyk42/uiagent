#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pack images from the scanned split-zip index into uncompressed tar shards.

Why: JuiceFS file creation latency (~1s/file) makes extracting 350k individual
PNGs painfully slow. Tar shards = one file per worker, pure sequential
bandwidth on both read and write sides. A JSON index gives random access:
  index[name] = [shard_path, data_offset, size]
"""
import argparse
import io
import json
import os
import struct
import sys
import tarfile
import time
import zlib
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multizip import MultiFileReader  # noqa: E402
from scan_index import list_parts  # noqa: E402

READ_BUF = 16 * 1024 * 1024


def pack_range(args):
    part_paths, entries, shard_path = args
    r = io.BufferedReader(MultiFileReader(part_paths), buffer_size=READ_BUF)
    shard_index = {}
    ok, fail = 0, 0
    t0 = time.time()
    with open(shard_path, "wb", buffering=4 * 1024 * 1024) as tf:
        for i, (name, (off, method, cs, us, flags)) in enumerate(entries):
            try:
                if r.tell() != off:
                    r.seek(off)
                hdr = r.read(30)
                assert struct.unpack("<I", hdr[:4])[0] == 0x04034B50
                nl, el = struct.unpack("<HH", hdr[26:30])
                r.read(nl + el)
                data = r.read(cs)
                content = zlib.decompress(data, -15) if method == 8 else data
                assert len(content) == us

                # write tar entry manually so we know exact data offset
                data_offset = tf.tell() + 512
                ti = tarfile.TarInfo(name)
                ti.size = us
                ti.mtime = 0
                tf.write(ti.tobuf())
                tf.write(content)
                pad = (-us) % 512
                if pad:
                    tf.write(b"\0" * pad)
                shard_index[name] = [os.path.basename(shard_path), data_offset, us]
                ok += 1
                if i % 500 == 499:
                    print(f"  [{os.path.basename(shard_path)}] {i+1}/{len(entries)} "
                          f"({(i+1)/(time.time()-t0):.0f}/s)", flush=True)
            except Exception as e:
                fail += 1
                print(f"FAIL {name}: {e}", flush=True)
    return shard_path, shard_index, ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("index_json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.index_json) as f:
        index = json.load(f)

    items = [(name, tuple(v)) for name, v in index.items()]
    items.sort(key=lambda kv: kv[1][0])  # by offset -> sequential reads
    n = len(items)
    print(f"{n} entries -> {args.workers} tar shards in {args.out_dir}")

    per = (n + args.workers - 1) // args.workers
    part_paths = list_parts(args.directory)
    tasks = []
    for wi, i in enumerate(range(0, n, per)):
        shard = os.path.join(args.out_dir, f"shard_{wi:04d}.tar")
        tasks.append((part_paths, items[i:i + per], shard))

    full_index = {}
    ok_total, fail_total = 0, 0
    with Pool(args.workers) as pool:
        for shard_path, shard_index, ok, fail in pool.imap_unordered(pack_range, tasks):
            full_index.update(shard_index)
            ok_total += ok
            fail_total += fail
            print(f"shard done {os.path.basename(shard_path)}: "
                  f"total ok={ok_total} fail={fail_total}", flush=True)

    idx_path = os.path.join(args.out_dir, "tar_index.json")
    with open(idx_path, "w") as f:
        json.dump(full_index, f)
    print(f"DONE: ok={ok_total} fail={fail_total}, index -> {idx_path}")


if __name__ == "__main__":
    main()
