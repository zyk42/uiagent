#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast parallel extraction for the scanned split-zip index.

Key optimizations vs the naive version:
1. Byte-range partitioning: sort entry offsets, split into N contiguous ranges;
   each worker reads its range SEQUENTIALLY (no 350k random seek RPCs).
2. Large read buffer (16 MB) over the multi-part reader.
3. exists()-check once per batch with a single scandir instead of per-file stat.
"""
import argparse
import io
import json
import os
import struct
import sys
import zlib
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multizip import MultiFileReader  # noqa: E402
from scan_index import list_parts  # noqa: E402

READ_BUF = 16 * 1024 * 1024


def extract_range(args):
    part_paths, entries, out_dir = args
    r = io.BufferedReader(MultiFileReader(part_paths), buffer_size=READ_BUF)
    ok, fail, skip = 0, 0, 0
    import time
    t0 = time.time()
    for idx, (name, (off, method, cs, us, flags)) in enumerate(entries):
        if idx % 500 == 499:
            print(f"  [w] {idx+1}/{len(entries)} "
                  f"({(idx+1)/(time.time()-t0):.0f}/s)", flush=True)
        try:
            out_path = os.path.join(out_dir, name)
            # entries are sorted by offset & contiguous: only seek when we must.
            # BufferedReader.seek() would otherwise discard the whole buffer.
            if r.tell() != off:
                r.seek(off)
            hdr = r.read(30)
            sig = struct.unpack("<I", hdr[:4])[0]
            assert sig == 0x04034B50
            nl, el = struct.unpack("<HH", hdr[26:30])
            r.read(nl + el)  # skip name+extra (sequential read stays buffered)
            data = r.read(cs)
            content = zlib.decompress(data, -15) if method == 8 else data
            assert len(content) == us
            with open(out_path, "wb") as f:
                f.write(content)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"FAIL {name}: {e}", flush=True)
    return ok, fail, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("index_json")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()

    out_dir = args.out_dir or args.directory
    os.makedirs(out_dir, exist_ok=True)

    with open(args.index_json) as f:
        index = json.load(f)

    # resume support: one scandir for all existing files
    existing = {e.name for e in os.scandir(out_dir) if e.name.endswith(".png")}
    print(f"{len(existing)} already extracted, skipping")

    items = [(name, tuple(v)) for name, v in index.items() if name not in existing]
    items.sort(key=lambda kv: kv[1][0])  # sort by offset -> sequential I/O
    n = len(items)
    print(f"{n} entries to extract")

    # partition into contiguous byte ranges
    per = (n + args.workers - 1) // args.workers
    part_paths = list_parts(args.directory)
    tasks = [(part_paths, items[i:i + per], out_dir)
             for i in range(0, n, per)]

    ok_total, fail_total = 0, 0
    with Pool(args.workers) as pool:
        for i, (ok, fail, _) in enumerate(pool.imap_unordered(extract_range, tasks)):
            ok_total += ok
            fail_total += fail
            print(f"  worker done {i+1}/{len(tasks)}: ok={ok_total} fail={fail_total}",
                  flush=True)
    print(f"DONE: ok={ok_total} fail={fail_total}")


if __name__ == "__main__":
    main()
