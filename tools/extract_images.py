#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel extraction using the scanned local-header index (bypasses the
broken central-directory offsets in these split archives).
"""
import argparse
import json
import os
import struct
import sys
import zlib
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multizip import MultiFileReader  # noqa: E402
from scan_index import list_parts  # noqa: E402


def extract_batch(args):
    part_paths, entries, out_dir = args
    r = MultiFileReader(part_paths)
    ok, fail = 0, 0
    for name, (off, method, cs, us, flags) in entries:
        try:
            out_path = os.path.join(out_dir, name)
            if os.path.exists(out_path) and os.path.getsize(out_path) == us:
                ok += 1
                continue
            r.seek(off)
            hdr = r.read(30)
            (sig, ver, fl, m, mt, md, crc, cs2, us2, nl, el) = \
                struct.unpack("<IHHHHHIIIHH", hdr)
            assert sig == 0x04034B50
            r.seek(off + 30 + nl + el)
            data = r.read(cs)
            if method == 8:
                content = zlib.decompress(data, -15)
            else:
                content = data
            assert len(content) == us, f"size mismatch {len(content)} != {us}"
            with open(out_path, "wb") as f:
                f.write(content)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"FAIL {name}: {e}", flush=True)
    return ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("index_json")
    ap.add_argument("--out_dir", default=None,
                    help="default: <directory> itself (flat uuid.png layout)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=500)
    args = ap.parse_args()

    out_dir = args.out_dir or args.directory
    os.makedirs(out_dir, exist_ok=True)

    with open(args.index_json) as f:
        index = json.load(f)
    entries = [(name, tuple(v)) for name, v in index.items()]
    print(f"{len(entries)} entries to extract -> {out_dir}")

    part_paths = list_parts(args.directory)
    batches = [entries[i:i + args.batch_size]
               for i in range(0, len(entries), args.batch_size)]
    tasks = [(part_paths, b, out_dir) for b in batches]

    ok_total, fail_total = 0, 0
    with Pool(args.workers) as pool:
        for i, (ok, fail) in enumerate(pool.imap_unordered(extract_batch, tasks)):
            ok_total += ok
            fail_total += fail
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)} batches, "
                      f"ok={ok_total} fail={fail_total}", flush=True)
    print(f"DONE: ok={ok_total} fail={fail_total}")


if __name__ == "__main__":
    main()
