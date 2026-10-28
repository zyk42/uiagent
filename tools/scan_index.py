#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan a (buggy-offset) split zip for local file headers and build a reliable
name -> (offset, ...) index. The CD in these archives has unusable header
offsets, so we treat the file as a linear stream of PK\x03\x04 records.
"""
import argparse
import io
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multizip import MultiFileReader, open_split_zip  # noqa: E402

MAGIC = b"PK\x03\x04"
NAME_RE = re.compile(rb"^[0-9a-fA-F\-]{36}\.(png|jpg|jpeg)$")
CHUNK = 64 * 1024 * 1024


def list_parts(directory):
    parts = sorted(p for p in os.listdir(directory)
                   if p.startswith("images.z") and p != "images.zip")
    parts.append("images.zip")
    return [os.path.join(directory, p) for p in parts]


def scan(directory, out_path):
    paths = list_parts(directory)
    r = MultiFileReader(paths)
    total = r.total
    print(f"total size: {total/1e9:.1f} GB, parts: {len(paths)}")

    index = {}   # name -> (offset, method, compsize_local, size_local, flags)
    pos = 0
    carry = b""
    n_found = 0
    while pos < total:
        r.seek(pos)
        buf = r.read(CHUNK)
        if not buf:
            break
        data = carry + buf
        start = 0
        while True:
            j = data.find(MAGIC, start)
            if j < 0:
                break
            abs_off = pos - len(carry) + j
            # need 30 bytes header + name; if near chunk end, leave to carry
            if j + 30 > len(data):
                break
            (sig, ver, flags, method, mtime, mdate,
             crc, compsize, uncsize, namelen, extralen) = \
                struct.unpack("<IHHHHHIIIHH", data[j:j + 30])
            if namelen > 0 and namelen < 512 and j + 30 + namelen <= len(data) \
                    and method in (0, 8) and extralen < 4096:
                name = data[j + 30: j + 30 + namelen]
                if NAME_RE.match(name):
                    index[name.decode()] = (abs_off, method, compsize,
                                            uncsize, flags)
                    n_found += 1
                    if n_found % 50000 == 0:
                        print(f"  found {n_found} entries @ {abs_off/1e9:.1f} GB",
                              flush=True)
            start = j + 4
        # keep tail for boundary-crossing headers
        carry = data[-(4096 + 512):] if len(data) > 4608 else data
        pos += len(buf)
        if pos % (CHUNK * 16) < CHUNK:
            print(f"  scanned {pos/1e9:.1f}/{total/1e9:.1f} GB", flush=True)

    print(f"scan done: {n_found} local headers")
    with open(out_path, "w") as f:
        json.dump(index, f)
    print(f"index written to {out_path}")

    # cross-check against central directory
    zf = open_split_zip(directory)
    cd = {i.filename: (i.compress_size, i.file_size) for i in zf.infolist()}
    missing_in_scan = set(cd) - set(index)
    missing_in_cd = set(index) - set(cd)
    print(f"CD entries: {len(cd)}, scanned: {len(index)}")
    print(f"in CD but not scanned: {len(missing_in_scan)}")
    print(f"scanned but not in CD: {len(missing_in_cd)}")
    mismatch = 0
    for name in list(set(cd) & set(index)):
        off, method, cs_local, us_local, flags = index[name]
        cs_cd, us_cd = cd[name]
        if not (flags & 0x08):  # sizes present in local header
            if cs_local != cs_cd or us_local != us_cd:
                mismatch += 1
    print(f"size mismatches (local vs CD): {mismatch}")
    if missing_in_scan:
        print("  sample missing:", list(missing_in_scan)[:5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("out_index")
    args = ap.parse_args()
    scan(args.directory, args.out_index)
