#!/usr/bin/env python3
"""Extract PC-Agent tar-shard images to /dev/shm for SpecForge training.

Reads tar_index.json (filename -> [shard, offset, size]) for one image set,
extracts every image to <out_dir>/<shard_stem>_<offset>.png so that a sample's
image path is derivable purely from its `image_tar` field:

    shard, offset, size = sample["image_tar"]
    path = f"{out_dir}/{shard.removesuffix('.tar')}_{offset}.png"

Shards are processed in parallel (one reader per shard, near-sequential IO).
Existing files with the right size are skipped (resumable).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool


def extract_shard(args):
    tar_dir, shard, entries, out_dir, nested = args
    # entries: list of (offset, size, out_name), sorted by offset
    done = skipped = 0
    bytes_written = 0
    tar_path = os.path.join(tar_dir, shard)
    if nested:
        # per-shard subdir avoids JuiceFS same-directory write-lock contention
        out_dir = os.path.join(out_dir, shard.removesuffix(".tar"))
        os.makedirs(out_dir, exist_ok=True)
    try:
        with open(tar_path, "rb") as f:
            for offset, size, out_name in entries:
                out_path = os.path.join(out_dir, out_name)
                if os.path.exists(out_path) and os.path.getsize(out_path) == size:
                    skipped += 1
                    continue
                f.seek(offset)
                data = f.read(size)
                if len(data) != size:
                    return f"ERROR {shard}@{offset}: short read {len(data)}/{size}"
                with open(out_path, "wb") as out:
                    out.write(data)
                done += 1
                bytes_written += size
    except Exception as e:
        return f"ERROR {shard}: {e}"
    return f"OK {shard}: wrote {done} ({bytes_written/1e6:.0f}MB), skipped {skipped}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--nested", action="store_true",
                    help="write into per-shard subdirs (JuiceFS lock-safe)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.tar_dir, "tar_index.json")) as f:
        index = json.load(f)

    by_shard = defaultdict(list)
    for _fname, (shard, offset, size) in index.items():
        if args.nested:
            out_name = f"{offset}.png"
        else:
            out_name = f"{shard.removesuffix('.tar')}_{offset}.png"
        by_shard[shard].append((offset, size, out_name))

    jobs = []
    for shard, entries in sorted(by_shard.items()):
        entries.sort(key=lambda x: x[0])
        jobs.append((args.tar_dir, shard, entries, args.out_dir, args.nested))

    total = len(index)
    print(f"{args.tar_dir}: {total} images in {len(jobs)} shards -> {args.out_dir}",
          flush=True)

    finished = 0
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(extract_shard, jobs):
            finished += 1
            print(f"[{finished}/{len(jobs)}] {res}", flush=True)
            if res.startswith("ERROR"):
                print("FATAL: shard failed, aborting", file=sys.stderr)
                pool.terminate()
                sys.exit(1)

    # verify count (per-shard results already account for every entry)
    if args.nested:
        n_files = 0
        for shard in by_shard:
            sub = os.path.join(args.out_dir, shard.removesuffix(".tar"))
            n_files += sum(1 for _ in os.scandir(sub))
    else:
        n_files = sum(1 for _ in os.scandir(args.out_dir))
    print(f"DONE {args.out_dir}: {n_files} files (expected {total})", flush=True)
    if n_files != total:
        sys.exit(2)


if __name__ == "__main__":
    main()
