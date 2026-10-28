#!/usr/bin/env python3
"""Join GUI-Owl-32B distill outputs back to PC-Agent samples and emit
SpecForge DFlash2 VLM-format JSONL.

Output record (one per line):
{
  "id":     sample_id,
  "system": PC-Agent system prompt (from messages[0]),
  "image":  /dev/shm/.../<shard>_<offset>.png,
  "conversations": [
    {"role": "user",      "content": <prompt> + "\n<image>"},
    {"role": "assistant", "content": ```json fenced model output```}
  ],
  "domain": ..., "gt_action": ...
}

Filtering: keep only parse_ok == true distill rows (unparseable outputs are
not valid SFT targets). Extra quality filters (task_completed etc.) are
deliberately NOT applied here; filter downstream if needed.
"""

import argparse
import json
import os
import sys

PC_AGENT_SYSTEM_FALLBACK = (
    "You are a helpful AI PC operating assistant. You need to help me "
    "operate the PC to complete the user's instruction."
)


def load_distill(path):
    table = {}
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if not r.get("parse_ok"):
                continue
            table[r["sample_id"]] = r
    print(f"distill {os.path.basename(path)}: {n} rows, {len(table)} parse_ok unique",
          flush=True)
    return table


def image_path(image_root, image_tar):
    shard, offset, _size = image_tar
    return os.path.join(image_root, f"{shard.removesuffix('.tar')}_{offset}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="pcagent-format input jsonl")
    ap.add_argument("--distill", required=True, help="deduped distill jsonl")
    ap.add_argument("--image-root", required=True,
                    help="dir holding extracted images for this set")
    ap.add_argument("--output", required=True)
    ap.add_argument("--check-images", action="store_true",
                    help="verify every referenced image file exists")
    args = ap.parse_args()

    distill = load_distill(args.distill)

    n_in = n_hit = n_miss = n_noimg = 0
    with open(args.input, encoding="utf-8") as fin, \
            open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            d = json.loads(line)
            n_in += 1
            r = distill.get(d["sample_id"])
            if r is None:
                n_miss += 1
                continue

            resp = r["model_output"].strip()
            if not resp.startswith("```"):
                resp = "```json\n" + resp + "\n```"

            # system prompt: messages[0] if present, else canonical fallback
            msgs = d.get("messages") or []
            if msgs and msgs[0].get("role") == "system":
                system = msgs[0]["content"]
            else:
                system = PC_AGENT_SYSTEM_FALLBACK

            img = image_path(args.image_root, d["image_tar"])
            if args.check_images and not os.path.exists(img):
                n_noimg += 1
                continue

            rec = {
                "id": d["sample_id"],
                "system": system,
                "image": img,
                "conversations": [
                    {"role": "user", "content": d["prompt"] + "\n<image>"},
                    {"role": "assistant", "content": resp},
                ],
                "domain": d.get("domain"),
                "gt_action": d.get("action"),
                "task_completed": d.get("task_completed"),
                "this_step_correct": d.get("this_step_correct"),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_hit += 1
            if n_hit % 50000 == 0:
                print(f"  joined {n_hit} ...", flush=True)

    print(f"DONE {args.output}: in={n_in} written={n_hit} "
          f"miss={n_miss} no_image={n_noimg}", flush=True)
    if args.check_images and n_noimg:
        sys.exit(2)


if __name__ == "__main__":
    main()
