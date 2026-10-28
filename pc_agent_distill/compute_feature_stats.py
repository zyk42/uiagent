#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute real statistics over the distilled PC-Agent dataset for the
"2.1 PC GUI Agent 推理任务与输出特征" document.
"""
import collections
import io
import json
import math
import random
import re
import struct
import sys

import numpy as np
from PIL import Image

BASE = '/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill'
MODEL = '/mnt/lisiqi23/internship/zhangyikang/uiagent/GUI-Owl-1.5-32B-Instruct'
SUBSAMPLE = 30000
random.seed(0)


def pct(vals, ps=(50, 90, 95, 99)):
    a = np.asarray(vals, dtype=np.float64)
    out = {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


def verb_of(action):
    if not action:
        return None
    m = re.match(r"([A-Za-z ]+?)\s*\(", action)
    if m:
        return m.group(1).strip()
    return action.strip()


# ---------- 1. load converted samples: prompt, image size, history ----------
print("loading converted samples...", flush=True)
conv = {}  # sample_id -> dict
for name in ["agentnet_win_mac_pcagent.jsonl", "agentnet_ubuntu_5k_pcagent.jsonl"]:
    with open(f"{BASE}/{name}") as f:
        for line in f:
            d = json.loads(line)
            conv[d["sample_id"]] = {
                "prompt": d["prompt"],
                "w": d["image_width"],
                "h": d["image_height"],
                "image_tar": d["image_tar"],
                "tar_dir": "win_mac_tar" if "win_mac" in name else "ubuntu_tar",
                "gt_action": d["action"],
                "hist_steps": d["prompt"].count("Step-"),
                "task_completed": d.get("task_completed"),
                "this_step_correct": d.get("this_step_correct"),
                "is_continuation": d.get("is_continuation"),
            }
print(f"  {len(conv)} converted samples", flush=True)

# ---------- 2. load distill outputs ----------
print("loading distill outputs...", flush=True)
recs = []
for name in ["distill_32b_dedup.jsonl", "distill_ubuntu_32b_dedup.jsonl"]:
    with open(f"{BASE}/{name}") as f:
        for line in f:
            r = json.loads(line)
            if r.get("parse_ok") and r["sample_id"] in conv:
                recs.append(r)
print(f"  {len(recs)} parse_ok distill records joined", flush=True)

stats = {}
stats["n_records"] = len(recs)

# ---------- 3. action verb distribution (full) ----------
model_verbs = collections.Counter(verb_of(r["model_action"]) for r in recs)
gt_verbs = collections.Counter(verb_of(conv[r["sample_id"]]["gt_action"]) for r in recs)
stats["model_verb_dist"] = model_verbs.most_common()
stats["gt_verb_dist"] = gt_verbs.most_common()

def entropy(counter):
    tot = sum(counter.values())
    return -sum(c / tot * math.log2(c / tot) for c in counter.values())
stats["model_verb_entropy_bits"] = entropy(model_verbs)
stats["gt_verb_entropy_bits"] = entropy(gt_verbs)

# ---------- 4. response length (full, from API usage) ----------
stats["completion_tokens"] = pct([r["completion_tokens"] for r in recs])
stats["prompt_tokens_api"] = pct([r["prompt_tokens"] for r in recs])

# ---------- 5. history steps (full) ----------
stats["history_steps"] = pct([conv[r["sample_id"]]["hist_steps"] for r in recs])

# ---------- 6. coordinate stats (full, from model actions) ----------
xs, ys, n_coord_actions = [], [], 0
key_counter = collections.Counter()
type_text_lens = []
for r in recs:
    a = r["model_action"] or ""
    nums = re.findall(r"\((\d+),\s*(\d+)\)", a)
    if nums and verb_of(a) in ("Tap", "Double Tap", "Right Tap", "Triple Tap", "Middle Tap", "Type"):
        x, y = int(nums[0][0]), int(nums[0][1])
        xs.append(x); ys.append(y); n_coord_actions += 1
    if verb_of(a) == "Shortcut":
        keys = re.findall(r"\(([^)]+)\)", a)
        if keys:
            key_counter[keys[0].lower()] += 1
    elif verb_of(a) == "Press":
        m = re.search(r"\(([^)]+)\)", a)
        if m:
            key_counter["press:" + m.group(1).lower()] += 1
    elif verb_of(a) == "Type":
        m = re.findall(r'\("((?:[^"\\]|\\.)*)"\)', a)
        if m:
            type_text_lens.append(len(m[-1]))

stats["coord_actions"] = n_coord_actions
stats["coord_x"] = pct(xs)
stats["coord_y"] = pct(ys)
stats["coord_unique_x"] = len(set(xs))
stats["coord_unique_y"] = len(set(ys))
x_c = collections.Counter(xs); y_c = collections.Counter(ys)
stats["coord_x_entropy_bits"] = entropy(x_c)
stats["coord_y_entropy_bits"] = entropy(y_c)
stats["top_keys"] = key_counter.most_common(20)
stats["type_text_char_len"] = pct(type_text_lens)

# ---------- 7. tokenizer-based field shares (subsample) ----------
print("tokenizing subsample...", flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)

sub = random.sample(recs, min(SUBSAMPLE, len(recs)))
thoughts = [r["model_thought"] or "" for r in sub]
actions = [r["model_action"] or "" for r in sub]
summaries = [r["model_summary"] or "" for r in sub]
prompts = [conv[r["sample_id"]]["prompt"] for r in sub]
responses = [r["model_output"] for r in sub]

def tok_lens(texts, bs=1000):
    out = []
    for i in range(0, len(texts), bs):
        out.extend(len(x) for x in tok(texts[i:i + bs], add_special_tokens=False)["input_ids"])
    return out

th_t = tok_lens(thoughts)
ac_t = tok_lens(actions)
su_t = tok_lens(summaries)
pr_t = tok_lens(prompts)
resp_t = tok_lens(responses)

stats["field_tokens"] = {
    "thought": pct(th_t),
    "action": pct(ac_t),
    "summary": pct(su_t),
    "response_full": pct(resp_t),
    "prompt_text": pct(pr_t),
}
tot_th, tot_ac, tot_su = sum(th_t), sum(ac_t), sum(su_t)
tot = tot_th + tot_ac + tot_su
stats["field_token_share"] = {
    "thought": tot_th / tot, "action": tot_ac / tot, "summary": tot_su / tot,
}

# prompt = text tokens + vision tokens; estimate vision via api: prompt_tokens_api - text - overhead
api_prompt = np.array([r["prompt_tokens"] for r in sub])
text_tok = np.array(pr_t)
vision_est = api_prompt - text_tok
stats["vision_tokens_est_raw"] = pct(vision_est.tolist())
# overhead per request (chat template wrappers + image placeholders), estimate via mode of small residual
stats["vision_plus_overhead_median"] = float(np.median(vision_est))

# ---------- 8. action internals: verb / coords / text token shares (subsample) ----------
verb_tokens, coord_tokens, text_tokens, key_tokens, other_tokens = [], [], [], [], []
for a in actions:
    v = verb_of(a)
    if not v:
        continue
    rest = a[len(v):].strip()
    coords = re.findall(r"\d+", rest)
    texts = re.findall(r'"((?:[^"\\]|\\.)*)"', rest)
    coord_str = " ".join(coords)
    text_str = " ".join(texts)
    verb_tokens.append(len(tok(v, add_special_tokens=False)["input_ids"]))
    coord_tokens.append(len(tok(coord_str, add_special_tokens=False)["input_ids"]) if coord_str else 0)
    text_tokens.append(len(tok(text_str, add_special_tokens=False)["input_ids"]) if text_str else 0)
    key_tokens.append(0)
stats["action_component_tokens_mean"] = {
    "verb": float(np.mean(verb_tokens)),
    "coords_digits": float(np.mean(coord_tokens)),
    "quoted_text": float(np.mean(text_tokens)),
}

# ---------- 9. vision tokens: exact measurement per unique size ----------
print("measuring vision tokens per unique image size...", flush=True)
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained(MODEL)

size_to_sample = {}
for r in recs:
    c = conv[r["sample_id"]]
    size_to_sample.setdefault((c["w"], c["h"]), r["sample_id"])
print(f"  {len(size_to_sample)} unique image sizes", flush=True)

_handles = {}
def load_img(tar_dir, image_tar):
    shard, off, size = image_tar
    key = (tar_dir, shard)
    if key not in _handles:
        _handles[key] = open(f"{BASE}/{tar_dir}/{shard}", "rb")
    f = _handles[key]
    f.seek(off)
    return Image.open(io.BytesIO(f.read(size)))

size_tokens = {}
sizes = list(size_to_sample.items())
for i in range(0, len(sizes), 50):
    batch_imgs, batch_sizes = [], []
    for (w, h), sid in sizes[i:i + 50]:
        c = conv[sid]
        batch_imgs.append(load_img(c["tar_dir"], c["image_tar"]))
        batch_sizes.append((w, h))
    out = proc(images=batch_imgs, return_tensors=None)
    grids = out["image_grid_thw"]
    merge = 2
    for (w, h), g in zip(batch_sizes, grids):
        t, gh, gw = int(g[0]), int(g[1]), int(g[2])
        size_tokens[f"{w}x{h}"] = t * gh * gw // (merge * merge)
    if (i // 50) % 10 == 0:
        print(f"  {i + len(batch_sizes)}/{len(sizes)} sizes measured", flush=True)

vtoks = [size_tokens[f"{conv[r['sample_id']]['w']}x{conv[r['sample_id']]['h']}"] for r in recs]
stats["vision_tokens_exact"] = pct(vtoks)
json.dump(size_tokens, open(f"{BASE}/vision_tokens_by_size.json", "w"))

# per-request input total = prompt text tokens + vision tokens
sub_v = np.array([size_tokens[f"{conv[r['sample_id']]['w']}x{conv[r['sample_id']]['h']}"] for r in sub])
stats["input_total_tokens_est"] = pct((text_tok + sub_v).tolist())

json.dump(stats, open(f"{BASE}/feature_stats.json", "w"), ensure_ascii=False, indent=2)
print("DONE -> feature_stats.json", flush=True)
print(json.dumps({k: v for k, v in stats.items() if k not in ("model_verb_dist", "gt_verb_dist", "top_keys")},
                 ensure_ascii=False, indent=2)[:3000])
