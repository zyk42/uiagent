#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert AgentNet (OpenCUA) win/mac trajectories to PC-Agent response format
for distilling PC-Agent-style answers with GUI-Owl-1.5-8B-Instruct.

Output: one JSONL row per atomic action, with
  - prompt    : built by PC-Agent's own get_action_prompt() (no-perception mode,
                so coordinates are normalized to [0, 999], matching run.py's
                denormalization `x * width / 1000`)
  - response  : ```json {"Thought": ..., "Action": ..., "Summary": ...} ```
  - messages  : OpenAI multi-modal chat format, ready for SFT
  - plus quality labels / raw code / aux pixel-coordinate action
"""

import argparse
import ast
import importlib.util
import json
import os
import re
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# import PC-Agent's prompt builder without importing the whole package
# ---------------------------------------------------------------------------
PCAGENT_PROMPT_PATH = "/root/MobileAgent-main/PC-Agent/PCAgent/prompt_qwen.py"
spec = importlib.util.spec_from_file_location("pcagent_prompt", PCAGENT_PROMPT_PATH)
pcagent_prompt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcagent_prompt)
get_action_prompt = pcagent_prompt.get_action_prompt

SYSTEM_PROMPT = ("You are a helpful AI PC operating assistant. "
                 "You need to help me operate the PC to complete the user's instruction.")

KEY_MAP = {"cmd": "command", "win": "win", "ctrl": "ctrl", "alt": "alt",
           "shift": "shift", "option": "option"}


def norm999(v):
    return int(round(float(v) * 999))


def to_px(v, dim):
    return int(round(float(v) * dim))


def clean_text(t):
    # strip code fences: they would break PC-Agent's ```json ... ``` extraction
    return str(t).replace("```", "'''").replace("\n", " ").replace("\r", " ").strip()


# ---------------------------------------------------------------------------
# pyautogui code parsing
# ---------------------------------------------------------------------------
def parse_call(line):
    """Parse one line of pyautogui/computer code -> (func, args, kwargs) or None."""
    line = line.strip()
    if not line:
        return None
    try:
        tree = ast.parse(line, mode="eval")
        call = tree.body
        if not isinstance(call, ast.Call):
            return None
        func = ast.unparse(call.func)  # e.g. pyautogui.click
        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        return func, args, kwargs
    except Exception:
        # fallback: rough regex
        m = re.match(r"([\w.]+)\((.*)\)", line)
        if not m:
            return None
        return m.group(1), [], {}


def get_xy(args, kwargs):
    x = kwargs.get("x", args[0] if len(args) > 0 else None)
    y = kwargs.get("y", args[1] if len(args) > 1 else None)
    if x is None or y is None:
        return None
    return float(x), float(y)


class Atomic:
    """One PC-Agent action in both coordinate systems."""
    def __init__(self, action_norm, action_px, summary):
        self.action_norm = action_norm
        self.action_px = action_px
        self.summary = summary


def code_to_atomics(code, last_pos):
    """
    Convert one step's (possibly multi-line) pyautogui code into a list of
    Atomic actions in PC-Agent vocabulary. Returns (atomics, new_last_pos, stats).
    Coordinates in code are normalized [0,1]; we emit [0,999] and placeholder
    pixel coords (converted later once resolution is known -> store [0,1] too).

    To keep it simple we emit actions with [0,1] floats inside and let the
    caller render both coordinate systems.
    """
    stats = Counter()
    atomics = []  # list of (kind, payload)
    lines = [l for l in code.strip().split("\n") if l.strip()]
    i = 0
    pos = last_pos  # last known pointer position in [0,1]
    while i < len(lines):
        parsed = parse_call(lines[i])
        if parsed is None:
            stats["unparsed_line"] += 1
            i += 1
            continue
        func, args, kwargs = parsed

        if func == "pyautogui.moveTo":
            xy = get_xy(args, kwargs)
            if xy:
                pos = xy
            # look-ahead: dragTo / scroll handled below consume this move
            stats["moveTo"] += 1
            i += 1
            continue

        if func == "pyautogui.dragTo":
            xy = get_xy(args, kwargs)
            start = pos
            if start and xy:
                atomics.append(("drag", (start, xy)))
                pos = xy
            stats["drag"] += 1
            i += 1
            continue

        if func in ("pyautogui.click", "pyautogui.doubleClick",
                    "pyautogui.rightClick", "pyautogui.middleClick",
                    "computer.tripleClick", "pyautogui.tripleClick"):
            xy = get_xy(args, kwargs)
            if xy is None:
                xy = pos
            if xy is None:
                stats["click_no_coord"] += 1
                i += 1
                continue
            button = kwargs.get("button", "left")
            clicks = kwargs.get("clicks", 1)
            if func == "pyautogui.doubleClick" or clicks == 2:
                kind = "double_tap"
            elif func == "pyautogui.rightClick" or button == "right":
                kind = "right_tap"
            elif func == "pyautogui.middleClick" or button == "middle":
                kind = "middle_tap"
            elif "tripleClick" in func or clicks == 3:
                kind = "triple_tap"
            else:
                kind = "tap"
            atomics.append((kind, xy))
            pos = xy
            stats[kind] += 1
            i += 1
            continue

        if func == "pyautogui.write" or func == "pyautogui.typewrite":
            text = kwargs.get("message", args[0] if args else "")
            text = clean_text(text)
            # consume a following press('enter') / press('return')
            j = i + 1
            if j < len(lines):
                p2 = parse_call(lines[j])
                if p2 and p2[0] == "pyautogui.press":
                    k2 = p2[2].get("keys", p2[1][0] if p2[1] else None)
                    if k2 in ("enter", "return"):
                        i = j  # skip it
            if text == "":
                stats["type_empty_dropped"] += 1
                i += 1
                continue
            atomics.append(("type", (pos, text)))
            stats["type"] += 1
            i += 1
            continue

        if func == "pyautogui.press":
            key = kwargs.get("keys", args[0] if args else None)
            presses = int(kwargs.get("presses", 1))
            if key is None:
                i += 1
                continue
            atomics.append(("press", (str(key).lower(), presses)))
            stats["press"] += 1
            i += 1
            continue

        if func == "pyautogui.hotkey":
            keys = args[0] if args and isinstance(args[0], (list, tuple)) else list(args)
            keys = [KEY_MAP.get(str(k).lower(), str(k).lower()) for k in keys]
            if len(keys) == 1:
                atomics.append(("press", (keys[0], 1)))
            else:
                atomics.append(("shortcut", tuple(keys)))
            stats["hotkey"] += 1
            i += 1
            continue

        if func in ("pyautogui.scroll", "pyautogui.hscroll"):
            clicks = args[0] if args else kwargs.get("clicks", 0)
            kind = "hscroll" if "hscroll" in func else "scroll"
            atomics.append((kind, (pos, int(clicks))))
            stats[kind] += 1
            i += 1
            continue

        if func == "computer.terminate":
            status = kwargs.get("status", args[0] if args else "success")
            atomics.append(("stop", str(status)))
            stats["stop"] += 1
            i += 1
            continue

        stats["unknown:" + func] += 1
        i += 1

    return atomics, pos, stats


def render_action(kind, payload, coord_fn):
    """Render one atomic into a PC-Agent action string. coord_fn: (x,y)->(X,Y)."""
    if kind == "tap":
        x, y = coord_fn(*payload)
        return f"Tap ({x}, {y})"
    if kind == "double_tap":
        x, y = coord_fn(*payload)
        return f"Double Tap ({x}, {y})"
    if kind == "right_tap":
        x, y = coord_fn(*payload)
        return f"Right Tap ({x}, {y})"
    if kind == "middle_tap":
        x, y = coord_fn(*payload)
        return f"Middle Tap ({x}, {y})"
    if kind == "triple_tap":
        x, y = coord_fn(*payload)
        return f"Triple Tap ({x}, {y})"
    if kind == "drag":
        (x1, y1), (x2, y2) = payload
        X1, Y1 = coord_fn(x1, y1)
        X2, Y2 = coord_fn(x2, y2)
        return f"Drag ({X1}, {Y1}), ({X2}, {Y2})"
    if kind == "type":
        pos, text = payload
        if pos is not None:
            x, y = coord_fn(*pos)
            return f'Type ({x}, {y}), ("{text}")'
        return f'Type ("{text}")'
    if kind == "press":
        key, presses = payload
        if presses > 1:
            return f"Press ({key}) [x{presses}]"
        return f"Press ({key})"
    if kind == "shortcut":
        return "Shortcut (" + ", ".join(payload) + ")"
    if kind == "scroll":
        pos, clicks = payload
        if pos is not None:
            x, y = coord_fn(*pos)
            return f"Scroll ({x}, {y}), ({clicks})"
        return f"Scroll ({clicks})"
    if kind == "hscroll":
        pos, clicks = payload
        if pos is not None:
            x, y = coord_fn(*pos)
            return f"HScroll ({x}, {y}), ({clicks})"
        return f"HScroll ({clicks})"
    if kind == "stop":
        return "Stop"
    return None


def atomic_summary(kind, payload, coord_fn):
    """Short English summary for one atomic action."""
    if kind == "tap":
        x, y = coord_fn(*payload)
        return f"Tap at ({x}, {y})"
    if kind == "double_tap":
        x, y = coord_fn(*payload)
        return f"Double tap at ({x}, {y})"
    if kind == "right_tap":
        x, y = coord_fn(*payload)
        return f"Right tap at ({x}, {y})"
    if kind == "middle_tap":
        x, y = coord_fn(*payload)
        return f"Middle tap at ({x}, {y})"
    if kind == "triple_tap":
        x, y = coord_fn(*payload)
        return f"Triple tap at ({x}, {y})"
    if kind == "drag":
        (x1, y1), (x2, y2) = payload
        X1, Y1 = coord_fn(x1, y1)
        X2, Y2 = coord_fn(x2, y2)
        return f"Drag from ({X1}, {Y1}) to ({X2}, {Y2})"
    if kind == "type":
        pos, text = payload
        short = text if len(text) <= 40 else text[:37] + "..."
        if pos is not None:
            x, y = coord_fn(*pos)
            return f'Type "{short}" at ({x}, {y}) and press Enter'
        return f'Type "{short}" and press Enter'
    if kind == "press":
        key, presses = payload
        if presses > 1:
            return f"Press the {key} key {presses} times"
        return f"Press the {key} key"
    if kind == "shortcut":
        return "Press shortcut " + "+".join(payload)
    if kind == "scroll":
        pos, clicks = payload
        direction = "up" if clicks > 0 else "down"
        return f"Scroll {direction} {abs(clicks)} clicks"
    if kind == "hscroll":
        pos, clicks = payload
        return f"Scroll horizontally {abs(clicks)} clicks"
    if kind == "stop":
        return "Stop the task"
    return ""


# ---------------------------------------------------------------------------
# main conversion
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agentnet_dir", default="/mnt/lisiqi23/internship/zhangyikang/uiagent/AgentNet")
    ap.add_argument("--out_dir", default="/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill")
    ap.add_argument("--input", default="agentnet_win_mac_18k.jsonl")
    ap.add_argument("--image_prefix", default="win_mac_images")
    ap.add_argument("--size_map", default=None,
                    help="task_id -> [actual_image_w, actual_image_h] json; "
                         "overrides meta_data resolution (images are usually resized)")
    ap.add_argument("--tar_index", default=None,
                    help="tar_index.json to attach image_tar=[shard,offset,size] per sample")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    size_map = {}
    if args.size_map:
        with open(args.size_map) as f:
            size_map = json.load(f)

    tar_index = {}
    if args.tar_index:
        with open(args.tar_index) as f:
            tar_index = json.load(f)

    # task_id -> (width, height, system)
    meta = {}
    with open(os.path.join(args.agentnet_dir, "meta_data_merged.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            meta[d["task_id"]] = (d.get("screen_width"), d.get("screen_height"),
                                  d.get("system"))

    out_name = args.input.replace(".jsonl", "") + "_pcagent.jsonl"
    out_path = os.path.join(args.out_dir, out_name)
    stats = Counter()
    n_traj = 0
    n_sample = 0

    with open(os.path.join(args.agentnet_dir, args.input)) as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            traj = json.loads(line)
            task_id = traj["task_id"]
            w, h, system = meta.get(task_id, (None, None, None))
            if task_id in size_map:
                w, h = size_map[task_id]  # actual image pixels (post-resize)
            n_traj += 1

            # per-trajectory state
            thought_hist, summary_hist, action_hist, reflection_hist = [], [], [], []
            last_pos = None  # pointer position in [0,1]
            prev_step_correct = None
            prev_summary, prev_action = "", ""

            for step in traj["traj"]:
                idx = step["index"]
                val = step["value"]
                thought = clean_text(val.get("thought", ""))
                nl_action = clean_text(val.get("action", ""))
                reflection = clean_text(val.get("reflection", ""))
                correct = val.get("last_step_correct", None)
                code = val.get("code", "")

                atomics, last_pos, astats = code_to_atomics(code, last_pos)
                stats.update(astats)
                if not atomics:
                    stats["empty_step"] += 1
                    continue

                for j, (kind, payload) in enumerate(atomics):
                    # error_flag: previous step's action was judged wrong/redundant
                    error_flag = bool(prev_step_correct is False) and j == 0
                    ls, la = (prev_summary, prev_action) if error_flag else ("", "")

                    prompt = get_action_prompt(
                        traj["instruction"], [], w or 1920, h or 1080,
                        thought_hist, summary_hist, action_hist, reflection_hist,
                        ls, la, "", "", error_flag, "", "")

                    coord_norm = lambda x, y: (norm999(x), norm999(y))
                    coord_px = (lambda x, y: (to_px(x, w), to_px(y, h))) if w else None

                    action_str = render_action(kind, payload, coord_norm)
                    action_px = render_action(kind, payload, coord_px) if coord_px else None
                    if action_str is None:
                        continue

                    if j == 0:
                        this_thought = thought
                        this_summary = nl_action if len(atomics) == 1 else \
                            (nl_action + f" (part 1: {atomic_summary(kind, payload, coord_norm)})")
                    else:
                        this_summary = atomic_summary(kind, payload, coord_norm)
                        this_thought = f"Following the plan, I will now {this_summary[0].lower() + this_summary[1:]}."

                    resp_obj = {"Thought": this_thought,
                                "Action": action_str,
                                "Summary": this_summary}
                    response = "```json\n" + json.dumps(resp_obj, ensure_ascii=False) + "\n```"

                    image_rel = f"{args.image_prefix}/{step['image']}"
                    record = {
                        "sample_id": f"{task_id}_s{idx}_{j}",
                        "task_id": task_id,
                        "system": system,
                        "domain": traj.get("domain", ""),
                        "instruction": traj["instruction"],
                        "natural_language_task": traj.get("natural_language_task", ""),
                        "step_index": idx,
                        "sub_index": j,
                        "is_continuation": j > 0,
                        "image": image_rel,
                        "image_width": w,   # actual image pixels (after AgentNet resize)
                        "image_height": h,
                        "coord_format": "normalized_0_999",
                        "prompt": prompt,
                        "response": response,
                        "thought": this_thought,
                        "action": action_str,
                        "action_pixel": action_px,
                        "summary": this_summary,
                        "observation": clean_text(val.get("observation", "")),
                        "raw_code": code,
                        "reflection_of_this_step": reflection,
                        "this_step_correct": correct,
                        "this_step_redundant": val.get("last_step_redundant", None),
                        "task_completed": traj.get("task_completed", None),
                        "terminated_with_fail": (kind == "stop" and payload == "fail"),
                        "alignment_score": traj.get("alignment_score"),
                        "efficiency_score": traj.get("efficiency_score"),
                        "task_difficulty": traj.get("task_difficulty"),
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "image": image_rel},
                            ]},
                            {"role": "assistant", "content": response},
                        ],
                    }
                    if tar_index:
                        record["image_tar"] = tar_index[step["image"]]  # [shard, offset, size]
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_sample += 1

                    # update per-atomic history
                    thought_hist.append(this_thought)
                    summary_hist.append(this_summary)
                    action_hist.append(action_str)
                    # reflection attaches to the LAST atomic of the original step
                    reflection_hist.append(reflection if j == len(atomics) - 1 else "")

                prev_step_correct = correct
                prev_summary = summary_hist[-1] if summary_hist else ""
                prev_action = action_hist[-1] if action_hist else ""

            if n_traj % 2000 == 0:
                print(f"  {n_traj} trajectories -> {n_sample} samples", flush=True)

    stats_out = {k: v for k, v in sorted(stats.items(), key=lambda x: -x[1])}
    summary = {"trajectories": n_traj, "samples": n_sample, "action_stats": stats_out}
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
