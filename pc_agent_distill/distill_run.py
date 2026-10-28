#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distill PC-Agent-format answers from GUI-Owl-1.5-32B-Instruct (sglang server).

Reads agentnet_win_mac_pcagent.jsonl, sends each sample's (system + prompt +
image) to the server, and saves the model's response alongside the ground
truth. Resumable: sample_ids already present in the output file are skipped.

Progress: one line every 5 seconds to stdout (done/total, rate, ETA, fails),
plus a machine-readable status.json.
"""
import asyncio
import base64
import json
import os
import sys
import time

from openai import AsyncOpenAI

BASE = '/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill'
INPUT = os.environ.get('DISTILL_INPUT', f'{BASE}/agentnet_win_mac_pcagent.jsonl')
OUTPUT = os.environ.get('DISTILL_OUTPUT', f'{BASE}/distill_32b_raw.jsonl')
STATUS = os.environ.get('DISTILL_STATUS', f'{BASE}/distill_32b_status.json')
TAR_DIR = os.environ.get('DISTILL_TAR_DIR', f'{BASE}/win_mac_tar')

MODEL = 'default'
URL = 'http://localhost:30000/v1'
MAX_TOKENS = 2048
TEMPERATURE = 0.0
CONCURRENCY = 256
MAX_RETRY = 3

SYSTEM_PROMPT = ("You are a helpful AI PC operating assistant. "
                 "You need to help me operate the PC to complete the user's instruction.")

_tar_handles = {}


def load_b64(image_tar):
    shard, off, size = image_tar
    if shard not in _tar_handles:
        _tar_handles[shard] = open(f'{TAR_DIR}/{shard}', 'rb')
    f = _tar_handles[shard]
    f.seek(off)
    return base64.b64encode(f.read(size)).decode()


def parse_output(out):
    try:
        obj = json.loads(out.split('```json')[-1].split('```')[0].strip().strip('`'))
        return obj.get('Thought'), obj.get('Action'), obj.get('Summary'), True
    except Exception:
        return None, None, None, False


async def do_one(client, sem, d, out_f, lock, stats):
    async with sem:
        t0 = time.time()
        rec = None
        for attempt in range(MAX_RETRY):
            try:
                img_b64 = await asyncio.get_event_loop().run_in_executor(
                    None, load_b64, d['image_tar'])
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": d['prompt']},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        ]},
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    timeout=180,
                )
                out = resp.choices[0].message.content
                thought, action, summary, ok = parse_output(out)
                rec = {
                    "sample_id": d['sample_id'],
                    "model_output": out,
                    "model_thought": thought,
                    "model_action": action,
                    "model_summary": summary,
                    "parse_ok": ok,
                    "gt_action": d['action'],
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "latency_s": round(time.time() - t0, 2),
                }
                break
            except Exception as e:
                if attempt == MAX_RETRY - 1:
                    rec = {"sample_id": d['sample_id'], "error": str(e)[:300],
                           "parse_ok": False}
                    stats['fail'] += 1
                else:
                    await asyncio.sleep(2 * (attempt + 1))
        async with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out_f.flush()
            stats['done'] += 1
            if rec.get('parse_ok'):
                stats['ok'] += 1


async def stats_reporter(stats, total, t_start):
    while True:
        await asyncio.sleep(5)
        done = stats['done']
        rate = done / max(time.time() - t_start, 1e-9)
        eta_h = (total - done) / rate / 3600 if rate > 0 else float('inf')
        line = (f"[{time.strftime('%H:%M:%S')}] {done}/{total} "
                f"({done/total*100:.1f}%) | {rate:.2f} req/s | "
                f"parse_ok={stats['ok']} fail={stats['fail']} | ETA {eta_h:.1f}h")
        print(line, flush=True)
        with open(STATUS, 'w') as f:
            json.dump({"done": done, "total": total, "rate": rate,
                       "eta_hours": eta_h, "parse_ok": stats['ok'],
                       "fail": stats['fail'],
                       "updated": time.strftime('%F %T')}, f)
        if done >= total:
            break


async def main():
    # resume: collect done sample_ids
    done_ids = set()
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)['sample_id'])
                except Exception:
                    pass
    print(f'resuming: {len(done_ids)} already done', flush=True)

    todo = []
    with open(INPUT) as f:
        for line in f:
            d = json.loads(line)
            if d['sample_id'] not in done_ids:
                todo.append({"sample_id": d['sample_id'], "prompt": d['prompt'],
                             "image_tar": d['image_tar'], "action": d['action']})
    total = len(todo) + len(done_ids)
    print(f'todo: {len(todo)}, total: {total}', flush=True)

    client = AsyncOpenAI(base_url=URL, api_key='EMPTY',
                         max_retries=0, timeout=180)
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    stats = {'done': len(done_ids), 'ok': 0, 'fail': 0}
    stats['done'] = len(done_ids)

    out_f = open(OUTPUT, 'a', buffering=1)
    t_start = time.time()
    reporter = asyncio.create_task(stats_reporter(stats, total, t_start))

    # stats['done'] counts include previously done; bump via tasks only for new
    base_done = len(done_ids)
    stats['done'] = base_done

    BATCH = 2000
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        await asyncio.gather(*[do_one(client, sem, d, out_f, lock, stats)
                               for d in batch])
    await asyncio.sleep(6)
    reporter.cancel()
    out_f.close()
    print(f"ALL DONE. done={stats['done']} fail={stats['fail']}", flush=True)


if __name__ == '__main__':
    asyncio.run(main())
