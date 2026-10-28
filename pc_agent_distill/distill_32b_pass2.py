#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distillation pass 2: re-sample GUI-Owl-1.5-32B-Instruct answers with the
model's official sampling config (T=0.6, top_p=0.95, top_k=40) so each prompt
gets a SECOND, different sample of the target distribution.

Differences from distill_32b.py (pass 1):
  - temperature 0.0 -> 0.6 + top_p/top_k (pass 1 was deterministic; a pass-2
    at T=0 would be byte-identical and useless)
  - input/output/tar paths parameterized via CLI (win_mac and ubuntu sets)
  - input deduped by sample_id (win_mac has 5,370 duplicated lines; pass 1 was
    also trained deduped)

Resumable: sample_ids already present in the output file are skipped.
"""

import argparse
import asyncio
import base64
import json
import os
import time

from openai import AsyncOpenAI

MODEL = 'default'
URL = 'http://localhost:30000/v1'
MAX_TOKENS = 2048
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 40
CONCURRENCY = 256
MAX_RETRY = 3

SYSTEM_PROMPT = ("You are a helpful AI PC operating assistant. "
                 "You need to help me operate the PC to complete the user's instruction.")

_tar_handles = {}


def load_b64(tar_dir, image_tar):
    shard, off, size = image_tar
    key = (tar_dir, shard)
    if key not in _tar_handles:
        _tar_handles[key] = open(f'{tar_dir}/{shard}', 'rb')
    f = _tar_handles[key]
    f.seek(off)
    return base64.b64encode(f.read(size)).decode()


def parse_output(out):
    try:
        obj = json.loads(out.split('```json')[-1].split('```')[0].strip().strip('`'))
        return obj.get('Thought'), obj.get('Action'), obj.get('Summary'), True
    except Exception:
        return None, None, None, False


async def do_one(client, sem, d, out_f, lock, stats, tar_dir):
    async with sem:
        t0 = time.time()
        rec = None
        for attempt in range(MAX_RETRY):
            try:
                img_b64 = await asyncio.get_event_loop().run_in_executor(
                    None, load_b64, tar_dir, d['image_tar'])
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
                    top_p=TOP_P,
                    extra_body={"top_k": TOP_K},
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


async def stats_reporter(stats, total, t_start, status_path):
    while True:
        await asyncio.sleep(5)
        done = stats['done']
        rate = done / max(time.time() - t_start, 1e-9)
        eta_h = (total - done) / rate / 3600 if rate > 0 else float('inf')
        line = (f"[{time.strftime('%H:%M:%S')}] {done}/{total} "
                f"({done/total*100:.1f}%) | {rate:.2f} req/s | "
                f"parse_ok={stats['ok']} fail={stats['fail']} | ETA {eta_h:.1f}h")
        print(line, flush=True)
        with open(status_path, 'w') as f:
            json.dump({"done": done, "total": total, "rate": rate,
                       "eta_hours": eta_h, "parse_ok": stats['ok'],
                       "fail": stats['fail'],
                       "updated": time.strftime('%F %T')}, f)
        if done >= total:
            break


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tar-dir", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N todo")
    args = ap.parse_args()

    done_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)['sample_id'])
                except Exception:
                    pass
    print(f'resuming: {len(done_ids)} already done', flush=True)

    todo = []
    seen = set()
    with open(args.input) as f:
        for line in f:
            d = json.loads(line)
            sid = d['sample_id']
            if sid in seen or sid in done_ids:
                continue
            seen.add(sid)
            todo.append({"sample_id": sid, "prompt": d['prompt'],
                         "image_tar": d['image_tar'], "action": d['action']})
    if args.limit:
        todo = todo[:args.limit]
    total = len(todo) + len(done_ids)
    print(f'todo: {len(todo)}, total: {total}', flush=True)

    client = AsyncOpenAI(base_url=URL, api_key='EMPTY',
                         max_retries=0, timeout=180)
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    stats = {'done': len(done_ids), 'ok': 0, 'fail': 0}

    out_f = open(args.output, 'a', buffering=1)
    t_start = time.time()
    reporter = asyncio.create_task(stats_reporter(stats, total, t_start, args.status))

    BATCH = 2000
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        await asyncio.gather(*[do_one(client, sem, d, out_f, lock, stats, args.tar_dir)
                               for d in batch])
    await asyncio.sleep(6)
    reporter.cancel()
    out_f.close()
    print(f"ALL DONE. done={stats['done']} fail={stats['fail']}", flush=True)


if __name__ == '__main__':
    asyncio.run(main())
