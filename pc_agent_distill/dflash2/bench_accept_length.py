# coding=utf-8
"""Measure DFlash2 draft accept length on training-distribution GUI samples.

Renders each sample with the target's chat template (image attached to its
original turn), sends native /generate requests, and aggregates the server's
spec_accept_length / spec_accept_rate.

Usage:
  python bench_accept_length.py --server http://127.0.0.1:30000 \
      --tokenizer /path/to/target --data data_1000.jsonl --num-samples 200
"""

import argparse
import asyncio
import json
import logging

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("bench")


def build_messages(rec):
    """Mirror the training-time assembly; image only in the turn that holds it."""
    msgs = []
    if rec.get("system"):
        msgs.append({"role": "system", "content": rec["system"]})
    convs = rec["conversations"]
    if convs and convs[-1]["role"] == "assistant":
        convs = convs[:-1]  # the last assistant turn is what we generate
    for c in convs:
        content = c.get("content") or ""
        if c["role"] == "user" and "<image>" in content and rec.get("image"):
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content.replace("<image>", "").strip()},
                        {"type": "image", "image": rec["image"]},
                    ],
                }
            )
        else:
            msgs.append({"role": c["role"], "content": content})
    return msgs


async def one_request(session, args, tokenizer, rec, sem, out):
    async with sem:
        text = tokenizer.apply_chat_template(
            build_messages(rec), tokenize=False, add_generation_prompt=True
        )
        body = {
            "text": text,
            "image_data": [rec["image"]] if rec.get("image") else None,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
            },
        }
        try:
            async with session.post(
                f"{args.server}/generate", json=body,
                timeout=aiohttp.ClientTimeout(total=900),
            ) as resp:
                data = await resp.json()
            mi = data.get("meta_info") or {}
            out.append(
                {
                    "completion_tokens": mi.get("completion_tokens", 0),
                    "verify_ct": mi.get("spec_verify_ct"),
                    "accept_length": mi.get("spec_accept_length"),
                    "accept_rate": mi.get("spec_accept_rate"),
                }
            )
        except Exception as exc:
            logger.warning("request failed: %s", str(exc)[:150])
            out.append(None)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:30000")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    records = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= args.num_samples:
                break
    logger.info("loaded %d samples", len(records))
    import time as _t; _t0 = _t.time()

    sem = asyncio.Semaphore(args.concurrency)
    out = []
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            *(
                one_request(session, args, tokenizer, r, sem, out)
                for r in records
            )
        )

    ok = [r for r in out if r and r["completion_tokens"] > 0]
    with_spec = [r for r in ok if r["verify_ct"]]
    _elapsed = _t.time() - _t0
    _total_all = sum(r["completion_tokens"] for r in ok)
    print("\n================ RESULT ================")
    print(f"requests ok: {len(ok)}/{len(records)}")
    print(f"wall time: {_elapsed:.1f}s, total completion tokens: {_total_all}, agg throughput: {_total_all/_elapsed:.1f} tok/s")
    if with_spec:
        total_tokens = sum(r["completion_tokens"] for r in with_spec)
        total_verify = sum(r["verify_ct"] for r in with_spec)
        acc_lens = sorted(r["accept_length"] for r in with_spec)
        acc_rates = sorted(r["accept_rate"] for r in with_spec if r["accept_rate"] is not None)
        n = len(acc_lens)
        print(f"samples with spec stats: {n}")
        print(f"avg accept length (token-weighted): {total_tokens / total_verify:.3f}")
        print(f"avg accept length (per-request mean): {sum(acc_lens) / n:.3f}")
        print(f"accept length p50/p90: {acc_lens[n // 2]:.2f} / {acc_lens[int(n * 0.9)]:.2f}")
        if acc_rates:
            print(f"avg accept rate: {sum(acc_rates) / len(acc_rates):.3f}")
    else:
        print("no spec stats in meta_info — is speculative decoding active?")
    print("========================================")


if __name__ == "__main__":
    asyncio.run(main())
