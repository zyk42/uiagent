#!/usr/bin/env python3
"""Build the SpecForge VLM tokenize cache WITHOUT launching training.

Reuses VLMServerInputAdapter.prepare_prompts with a fabricated config so the
cache file (keyed by data path/size/mtime + target model + max_length) is
bit-identical to what `specforge train` would build at startup. Once this has
run, training startup skips tokenization.

Usage:
  python3 precache.py --data data_all.jsonl \
      --target-model /mnt/.../checkpoints/RL/final/ \
      --cache-dir ./cache --max-length 8192 --num-proc 64
"""

import argparse
import os
from types import SimpleNamespace

from specforge.algorithms.dflash.vlm_input_adapter import VLMServerInputAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--num-proc", type=int, default=64)
    args = ap.parse_args()

    config = SimpleNamespace(
        data=SimpleNamespace(
            train_data_path=os.path.abspath(args.data),
            max_length=args.max_length,
            cache_dir=os.path.abspath(args.cache_dir),
            build_dataset_num_proc=args.num_proc,
        ),
        model=SimpleNamespace(
            target_model_path=args.target_model,
            trust_remote_code=True,
            cache_dir=None,
        ),
    )

    adapter = VLMServerInputAdapter(config)
    prompts = adapter.prepare_prompts(config, None, draft_config=None)
    print(f"PRECACHE DONE: {len(prompts)} prompts ready")


if __name__ == "__main__":
    main()
